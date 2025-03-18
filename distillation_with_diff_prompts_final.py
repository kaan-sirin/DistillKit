import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
from dotenv import load_dotenv
from distillation_utils import LivePlotCallback

# PAINSTAKINGLY SLOW:   1%|          | 5/420 [23:07<31:59:44, 277.55s/it]  
# Changing batchsize from 1 to 2 

# Got the following error:
# torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.27 GiB. GPU 0 has a total capacity of 47.53 GiB of which 844.06 MiB is free. Including non-PyTorch memory, this process has 46.68 GiB memory in use. Of the allocated memory 41.21 GiB is allocated by PyTorch, and 5.15 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
#   1%|          | 2/210 [14:14<24:41:03, 427.23s/it]   

# Cannot use FlashAttention2 due to CUDA version                                                                                                            

# ASSUMPTIONS
# [] a chat template is not needed for this dataset

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

config = {
    "project_name": "distil-logits",
    "dataset": {
        "name": "nicher92/medqa-swe",
        "split": "train",
        "num_samples": 2500,
        "seed": 42,
    },
    "models": {
        "teacher": "meta-llama/Llama-3.2-3B-Instruct",
        "student": "meta-llama/Llama-3.2-1B-Instruct",
    },
    "tokenizer": {
        "max_length": 1024,
    },
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True,
        "remove_unused_columns": False,  # TODO: My additionImportant to keep teacher input columns
    },
    "distillation": {"temperature": 2.0, "alpha": 0.5, "method": "soft_targets"},
    "model_config": {"use_flash_attention": True},
    # "spectrum": {
    #     "layers_to_unfreeze": "/workspace/spectrum/snr_results_Qwen-Qwen2-1.5B_unfrozenparameters_50percent.yaml" # You can pass a spectrum yaml file here to freeze layers identified by spectrum.
    # }
}

# Set up environment
os.environ["WANDB_PROJECT"] = config["project_name"]
accelerator = Accelerator()
device = accelerator.device


# Load and preprocess dataset
dataset = (
    load_dataset(
        config["dataset"]["name"],
        config["dataset"]["subset"],
        split=config["dataset"]["split"],
    )
    if config["dataset"].get("subset")
    else load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
)

# First take a subset of the dataset if specified, then shuffle
if "num_samples" in config["dataset"]:
    dataset = dataset.select(range(config["dataset"]["num_samples"]))
dataset = dataset.shuffle(seed=config["dataset"]["seed"])

# Load tokenizers

student_tokenizer = AutoTokenizer.from_pretrained(
    config["models"]["student"], token=HF_TOKEN
)

# TODO: DANGEROUS: Is this correct? It complains about pad_token not being set. Even when padding is not set to True in tokenizer.
# Inspired by https://discuss.huggingface.co/t/how-to-set-the-pad-token-for-meta-llama-llama-3-models/103418/5
# Answer starting with "I have attempted fine-tuning the LLaMA-3.1-8B..."
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = "<|finetune_right_pad_id|>"


def medqa_format(sample):
    try:
        prompt = (
            "Du är en medicinsk expert. Förklara steg för steg varför alternativ "
            f"{sample['answer']} är korrekt för följande fråga:\n\n"
        )

        question = f"{sample['question']}\n\n" f"{sample['options']}\n\n" f"Svar: "

        return {
            "prompt": prompt,
            "question": question,
        }
    except Exception as e:
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Error formatting sample: {e}")
        raise


# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(
    medqa_format,
    remove_columns=original_columns,
)


def tokenize_function(examples):
    try:

        prompt = student_tokenizer(
            examples["prompt"],
            return_attention_mask=True,
        )
        # <|begin_of_text|>Du är en medicinsk expert. Förklara...
        # len(prompt["input_ids"]) = 33 (with <|begin_of_text|> token)
        # len(prompt["attention_mask"]) = 33 (all ones)

        inputs = student_tokenizer(
            [p + q for p, q in zip(examples["prompt"], examples["question"])],
            return_attention_mask=True,
        )
        # Example: Du är en medicinsk expert. Förklara steg för steg varför alternativ A är korrekt för följande fråga:\n\nAnna, 32 år, söker vård..."
        # When tokenized: <|begin_of_text|>Du är en medicinsk expert. Förklara steg för steg varför alternativ A är korrekt för följande fråga: Anna, 32 år, söker vård..."
        # Attention mask all ones

        return {
            # Prompt inputs
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    except Exception as e:
        print(f"Error in tokenization: {e}")
        print(
            f"First prompt: {examples['prompt'][0] if examples['prompt'] else 'No prompts'}"
        )
        print(
            f"First question: {examples['question'][0] if examples['question'] else 'No questions'}"
        )
        raise


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=["question", "prompt"],
)

tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

print("Dataset preparation complete. Loading models...")

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

teacher_model = AutoModelForCausalLM.from_pretrained(
    config["models"]["teacher"], **model_kwargs
).to(device)
student_model = AutoModelForCausalLM.from_pretrained(
    config["models"]["student"], **model_kwargs
)


class LogitsTrainer(SFTTrainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        prompt = {
            "input_ids": inputs["prompt_input_ids"].to(device),
            "attention_mask": inputs["prompt_attention_mask"].to(device),
        }
        # Separate teacher and student inputs
        inputs = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
        }
        # ============ DEBUGGING ============
        # A batch of 2 examples, both with the same number of tokens, the shorter one is padded.
        # Despite me not setting padding='max_length' in the tokenizer.

        print(f"\n\nInputs 0: {len(inputs['input_ids'][0])}")
        print(
            f"Student tokenizer: {student_tokenizer.decode(inputs['input_ids'][0])[:100]}"
        )
        print(
            f"Number of zeros in attention mask: {len(inputs['input_ids'][0]) - sum(inputs['attention_mask'][0])}"
        )
        print(f"Inputs 1: {len(inputs['input_ids'][1])}")
        print(
            f"Student tokenizer: {student_tokenizer.decode(inputs['input_ids'][1])[:100]}"
        )
        print(
            f"Number of zeros in attention mask: {len(inputs['input_ids'][1]) - sum(inputs['attention_mask'][1])}"
        )
        # exit()
        # ============ DEBUGGING ============

        teacher_input_tokens = inputs["input_ids"].shape[-1]
        print(f"Teacher input tokens: {teacher_input_tokens}")

        # Teacher-generated answer
        answers = self.teacher_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
        )

        # ============ DEBUGGING ============
        print(
            f"\n\nAnswers: {answers.sequences[0][teacher_input_tokens:].shape}"
        )  # 512 tokens
        print(f"Tokens in answer 0: {answers.sequences[0][teacher_input_tokens:]}")
        print(
            f"Decoded: {student_tokenizer.decode(answers.sequences[0][teacher_input_tokens:][:100])}"
        )
        print(
            f"\n\nAnswers: {answers.sequences[1][teacher_input_tokens:].shape}"
        )  # 512 tokens
        print(f"Tokens in answer 1: {answers.sequences[1][teacher_input_tokens:]}")
        print(
            f"Decoded: {student_tokenizer.decode(answers.sequences[1][teacher_input_tokens:][:100])}"
        )
        # Sometimes answers are padded with eos_token's (128001) and sometimes not.
        # exit()
        # ============ DEBUGGING ============

        teacher_outputs = [answer[teacher_input_tokens:] for answer in answers]

        # ============ DEBUGGING ============
        # decoded_example = student_tokenizer.decode(
        #     teacher_outputs[0], skip_special_tokens=True
        # )
        # print(f"Example teacher output, decoded: {decoded_example}")

        #  , E) Förstäkringssjukdomen efter att du har fått ett gott svar från behandlingen.

        # Utlagning:
        # Förklaring:
        # Induktions-behandlingen är den första behandlingen som ges vid diagnosen av en cancerart. Den syftar dock inte till att eliminera alla cancer celler utan snarare till att minska antalet cancer cellerna så mycket som möjligt. Detta kan vara en kemoterapi eller strålbehandling. Induktions- behandlingen är vanligast hos lymfoma och leukemi men även andra typer av cancer kan behöva denna typ av behandling.
        # Det är viktigt att notera att induktion inte betyder att det är en underhålsbehandling eftersom dess mål är att minsuka cancer-celltalen istället för att bara hålla dem under kontroll. Dessutom är induktionsbehandlingen inte lika med observation av tillständet innan behandlingsnödvändigheten uppkommer (alternativ B). Induktionshandlingarna kan också inte ses över som en transplantationsbehandling (alternativa C), eftarsom de inte innehåller transplantering av nya stamcellsgrupper. Slutligen är induktionssjukan inte samma sak som en förstärkningsbehandling som ges efter att en god respons på initialbehandling har skett (alternativen A och D).
        # Alternativ E är rätt eft ersom induktiv behandling ofta ges för att öka effekten av en förskränkande behandling genom att minsuca cancer-celler mer effektivt. En indoktionssjuka kan ge patienten bättre chanser att fullständig remisjon ska kunna uppnas. Alternativ E beskriver alltså den korrekta definitionen av induktiva behandling.
        # ============ DEBUGGING ============

        # Creating inputs and attention masks for student and teacher
        # Teacher will be given the prompt + question + answer

        # The output from generate is everything the teacher needs for the forward pass.
        # print(f"Teacher inputs 0: {answers.sequences[0]}")
        print(f"\n\nTeacher inputs 0 shape: {answers.sequences[0].shape}")
        # print(f"Teacher inputs 1: {answers.sequences[1]}")
        print(f"Teacher inputs 1 shape: {answers.sequences[1].shape}")

        # I assume I can simply add ones to the original attention mask for the teacher inputs.
        # Sometimes the teacher inputs are padded with eos_token's (128001) and I'm not sure if they need to be masked.
        teacher_attention_masks_list = []
        for i, mask in enumerate(inputs["attention_mask"]):
            # The number of tokens in the generated answer for this example, I think it's always 512 with eventual padding.
            answer_length = answers.sequences[i][teacher_input_tokens:].shape[0]
            # Extend the original mask with ones for the answer tokens
            extended_mask = torch.cat(
                [mask, torch.ones(answer_length, device=mask.device, dtype=mask.dtype)]
            )
            teacher_attention_masks_list.append(extended_mask)

        # Stack the extended masks to create a batch
        teacher_attention_mask = torch.stack(teacher_attention_masks_list)

        student_input_ids_list = []
        student_attention_masks_list = []
        teacher_input_ids_list = []
        for i, seq in enumerate(answers.sequences):
            teacher_input_ids_list.append(seq)  # Use the complete sequence from generation
            # Last token of the CoT prompt should be set to <|begin_of_text|>
            # I think I can do this by replacing the last token of the prompt with <|begin_of_text|>
            # I need to get the index of the last token of the prompt first.
            last_prompt_token_index = len(prompt["input_ids"][i])-1 ## TODO: CONT FROM HERE
            print(f"\n\nLast prompt token index: {last_prompt_token_index}")
            print(f"Prompt input at index: {prompt['input_ids'][i][last_prompt_token_index]}: {student_tokenizer.decode(prompt['input_ids'][i][last_prompt_token_index])}")
            print(f"5 tokens before and after the last prompt token: {prompt['input_ids'][i][last_prompt_token_index-5:last_prompt_token_index+5]}: {student_tokenizer.decode(prompt['input_ids'][i][last_prompt_token_index-5:last_prompt_token_index+5])}")
            print(f"5 sequence tokens before and after the last prompt token: {seq[last_prompt_token_index-5:last_prompt_token_index+5]}: {student_tokenizer.decode(seq[last_prompt_token_index-5:last_prompt_token_index+5])}")
            
            # Set the last prompt token to <|begin_of_text|>, and the attention mask to 0 for the tokens coming before.
            seq[last_prompt_token_index] = student_tokenizer.bos_token_id
            print(f"Same sequence tokens after setting the last prompt token to <|begin_of_text|>")
            print(f"{seq[last_prompt_token_index-5:last_prompt_token_index+5]}: {student_tokenizer.decode(seq[last_prompt_token_index-5:last_prompt_token_index+5])}")
            student_input_ids_list.append(seq)
            # Copy the teacher attention mask
            new_attention_mask = teacher_attention_masks_list[i].clone()
            new_attention_mask[:last_prompt_token_index] = 0
            student_attention_masks_list.append(new_attention_mask)
            print(f"Prompt attention mask after setting prompt tokens to 0 (should be all zeros + 3 ones): {new_attention_mask[:last_prompt_token_index + 3]}")


        teacher_input_ids = torch.stack(teacher_input_ids_list)
        student_input_ids = torch.stack(student_input_ids_list)
        student_attention_masks = torch.stack(student_attention_masks_list)
        
        
        teacher_inputs = {
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
        }
        student_inputs = {
            "input_ids": student_input_ids,
            "attention_mask": student_attention_masks,
        }
        
        # Create labels for next-token prediction (shift input_ids right by 1)
        labels = student_inputs["input_ids"].clone()
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = -100  # Don't calculate loss for the last token prediction
        labels[labels == student_tokenizer.pad_token_id] = -100  # Ignore padding tokens
        student_inputs["labels"] = labels

        
        student_outputs = student_model(**student_inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)
        
        # Print the shapes and the first 100 logits of the student and teacher outputs.
        print(f"\n\nStudent logits shape: {student_outputs.logits.shape}") # torch.Size([2, 1031, 128256])
        print(f"Teacher logits shape: {teacher_outputs.logits.shape}") # torch.Size([2, 1031, 128256])
        print(f"Student logits: {student_outputs.logits[0][:100]}")
        print(f"Teacher logits: {teacher_outputs.logits[0][:100]}")
        
        # Multiply both by the attention mask of the *student inputs*
        student_logits_masked = student_outputs.logits * student_inputs["attention_mask"].unsqueeze(-1)
        teacher_logits_masked = teacher_outputs.logits * student_inputs["attention_mask"].unsqueeze(-1)
        print(f"\n\nStudent logits masked: {student_logits_masked[0][:100]}")
        print(f"Teacher logits masked: {teacher_logits_masked[0][:100]}")
        


        # Compute distillation loss
        loss, loss_components = self.distillation_loss(
            student_logits_masked,
            teacher_logits_masked,
            student_inputs,
            student_outputs.loss,
        )

        # Get current learning rate for logging
        current_lr = (
            self.lr_scheduler.get_last_lr()[0]
            if self.lr_scheduler
            else self.args.learning_rate
        )

        # Log metrics
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, LivePlotCallback):
                callback.record_metrics(
                    step=self.state.global_step,
                    loss=loss.detach().item(),
                    loss_kd=loss_components["loss_kd"].detach().item(),
                    learning_rate=current_lr,
                    epoch=self.state.epoch,
                    original_loss=student_outputs.loss.detach().item(),
                    gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                )

        return (loss, student_outputs) if return_outputs else loss

    def distillation_loss(
        self, student_logits, teacher_logits, student_inputs, original_loss
    ):

        student_logits_scaled = student_logits / config["distillation"]["temperature"]
        teacher_logits_scaled = teacher_logits / config["distillation"]["temperature"]

        loss_kd = (
            F.kl_div(
                F.log_softmax(student_logits_scaled, dim=-1),
                F.softmax(teacher_logits_scaled, dim=-1),
                reduction="batchmean",
            )
            * (config["distillation"]["temperature"] ** 2)
            / config["tokenizer"]["max_length"]
        )

        total_loss = (
            config["distillation"]["alpha"] * loss_kd
            + (1 - config["distillation"]["alpha"]) * original_loss
        )

        return total_loss, {"loss_kd": loss_kd, "original_loss": original_loss}


live_plot_callback = LivePlotCallback(
    plot_path=os.path.join(config["training"]["output_dir"], "training_loss.png"),
    update_freq=1,
    moving_avg_window=10,
    distillation_method=config["distillation"]["method"],
)

training_arguments = TrainingArguments(
    **config["training"],
)

# Create the custom SFT Trainer
trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=student_tokenizer,
    args=training_arguments,
    callbacks=[live_plot_callback],
)

# Add the teacher model to the trainer
trainer.teacher_model = teacher_model

# Prepare for distributed training
trainer = accelerator.prepare(trainer)

# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model with appropriate naming
output_base = config["training"]["output_dir"]
output_dir = os.path.join(output_base, "medqa_swe_cot_distilled")

# Increment directory name if it exists
counter = 1
original_output_dir = output_dir
while os.path.exists(output_dir):
    output_dir = f"{original_output_dir}_{counter}"
    counter += 1

trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
