import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
from dotenv import load_dotenv
from distillation_utils import LivePlotCallback

torch.set_printoptions(threshold=10_000)  # Large enough threshold


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
        "per_device_train_batch_size": 1,
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
    "model_config": {"use_flash_attention": False},
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

# TODO: DANGEROUS: Is this correct? It complains about pad_token not being set.
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

        tokenized_prompt = student_tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=config["tokenizer"]["max_length"],
            padding="max_length",
            return_attention_mask=True,
        )

        # Tokenize student prompts
        student_inputs = student_tokenizer(
            examples["question"],
            truncation=True,
            max_length=config["tokenizer"]["max_length"],
            padding="max_length",
            return_attention_mask=True,
        )

        teacher_inputs = student_tokenizer(
            [p + q for p, q in zip(examples["prompt"], examples["question"])],
            truncation=True,
            max_length=config["tokenizer"]["max_length"],
            padding="max_length",
            return_attention_mask=True,
        )

        return {
            # Prompt inputs
            "prompt_input_ids": tokenized_prompt["input_ids"],
            "prompt_attention_mask": tokenized_prompt["attention_mask"],
            # Student inputs (only question)
            "input_ids": student_inputs["input_ids"],
            "attention_mask": student_inputs["attention_mask"],
            # Teacher inputs (prompt + question)
            "teacher_input_ids": teacher_inputs["input_ids"],
            "teacher_attention_mask": teacher_inputs["attention_mask"],
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
        student_inputs = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
        }
        teacher_inputs = {
            "input_ids": inputs["teacher_input_ids"].to(device),
            "attention_mask": inputs["teacher_attention_mask"].to(device),
        }

        teacher_input_tokens = teacher_inputs["input_ids"].shape[-1]

        # Teacher-generated answer
        answers = self.teacher_model.generate(
            **teacher_inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

        teacher_generated_tokens = [answer[teacher_input_tokens:] for answer in answers]
        # for i, answer in enumerate(teacher_generated_tokens):
        #     print(f"Answer {i} length: {len(answer)}")
        #     print(f"Answer tokens: {answer}")

        # for answer in teacher_generated_tokens:
        #     decoded = student_tokenizer.decode(
        #         answer, skip_special_tokens=True
        #     )
        #     print(f"First 20 chars: {decoded[:200]}...")
        #     print(
        #         f"Last 20 chars: ...{decoded[-200:] if len(decoded) > 200 else decoded}"
        #     )

        new_input_ids = []
        new_attention_masks = []

        for i in range(student_inputs["input_ids"].shape[0]):
            # Concatenate along dimension 0 for each example
            new_input_ids.append(
                torch.cat(
                    [student_inputs["input_ids"][i], teacher_generated_tokens[i]], dim=0
                )
            )
            new_attention_masks.append(
                torch.cat(
                    [
                        student_inputs["attention_mask"][i],
                        torch.ones_like(teacher_generated_tokens[i]),
                    ],
                    dim=0,
                )
            )

        student_inputs["input_ids"] = torch.stack(new_input_ids)
        student_inputs["attention_mask"] = torch.stack(new_attention_masks)

        student_model = model

        student_outputs = student_model(**student_inputs)
        original_loss = student_outputs.loss
        if not isinstance(original_loss, torch.Tensor):
            original_loss = torch.tensor(list(original_loss), device=device).mean()
        print(f"Student outputs loss: {original_loss}")

        new_teacher_input_ids = []
        new_teacher_attention_masks = []

        for i in range(teacher_inputs["input_ids"].shape[0]):
            new_teacher_input_ids.append(
                torch.cat(
                    [teacher_inputs["input_ids"][i], teacher_generated_tokens[i]], dim=0
                )
            )

            # Get the prompt attention mask for this example
            prompt_attention_mask = prompt["attention_mask"][i]
            inverse_prompt_mask = 1 - prompt_attention_mask
            teacher_attention_mask = (
                teacher_inputs["attention_mask"][i] * inverse_prompt_mask
            )

            new_teacher_attention_masks.append(
                torch.cat(
                    [
                        teacher_attention_mask,
                        torch.ones_like(teacher_generated_tokens[i]),
                    ],
                    dim=0,
                )
            )

        teacher_inputs["input_ids"] = torch.stack(new_teacher_input_ids)
        teacher_inputs["attention_mask"] = torch.stack(new_teacher_attention_masks)

        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)

        # Compute distillation loss
        loss, loss_components = self.distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            student_inputs,
            original_loss,
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
                    loss_kd=loss_components["loss_kd"].detach().mean().item(),
                    learning_rate=current_lr,
                    epoch=self.state.epoch,
                    original_loss=original_loss,
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
