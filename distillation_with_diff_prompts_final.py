import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
from dotenv import load_dotenv
from distillation_utils import LivePlotCallback
import time
import yaml

# PAINSTAKINGLY SLOW:   1%|          | 5/420 [23:07<31:59:44, 277.55s/it]
# Changing batchsize from 1 to 2

# Got the following error:
# torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.27 GiB. GPU 0 has a total capacity of 47.53 GiB of which 844.06 MiB is free. Including non-PyTorch memory, this process has 46.68 GiB memory in use. Of the allocated memory 41.21 GiB is allocated by PyTorch, and 5.15 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
#   1%|          | 2/210 [14:14<24:41:03, 427.23s/it]

# Cannot use FlashAttention2 due to CUDA version
# TODO: FlashAttention2 error

# ASSUMPTIONS
# [] a chat template is not needed for this dataset


def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            config["training"]["learning_rate"] = float(
                config["training"]["learning_rate"]
            )
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. Using default configuration.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing {config_path}: {e}")
        raise


def medqa_format(sample):
    try:
        prompt = (
            "Du är en medicinsk expert. "
            "Din uppgift är att **steg för steg** förklara varför alternativ "
            f"{sample['answer']} är det korrekta svaret på följande fråga:\n\n"
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


def tokenize_function(examples, tokenizer):
    try:
        prompt = tokenizer(
            examples["prompt"],
            return_attention_mask=True,
        )
        # <|begin_of_text|>Du är en medicinsk expert. Förklara...
        # len(prompt["input_ids"]) = 33 (with <|begin_of_text|> token)
        # len(prompt["attention_mask"]) = 33 (all ones)

        inputs = tokenizer(
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


class LogitsTrainer(SFTTrainer):        
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        if hasattr(model, "device"):
            device = model.device
        else:
            # For DataParallel models, use the device of the module inside
            device = (
                model.module.device
                if hasattr(model, "module")
                else torch.device("cuda")
            )

        prompt = {
            "input_ids": inputs["prompt_input_ids"].to(device),
            "attention_mask": inputs["prompt_attention_mask"].to(device),
        }

        inputs = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
        }
        # ============ DEBUGGING ============
        # A batch of 2 examples, both with the same number of tokens, the shorter one is padded.
        # Despite me not setting padding='max_length' in the tokenizer.

        for i in range(len(inputs["input_ids"])):
            print(
                f"\n\nLength of input {i+1} ({i+1}/{len(inputs['input_ids'])}): {len(inputs['input_ids'][i])}"
            )
            print(
                f"First 100 chars of input {i+1}, decoded: {self.tokenizer.decode(inputs['input_ids'][i])[:100]}..."
            )
            num_zeros = len(inputs["input_ids"][i]) - sum(inputs["attention_mask"][i])
            print(f"Number of zeros in attention mask of input {i+1}: {num_zeros}")
            if num_zeros > 0:
                print(
                    f"Last {num_zeros + 3} tokens of input {i+1} (expecting 3 non-padding tokens): {inputs['input_ids'][i][-(num_zeros+3):]}"
                )
        # ============ DEBUGGING ============

        # Get the number of tokens in the teacher input, to exclude them from the answer.
        teacher_input_tokens = inputs["input_ids"].shape[-1]
        print(
            f"\n\nNumber of teacher input tokens (prompt + question): {teacher_input_tokens}"
        )

        # Teacher-generated answer
        MAX_NEW_TOKENS = 128
        answers = self.teacher_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # ============ DEBUGGING ============
        for i in range(len(answers.sequences)):
            print(
                f"\n\nShape of the {i}th answer: {answers.sequences[i][teacher_input_tokens:].shape}"
            )  # 512 tokens
            # print(f"Tokens in answer {i}: {answers.sequences[i][teacher_input_tokens:]}")
            # print(
            #     f"First 100 chars of answer {i}, decoded: {student_tokenizer.decode(answers.sequences[i][teacher_input_tokens:][:100])}..."
            # )
            print(
                f"Answer {i+1}, decoded: {self.tokenizer.decode(answers.sequences[i][teacher_input_tokens:])}"
            )
        # Sometimes answers are padded with eos_token's (128001) and sometimes not.
        # exit()
        # ============ DEBUGGING ============

        teacher_outputs = [answer[teacher_input_tokens:] for answer in answers]

        # Creating inputs and attention masks for student and teacher
        # Teacher will be given the prompt + question + answer

        # The output from generate is everything the teacher needs for the forward pass.
        # print(f"Teacher inputs 0: {answers.sequences[0]}")
        for i in range(len(answers.sequences)):
            # print(f"\n\nTeacher input tokens {i+1} ({i+1}/{len(answers.sequences)}): {answers.sequences[i]}")
            print(
                f"\n\nTeacher inputs {i+1} shape ({i+1}/{len(answers.sequences)}): {answers.sequences[i].shape}"
            )
            print(
                f"Teacher inputs {i+1} decoded, first 100 chars ({i+1}/{len(answers.sequences)}): {self.tokenizer.decode(answers.sequences[i])[:100]}..."
            )

        # I assume I can simply add ones to the original attention mask for the teacher inputs.
        # TODO: Sometimes the teacher inputs are padded with eos_token's (128001) and I'm not sure if they need to be masked.
        teacher_attention_masks_list = []
        # Make sure the number of output sequences is the same as the number of attention masks.
        if len(answers.sequences) != len(inputs["attention_mask"]):
            raise ValueError(
                f"Number of output sequences ({len(answers.sequences)}) is not the same as the number of attention masks ({len(inputs['attention_mask'])})"
            )
        print()
        for i, mask in enumerate(inputs["attention_mask"]):
            # The number of tokens in the generated answer for this example, I think it's always 512 (MAX_NEW_TOKENS) with eventual padding.
            answer_length = answers.sequences[i][teacher_input_tokens:].shape[0]
            print(
                f"Answer length of example {i+1} (expected {MAX_NEW_TOKENS}): {answer_length}"
            )
            # Extend the original mask with ones for the answer tokens
            extended_mask = torch.cat(
                [mask, torch.ones(answer_length, device=mask.device, dtype=mask.dtype)]
            )
            teacher_attention_masks_list.append(extended_mask)
            print(
                f"The length of the extended mask for example {i+1} is {len(extended_mask)}"
            )

        # Stack the extended masks to create a batch
        teacher_attention_mask = torch.stack(teacher_attention_masks_list)

        student_input_ids_list = []
        student_attention_masks_list = []
        teacher_input_ids_list = []
        for i, seq in enumerate(answers.sequences):
            print(f"\n\n---------> Example {i+1} of {len(answers.sequences)}")
            teacher_input_ids_list.append(
                seq
            )  # Use the complete sequence from generation
            # For student, I believe the last token of the CoT prompt should be set to <|begin_of_text|>
            # Here I'm simply replacing the last token of the prompt with <|begin_of_text|>
            # I need to get the index of the last token of the prompt first.
            last_prompt_token_index = len(prompt["input_ids"][i]) - 1
            print(f"\n\nLast prompt token index: {last_prompt_token_index}")
            print(
                f"Prompt token at index is {prompt['input_ids'][i][last_prompt_token_index]} which corresponds to '{self.tokenizer.decode(prompt['input_ids'][i][last_prompt_token_index])}' (expecting a colon)"
            )
            print(
                f"5 tokens before and after the last prompt token: {prompt['input_ids'][i][last_prompt_token_index-5:last_prompt_token_index+6]}: {self.tokenizer.decode(prompt['input_ids'][i][last_prompt_token_index-5:last_prompt_token_index+6])}"
            )
            print(
                f"5 sequence tokens before and after the last prompt token: {seq[last_prompt_token_index-5:last_prompt_token_index+6]}: {self.tokenizer.decode(seq[last_prompt_token_index-5:last_prompt_token_index+6])}"
            )

            # Set the last prompt token to <|begin_of_text|>, and the attention mask to 0 for the tokens coming before.
            seq[last_prompt_token_index] = self.tokenizer.bos_token_id
            print(
                f"Same sequence tokens after setting the last prompt token to <|begin_of_text|>"
            )
            print(
                f"{seq[last_prompt_token_index-5:last_prompt_token_index+5]}: {self.tokenizer.decode(seq[last_prompt_token_index-5:last_prompt_token_index+5])}"
            )
            student_input_ids_list.append(seq)
            # Copy the teacher attention mask
            new_attention_mask = teacher_attention_masks_list[i].clone()
            new_attention_mask[:last_prompt_token_index] = 0
            print(
                f"Student's attention mask after setting prompt tokens to 0 (should be all zeros + 3 ones): {new_attention_mask[:last_prompt_token_index + 3]}"
            )
            student_attention_masks_list.append(new_attention_mask)

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

        # Labels are necessary for the cross entropy loss (student_outputs.loss)
        # Create labels for next-token prediction (shift input_ids right by 1)
        # -100 is the default ignore_index in PyTorch's CrossEntropyLoss. So, any token with a label of -100 will be ignored in loss computation. [https://discuss.huggingface.co/t/will-trainer-loss-functions-automatically-ignore-100/36134]
        labels = student_inputs["input_ids"].clone()
        labels[:, :-1] = labels[:, 1:].clone()
        labels[:, -1] = -100  # Don't calculate loss for the last token prediction
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens
        student_inputs["labels"] = labels

        student_outputs = self.model(**student_inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)

        # Print the shapes and the first 100 logits of the student and teacher outputs.
        print(f"\n\nStudent logits shape: {student_outputs.logits.shape}")
        print(f"Teacher logits shape: {teacher_outputs.logits.shape}")
        print(f"Student logits: {student_outputs.logits[0][:100]}")
        print(f"Teacher logits: {teacher_outputs.logits[0][:100]}")

        # Multiply both by the attention mask of the *student inputs* before the KLD computation.
        # This is to ensure that the KLD computation is only done for the tokens that the student has attended to.
        student_logits_masked = student_outputs.logits * student_inputs[
            "attention_mask"
        ].unsqueeze(-1)
        teacher_logits_masked = teacher_outputs.logits * student_inputs[
            "attention_mask"
        ].unsqueeze(-1)
        print(f"Student logits masked: {student_logits_masked[0][:100]}")
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

        exit()
        return (loss, student_outputs) if return_outputs else loss

    def distillation_loss(
        self, student_logits, teacher_logits, student_inputs, original_loss
    ):

        student_logits_scaled = student_logits / self.config["distillation"]["temperature"]
        teacher_logits_scaled = teacher_logits / self.config["distillation"]["temperature"]

        # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        loss_kd = (
            F.kl_div(
                F.log_softmax(student_logits_scaled, dim=-1),
                F.softmax(teacher_logits_scaled, dim=-1),
                reduction="batchmean",
            )
            * (self.config["distillation"]["temperature"] ** 2)
            / self.config["tokenizer"]["max_length"]
        )

        total_loss = (
            self.config["distillation"]["alpha"] * loss_kd
            + (1 - self.config["distillation"]["alpha"]) * original_loss
        )

        return total_loss, {"loss_kd": loss_kd, "original_loss": original_loss}


def load_teacher_outputs(output_path):
    # Load the pre-generated teacher outputs (tokens, logits, shuffled indices)
    tokens_file = os.path.join(output_path, "generated_tokens.pt")
    logits_file = os.path.join(output_path, "sparse_logits.pt")
    indices_file = os.path.join(output_path, "indices.pt")
    
    print(f"Loading teacher outputs from {output_path}")
    token_tensors = torch.load(tokens_file)
    sparse_logit_tensors = torch.load(logits_file)
    shuffled_indices = torch.load(indices_file)
    
    print(f"Loaded {len(token_tensors)} teacher outputs")
    print(f"First 10 shuffled indices: {shuffled_indices[:10]}")
    
    return token_tensors, sparse_logit_tensors, shuffled_indices


def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Load configuration from file
    config = load_config()

    # Set up output directory
    output_base = config["training"]["output_dir"]
    # month, day, hour, minute
    output_dir = os.path.join(
        output_base, f"medqa_swe_{time.strftime('%m-%d_%H-%M').replace('-', '_')}"
    )
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

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

    # TODO: (DANGEROUS) Is this correct? It complains about pad_token not being set. Even when padding is not set to True in tokenizer.
    # Inspired by https://discuss.huggingface.co/t/how-to-set-the-pad-token-for-meta-llama-llama-3-models/103418/5
    # Answer starting with "I have attempted fine-tuning the LLaMA-3.1-8B..."
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    dataset = dataset.map(
        medqa_format,
        remove_columns=original_columns,
    )

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, student_tokenizer),
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

    live_plot_callback = LivePlotCallback(
        plot_path=output_dir + "/training_loss.png",
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
        args=training_arguments,
        callbacks=[live_plot_callback],
    )

    # Add the teacher model to the trainer
    trainer.teacher_model = teacher_model
    trainer.tokenizer = student_tokenizer
    trainer.config = config

    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
