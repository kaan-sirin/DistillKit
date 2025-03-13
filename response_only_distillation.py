# from distil_logits import pad_logits
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
import yaml
from dotenv import load_dotenv
import sys

from distillation_utils import LivePlotCallback

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Configuration #TODO: learning rate was turning into str for some reason
# with open("config.yaml", "r") as file:
#     config = yaml.safe_load(file)

config = {
    "project_name": "distil-logits",
    "dataset": {
        "name": "qiaojin/PubMedQA",
        "split": "train",
        "num_samples": 3000,
        "seed": 42,
        "subset": "pqa_artificial",
    },
    "models": {
        "teacher": "meta-llama/Llama-3.2-3B-Instruct",
        "student": "meta-llama/Llama-3.2-1B-Instruct",
    },
    "tokenizer": {
        "max_length": 512,  # TODO: Had to change from 4096 due to memory issues
        # TODO: Is chat_template necessary for MedQA-SWE?
        # "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
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
    if config["dataset"]["subset"]
    else load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
)

# First take a subset of the dataset if specified, then shuffle
if "num_samples" in config["dataset"]:
    dataset = dataset.select(range(config["dataset"]["num_samples"]))
dataset = dataset.shuffle(seed=config["dataset"]["seed"]) #TODO: uncomment this line

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(
    config["models"]["teacher"], token=HF_TOKEN
)
student_tokenizer = AutoTokenizer.from_pretrained(
    config["models"]["student"], token=HF_TOKEN
)


if teacher_tokenizer.pad_token is None: # TODO: Fråga, är det rimligt att sätta pad_token till eos_token?
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token
    

# Apply chat template to student tokenizer
# TODO: Removed for now, can be found in the original distil_logits.py


def pubmedqa_format(sample):
    try:
        question = (
            f"You are a biomedical expert. Provide a concise answer to the following biomedical question. \n\n"
            f"Question: {sample['question']}\n\n"
        )
        answer = sample["long_answer"]
        return {"question": question, "answer": answer}
    except Exception as e:
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Error formatting sample: {e}")
        raise

# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(
    pubmedqa_format,
    remove_columns=original_columns,
)

# TESTING ANOTHER TOKENIZE FUNCTION
def tokenize_function(examples):
    try:
        prompts = examples["question"]
        answers = examples["answer"]
        full_texts = [prompt + answer for prompt, answer in zip(prompts, answers)]
        
        # Batch tokenize full texts
        tokenized_full = student_tokenizer(
            full_texts,
            truncation=True,
            max_length=config["tokenizer"]["max_length"],
            padding="max_length",
        )
        
        # Batch tokenize just the prompts
        tokenized_prompts = student_tokenizer(
            prompts,
            truncation=True,
            max_length=config["tokenizer"]["max_length"],
            add_special_tokens=True,
        )
        
        # Create labels by masking prompt tokens (set to -100)
        labels = []
        for i, full_ids in enumerate(tokenized_full["input_ids"]):
            try:
                label_ids = full_ids.copy()
                prompt_len = len(tokenized_prompts["input_ids"][i])
                label_ids[:prompt_len] = [-100] * prompt_len
                labels.append(label_ids)
            except Exception as e:
                print(f"Error processing example {i}: {e}")
                raise
        
        tokenized_full["labels"] = labels
        
        # Validate output format
        expected_keys = {"input_ids", "attention_mask", "labels"}
        if not all(key in tokenized_full for key in expected_keys):
            missing_keys = expected_keys - set(tokenized_full.keys())
            raise ValueError(f"Missing required keys in tokenized output: {missing_keys}")
            
        return tokenized_full
        
    except Exception as e:
        print(f"Error in tokenization: {e}")
        print(f"First prompt: {prompts[0] if prompts else 'No prompts'}")
        print(f"First answer: {answers[0] if answers else 'No answers'}")
        raise

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,
    remove_columns=["question", "answer"],
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

# Optionally freeze layers of the student model based on spectrum configuration
# TODO: Removed for now, can be found in the original distil_logits.py


class LogitsTrainer(SFTTrainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        device = next(model.parameters()).device

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        # self.teacher_model = self.teacher_model.to(device) #TODO: I move it to device only once after it's first loaded

        student_model = model.module if hasattr(model, "module") else model
        teacher_model = (
            self.teacher_model.module
            if hasattr(self.teacher_model, "module")
            else self.teacher_model
        )

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss, loss_components = self.distillation_loss(
            student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss
        )
        
        # if self.state.global_step % 100 == 0:  # Only check periodically
        #     print("\n=== Student Output Verification ===")
        #     print(f"Student logits shape: {student_outputs.logits.shape}")
        #     print(f"Contains NaN: {torch.isnan(student_outputs.logits).any()}")
        #     print(f"Contains inf: {torch.isinf(student_outputs.logits).any()}")

        #     # Get most likely tokens and decode them
        #     top_tokens = torch.argmax(student_outputs.logits[0], dim=-1)  # Get predictions for first sequence
        #     decoded_text = student_tokenizer.decode(top_tokens)
        #     decoded_input = student_tokenizer.decode(inputs["input_ids"][0])
        #     print("\nSample decoded input:")
        #     print(decoded_input[:200] + "..." if len(decoded_input) > 200 else decoded_input)
        #     print("\nSample decoded output:")
        #     print(decoded_text[:200] + "..." if len(decoded_text) > 200 else decoded_text)
            
        #     sys.exit("Debug print complete - exiting")
        

        # Get current learning rate from scheduler
        if self.lr_scheduler:
            current_lr = self.lr_scheduler.get_last_lr()[0]
        else:
            current_lr = self.args.learning_rate

        for callback in self.callback_handler.callbacks:
            if isinstance(callback, LivePlotCallback):
                callback.record_metrics(
                    step=self.state.global_step,
                    loss=custom_loss.detach().mean().item(),
                    loss_kd=loss_components["loss_kd"].detach().mean().item(),
                    original_loss=loss_components["original_loss"]
                    .detach()
                    .mean()
                    .item(),
                    gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                    learning_rate=current_lr,
                    epoch=self.state.epoch,
                )
                break

        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):
        device = next(self.model.parameters()).device

        # Get the answer mask directly from inputs
        response_mask = (inputs["labels"] >= 0).to(device)

        # Calculate full-sequence loss for comparison/debugging
        if config["distillation"]["method"] == "soft_targets":
            student_logits_scaled = (
                student_logits / config["distillation"]["temperature"]
            )
            teacher_logits_scaled = (
                teacher_logits / config["distillation"]["temperature"]
            )

            

            kl = F.kl_div(
                F.log_softmax(student_logits_scaled, dim=-1),
                F.softmax(teacher_logits_scaled, dim=-1),
                reduction="none",
            ) # Supposedly returns [B, T, V]
            # print(f"FULL LOSS KD SHAPE: {kl.shape}")
            
            kl_per_token = kl.sum(dim=-1) # Turn it into [B, T]
            # print(f"KL PER TOKEN SHAPE: {kl_per_token.shape}")
            
            # ------------------------
            # 1) Full-sequence KL
            # ------------------------
            # Average across *all* tokens 
            full_loss_kd = (kl_per_token.sum() / kl_per_token.numel()) * (config["distillation"]["temperature"] ** 2) # [B, T] -> [1]

            # ------------------------
            # 2) Answer-only KL
            # ------------------------
            # response_mask = (inputs["labels"] >= 0).to(device)
            # print(f"KL PER TOKEN SHAPE: {kl_per_token.shape}")
            # print(f"LABELS SHAPE: {inputs['labels'].shape}")
            # print(f"RESPONSE MASK SHAPE: {response_mask.shape}")
            kl_flat       = kl_per_token.view(-1)                   # [B*T]
            labels_flat   = inputs['labels'].view(-1)               # [B*T]
            # response_mask_flat = response_mask.view(-1)           # [B*T]
            response_mask_flat = (labels_flat >= 0).to(device)      # [B*T]
            # print(f"KL FLAT SHAPE: {kl_flat.shape}")
            # print(f"LABELS FLAT SHAPE: {labels_flat.shape}")
            kl_filtered   = kl_flat[response_mask_flat]     # [B*T]
            
            if kl_filtered.numel() > 0:
                # Per-token average among *only* the valid (answer) tokens
                loss_kd = kl_filtered.mean() * (config["distillation"]["temperature"] ** 2)
            else:
                # If for some batch there are no valid tokens, define 0 or skip
                loss_kd = torch.tensor(0.0, device=kl.device)


        elif config["distillation"]["method"] == "hard_targets":
            teacher_predictions = torch.argmax(teacher_logits, dim=-1)

            # Full sequence loss
            full_loss_kd = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                teacher_predictions.view(-1),
                reduction="mean",
            )

            # Answer-only loss
            student_logits_flat = student_logits.view(-1, student_logits.size(-1))
            teacher_predictions_flat = teacher_predictions.view(-1)

            student_filtered = student_logits_flat[response_mask.view(-1)]
            teacher_filtered = teacher_predictions_flat[response_mask.view(-1)]

            loss_kd = F.cross_entropy(
                student_filtered,
                teacher_filtered,
                reduction="mean",
            )
        else:
            raise ValueError(
                f"Unknown distillation method: {config['distillation']['method']}"
            )

        # Log loss comparison
        # if self.state.global_step % 50 == 0:
        #     answer_ratio = response_mask.float().mean().item()
        #     print("\n--- Response-Only Loss Statistics ---")
        #     print(
        #         f"Answer token ratio: {answer_ratio:.2f} ({response_mask.sum().item()} / {response_mask.numel()})"
        #     )
        #     print(f"Full sequence loss: {full_loss_kd.item():.4f}")
        #     print(f"Answer-only loss: {loss_kd.item():.4f}")
        #     print(f"Difference: {(loss_kd - full_loss_kd).item():.4f}")
        #     print("--------------------------------------\n")

        total_loss = (
            config["distillation"]["alpha"] * loss_kd
            + (1 - config["distillation"]["alpha"]) * original_loss
        )

        loss_components = {
            "loss_kd": loss_kd,
            "original_loss": original_loss,
            "full_seq_loss_kd": full_loss_kd,
            "answer_ratio": response_mask.float().mean(),
        }

        return total_loss, loss_components


live_plot_callback = LivePlotCallback(
    plot_path=os.path.join(config["training"]["output_dir"], "training_loss.png"),
    update_freq=1,
    moving_avg_window=10,
    distillation_method=config["distillation"]["method"],
)

training_arguments = TrainingArguments(**config["training"])

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

# Save the final model
# Determine the output directory with distillation method
output_base = config["training"]["output_dir"]
distill_method = config["distillation"]["method"]
output_dir = os.path.join(output_base, f"{distill_method}")

# Check if directory exists and increment if necessary
counter = 1
original_output_dir = output_dir
while os.path.exists(output_dir):
    output_dir = f"{original_output_dir}_{counter}"
    counter += 1

# Save the model
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
