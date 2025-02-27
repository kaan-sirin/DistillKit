import logging
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
import yaml
from dotenv import load_dotenv
from datetime import datetime

log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/distil_logits_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Configuration
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
    "distillation": {"temperature": 2.0, "alpha": 0.5},
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
dataset = dataset.shuffle(seed=config["dataset"]["seed"])

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(
    config["models"]["teacher"], token=HF_TOKEN
)
student_tokenizer = AutoTokenizer.from_pretrained(
    config["models"]["student"], token=HF_TOKEN
)
# TODO: Is this necessary?
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token

# Apply chat template to student tokenizer
# TODO: is chat_template necessary for MedQA-SWE?
# student_tokenizer.chat_template = config["tokenizer"]["chat_template"]


# Dataset specific formatting
def pubmedqa_format(sample):
    prompt = (
        f"You are a biomedical expert. Provide a concise answer to the following biomedical question. \n\n"
        f"Question: {sample['question']}\n\n"
        f"Answer: {sample['long_answer']}"
    )

    return {"prompt": prompt}


# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(pubmedqa_format, remove_columns=original_columns)


def tokenize_function(examples):
    return student_tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=config["tokenizer"]["max_length"],
        padding="max_length",
    )


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, num_proc=8, remove_columns=["prompt"]
)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

print("Dataset preparation complete. Loading models...")

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

teacher_model = AutoModelForCausalLM.from_pretrained(
    config["models"]["teacher"], **model_kwargs
)
student_model = AutoModelForCausalLM.from_pretrained(
    config["models"]["student"], **model_kwargs
)

# Optionally freeze layers of the student model based on spectrum configuration
if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:

    def freeze_student_spectrum(model, unfrozen_layers_file):
        with open(unfrozen_layers_file, "r") as file:
            unfrozen_layers = yaml.safe_load(file)["unfrozen_parameters"]

        for name, param in model.named_parameters():
            if not any(layer in name for layer in unfrozen_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Apply freezing to student model
    freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"])
else:
    print(
        "Spectrum configuration not found. All layers of the student model will be trainable."
    )


def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros(
            (*teacher_logits.shape[:-1], pad_size),
            dtype=teacher_logits.dtype,
            device=teacher_logits.device,
        )
        return (
            (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits)
            if student_size < teacher_size
            else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
        )
    return student_logits, teacher_logits


class LogitsTrainer(SFTTrainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        device = next(model.parameters()).device

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        self.teacher_model = self.teacher_model.to(device)

        student_model = model.module if hasattr(model, "module") else model
        teacher_model = (
            self.teacher_model.module
            if hasattr(self.teacher_model, "module")
            else self.teacher_model
        )

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss = self.distillation_loss(
            student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss
        )
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):
        device = next(self.model.parameters()).device

        student_logits, teacher_logits = pad_logits(
            student_logits.to(device), teacher_logits.to(device)
        )

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

        return (
            config["distillation"]["alpha"] * loss_kd
            + (1 - config["distillation"]["alpha"]) * original_loss
        )


# Training arguments
# TODO: replacting training arg with SFTConfig
# training_arguments = TrainingArguments(**config["training"])
training_arguments = SFTConfig(
    **config["training"],
    max_seq_length=config["tokenizer"]["max_length"],
    dataset_text_field="prompt",
)

# Create the custom SFT Trainer
trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=student_tokenizer,
    args=training_arguments,
)

# Add the teacher model to the trainer
trainer.teacher_model = teacher_model

# Prepare for distributed training
trainer = accelerator.prepare(trainer)

# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
trainer.save_model(config["training"]["output_dir"])
