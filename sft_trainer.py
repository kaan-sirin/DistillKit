import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
import wandb
from dotenv import load_dotenv
from distillation_utils import load_config, medqa_format
import time


def tokenize_function(examples, tokenizer, config):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=config["tokenizer"]["max_length"],
        padding="max_length",
    )


def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    config = load_config()
    run_name = f"sft_{config['dataset']['name'].split('/')[-1].replace('-', '_')}_{config['dataset']['num_samples']}samples_{config['training']['num_train_epochs']}epochs_{time.strftime('%m-%d_%H-%M').replace('-', '_')}"
    # Output directory
    output_base = config["training"]["output_dir"]
    output_dir = os.path.join(
        output_base,
        f"sft_{config['dataset']['name'].split('/')[-1].replace('-', '_')}_{config['dataset']['num_samples']}_{config['training']['num_train_epochs']}_{time.strftime('%m-%d_%H-%M').replace('-', '_')}",
    )
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], token=HF_TOKEN
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    dataset = dataset.map(medqa_format, remove_columns=original_columns)

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, config),
        batched=True,
        num_proc=8,
        remove_columns=["text"],
    )

    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    print("Dataset preparation complete. Loading model...")

    # Load model with configurable flash attention
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"], **model_kwargs
    )

    # Training arguments
    training_arguments = TrainingArguments(
        **config["training"],
        report_to=["wandb"],
    )

    # Initialize wandb
    wandb.init(
        project=config["project_name"],
        name=run_name,
        config={
            "model": config["models"]["student"],
            "num_samples": config["dataset"]["num_samples"],
            "num_epochs": config["training"]["num_train_epochs"],
        },
    )

    # Create the SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
