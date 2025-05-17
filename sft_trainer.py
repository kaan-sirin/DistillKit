import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from accelerate import Accelerator
import wandb
from dotenv import load_dotenv
from distillation_utils import gsm8k_format, load_config, magpie_format, tokenize_function
import time


def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    config = load_config('config_sft.yaml')
    dataset_name = config["dataset"]["name"]
    num_samples = config["dataset"].get("num_samples", None)
    test_size = config["dataset"].get("test_size", 0.1)
    dataset_config = load_config(f"datasets.yaml").get(dataset_name, None)
    if dataset_config is None:
        raise ValueError(f"Dataset config not found for {dataset_name}")

    group_name = f"sft_{dataset_name.split('/')[-1].replace('-', '_')}_"
    if num_samples:
        group_name += f"{num_samples}s_"
    group_name += f"{config['training']['num_train_epochs']}e_{time.strftime('%m-%d_%H-%M').replace('-', '_')}"

    output_base = config["training"]["output_dir"]
    output_dir = os.path.join(output_base, group_name)

    # Set up environment
    os.environ["WANDB_PROJECT"] = config["project_name"] + "-sft"
    accelerator = Accelerator()
    run_name = f"process_{accelerator.process_index}_{time.strftime('%m_%d_%H_%M')}"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], token=HF_TOKEN
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Load and preprocess dataset
    dataset = (
        load_dataset(
            dataset_name,
            dataset_config.get("subset"),
            split=dataset_config.get("split"),
        )
        if dataset_config.get("subset")
        else load_dataset(dataset_name, split=dataset_config.get("split"))
    )

    # First take a subset of the dataset if specified, then shuffle
    if num_samples:
        dataset = dataset.select(range(num_samples))
    dataset = dataset.shuffle(seed=42)

    # Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    if dataset_name== "nicher92/magpie_llama70b_260k_filtered_swedish":
        dataset = dataset.filter(lambda x: x["task_category"] != "Math")
        dataset = dataset.filter(lambda x: "Math" not in x["other_task_category"])
        dataset = dataset.map(magpie_format, remove_columns=original_columns)
    elif dataset_name == "openai/gsm8k":
        dataset = dataset.map(gsm8k_format, remove_columns=original_columns)
    else:
        raise ValueError(
            f"Make sure to specify a format function for {config['dataset']['name']} in {__file__}"
        )

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(
            examples, tokenizer, config["tokenizer"]["max_length"]
        ),
        batched=True,
        num_proc=8,
        remove_columns=["text"],
    )

    tokenized_dataset = tokenized_dataset.train_test_split(test_size=test_size)

    print("Dataset preparation complete. Loading model...")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], **model_kwargs
    )

    # Training arguments
    
    training_args_dict = config["training"].copy()
    training_args_dict["output_dir"] = output_dir
    training_args_dict["report_to"] = ["wandb"]
    early_stopping_patience = training_args_dict.pop("early_stopping_patience", 3)
    training_arguments = TrainingArguments(**training_args_dict)

    
    callbacks = []
    if training_arguments.load_best_model_at_end:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience
            )
        )

    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(
            project=config["project_name"],
            name=run_name,
            group=group_name,
            config={
            "model": config["model"],
            "num_samples": config["dataset"]["num_samples"],
            "num_epochs": config["training"]["num_train_epochs"],
        },
        reinit=False,
    )

    # Create the SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        args=training_arguments,
        callbacks=callbacks,
    )

    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
