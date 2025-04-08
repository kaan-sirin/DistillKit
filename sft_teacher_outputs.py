import os
import torch
import logging
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
import wandb
from dotenv import load_dotenv
from distillation_utils import load_config
import time


def setup_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logger


def main():
    logger = setup_logger()
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    config = load_config()
    run_name = f"sft_distill_{config['dataset']['name'].split('/')[-1].replace('-', '_')}_{config['dataset']['num_samples']}samples_{config['training']['num_train_epochs']}epochs_{time.strftime('%m-%d_%H-%M').replace('-', '_')}"

    output_base = config["training"]["output_dir"]
    output_dir = os.path.join(output_base, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    os.environ["WANDB_PROJECT"] = config["project_name"]
    accelerator = Accelerator()

    # --- Load Teacher Tokens ---
    teacher_tokens_path = config["dataset"]["teacher_data"]["tokens_path"]
    try:
        teacher_tokens_list = torch.load(teacher_tokens_path)
        logger.info(
            f"Loaded {len(teacher_tokens_list)} teacher token sequences from {teacher_tokens_path}"
        )

        # Basic validation
        if not isinstance(teacher_tokens_list, list) or (
            len(teacher_tokens_list) > 0
            and not isinstance(teacher_tokens_list[0], torch.Tensor)
        ):
            raise TypeError("Loaded data is not a list of tensors.")

    except Exception as e:
        logger.error(f"Error loading teacher tokens: {e}")
        raise

    num_teacher_samples = len(teacher_tokens_list)
    # Check if num_samples in config matches loaded tokens
    if "num_samples" in config["dataset"]:
        config_num_samples = config["dataset"]["num_samples"]
        if config_num_samples != num_teacher_samples:
            logger.warning(
                f"Config specified {config_num_samples} samples, but loaded {num_teacher_samples} tokens. Using loaded count: {num_teacher_samples}"
            )
            # Adjust effective number of samples based on loaded tokens
            config["dataset"]["num_samples"] = num_teacher_samples

    # --- Load Original Dataset ---
    logger.info("Loading original dataset...")
    dataset_name = config["dataset"]["name"]
    dataset_subset = config["dataset"].get("subset")
    dataset_split = config["dataset"]["split"]
    num_samples = config["dataset"]["num_samples"]  # Use the potentially adjusted value

    try:
        original_dataset = (
            load_dataset(dataset_name, dataset_subset, split=dataset_split)
            if dataset_subset
            else load_dataset(dataset_name, split=dataset_split)
        )
        logger.info(f"Filtering out examples where input column isn't empty")
        original_dataset = original_dataset.filter(lambda x: x["input"] == "")
        logger.info(f"Filtered dataset to {len(original_dataset)} examples")
        # Select samples starting from the beginning
        original_dataset = original_dataset.select(range(num_samples))
        logger.info(f"Loaded and selected {len(original_dataset)} original examples.")
        if len(original_dataset) != num_teacher_samples:
            raise ValueError(
                f"Mismatch after selection: {len(original_dataset)} original samples vs {num_teacher_samples} teacher tokens."
            )

    except Exception as e:
        logger.error(f"Error loading or selecting original dataset: {e}")
        raise

    # --- Load Tokenizer (for Student Model) ---
    logger.info(f"Loading tokenizer for student model: {config['models']['student']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], token=HF_TOKEN, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # --- Combine Prompts and Teacher Completions ---
    logger.info("Combining prompts and teacher completions...")
    combined_data = []

    # 1. Format the dataset (format function as inspiration from batch_comparison.py)
    # 2. Tokenize the dataset
    # 3. Combine the prompts and teacher completions tokens into a single text column

    def alpaca_format(example):
        start_tokens = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        end_tokens = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        try:
            text = start_tokens + example["instruction"] + end_tokens
            return {"text": text}
        except Exception as e:
            logger.error(f"Sample keys: {list(example.keys())}")
            logger.error(f"Error formatting example: {e}")
            raise

    dataset = original_dataset.map(
        alpaca_format, remove_columns=original_dataset.column_names
    )
    decoded_teacher_outputs = [
        tokenizer.decode(
            token, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        for token in teacher_tokens_list
    ]

    if len(decoded_teacher_outputs) != len(dataset):
        raise ValueError(
            f"Mismatch after formatting: {len(dataset)} formatted prompts vs {len(teacher_tokens_list)} teacher tokens."
        )

    for example_data, teacher_output in zip(dataset, decoded_teacher_outputs):
        prompt_text = example_data["text"]  # Get the pre-formatted prompt text
        combined_text = prompt_text + teacher_output
        combined_data.append({"text": combined_text})

    sft_dataset = Dataset.from_list(combined_data)
    logger.info(f"Created SFT dataset with {len(sft_dataset)} examples.")
    sft_dataset = sft_dataset.shuffle(seed=config["dataset"]["seed"])
    logger.info(f"Shuffled SFT dataset with seed {config['dataset']['seed']}")

    # Optional: Split dataset (if desired)
    sft_dataset = sft_dataset.train_test_split(
        test_size=0.1, seed=config["dataset"]["seed"]
    )
    train_dataset = sft_dataset["train"]
    eval_dataset = sft_dataset["test"]

    logger.info(f"Loading student model: {config['models']['student']}")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "token": HF_TOKEN,
        "trust_remote_code": True,
    }
    if config["model_config"].get("use_flash_attention", False):
        if (
            "attn_implementation" not in model_kwargs
        ):  # Only add if not already specified by model config itself
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Enabled Flash Attention 2")

    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"], **model_kwargs
    )

    # --- Training Arguments ---
    logger.info("Setting up training arguments...")
    # Make sure output_dir is correctly passed
    config["training"]["output_dir"] = output_dir
    training_arguments = TrainingArguments(
        **config["training"],
        report_to=["wandb"],
    )

    # --- Initialize Wandb (Optional) ---
    logger.info("Initializing WandB...")
    wandb.init(
        project=config["project_name"],
        name=run_name,
        group="sft_on_teacher_outputs",
        config={
            "student_model": config["models"]["student"],
            "teacher_dataset": config["dataset"]["name"],
            "num_samples": num_samples,
            "num_epochs": config["training"]["num_train_epochs"],
            "training_args": config["training"],
            "teacher_tokens_path": teacher_tokens_path,
        },
    )

    # --- Create SFT Trainer ---
    logger.info("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # dataset_num_proc=config["tokenizer"].get("num_proc", 8), # Can speed up preprocessing if needed
    )

    # Prepare for distributed training if applicable
    trainer = accelerator.prepare(trainer)  # Trainer handles accelerator internally

    # --- Train ---
    logger.info("Starting training...")
    train_result = trainer.train(
        resume_from_checkpoint=config["training"].get("resume_from_checkpoint")
    )
    logger.info("Training finished.")

    # --- Save ---
    logger.info(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)  # Saves model, tokenizer, config
    # Log metrics if needed
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("Model saving complete.")

    # Default use_wandb to True if not provided
    if config.get("use_wandb", True):
        wandb.finish()


if __name__ == "__main__":
    main()
