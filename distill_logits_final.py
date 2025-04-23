import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
import wandb
from dotenv import load_dotenv
from distillation_utils import load_config, medqa_format, medlfqa_format
import time

from reverse_kld import reverse_kld


def tokenize_function(examples, tokenizer, config):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=config["tokenizer"]["max_length"],
        padding="max_length",
    )


class LogitsTrainer(SFTTrainer):
    def __init__(
        self, model, teacher_model=None, tokenizer=None, config=None, *args, **kwargs
    ):
        super().__init__(model=model, *args, **kwargs)
        self.config = config
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if hasattr(model, "device"):
            device = model.device
        elif hasattr(model, "module"):
            # For DataParallel models, use the device of the module inside
            device = model.module.device

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

        total_loss, loss_components = self.distillation_loss(
            student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss
        )

        self._current_loss_kd = loss_components["loss_kd"].detach().item()
        self._current_original_loss = loss_components["original_loss"].detach().item()

        return (total_loss, student_outputs) if return_outputs else total_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):
        use_reverse_kld = False

        student_logits_scaled = (
            student_logits / self.config["distillation"]["temperature"]
        )
        teacher_logits_scaled = (
            teacher_logits / self.config["distillation"]["temperature"]
        )

        if use_reverse_kld:
            loss_kd = (
                reverse_kld(student_logits_scaled, teacher_logits_scaled)
                * self.config["distillation"]["temperature"] ** 2
            )
        else:
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

    def log(self, logs, start_time=None):
        # First add our component losses if available
        if hasattr(self, "_current_loss_kd") and hasattr(
            self, "_current_original_loss"
        ):
            logs["loss_kd"] = self._current_loss_kd
            logs["original_loss"] = self._current_original_loss

        # Then let the parent handle logging
        super().log(logs)


def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    config = load_config()
    run_name = f"reverse_distill_{config['dataset']['name'].split('/')[-1].replace('-', '_')}_{config['dataset']['num_samples']}samples_{config['training']['num_train_epochs']}epochs_{time.strftime('%m-%d_%H-%M').replace('-', '_')}"

    # Output directory
    output_base = config["training"]["output_dir"]
    output_dir = os.path.join(output_base, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Set up environment
    os.environ["WANDB_PROJECT"] = config["project_name"]
    accelerator = Accelerator()

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
    # This is because I want to know which samples are used for training
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

    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    dataset = dataset.map(medlfqa_format, remove_columns=original_columns)

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, student_tokenizer, config),
        batched=True,
        num_proc=8,
        remove_columns=["text"],
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

    training_arguments = TrainingArguments(
        **config["training"],
        report_to=["wandb"],
    )

    wandb.init(
        project=config["project_name"],
        name=run_name,
        config={
            "teacher_model": config["models"]["teacher"],
            "student_model": config["models"]["student"],
            "distillation_method": config["distillation"]["method"],
            "alpha": config["distillation"]["alpha"],
            "temperature": config["distillation"]["temperature"],
            "num_samples": config["dataset"]["num_samples"],
            "num_epochs": config["training"]["num_train_epochs"],
        },
    )

    # Create the custom SFT Trainer
    trainer = LogitsTrainer(
        model=student_model,
        teacher_model=teacher_model,
        tokenizer=student_tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_arguments,
        config=config,
    )

    # Add the teacher model to the trainer
    trainer.teacher_model = teacher_model

    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
