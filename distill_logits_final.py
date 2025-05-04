import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
import wandb
from dotenv import load_dotenv
from distillation_utils import (
    load_config,
    code_alpaca_format,
    reverse_kld,
    tokenize_function,
)
import time

# --- seems necessary for it to work on the cluster for the 70B model
from transformers import BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(
    # load_in_4bit=True,  # 4‑bit weights
    # bnb_4bit_quant_type="nf4",  # best accuracy
    # bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_use_double_quant=True,
    # after moving teacher model to CPU, needed to avoid OOM (?)
    load_in_8bit=True,  # 8‑bit weights
    llm_int8_enable_fp32_cpu_offload=True,  # keep weights on CPU, cast to fp32 when needed
)
# ------------------------------------------------------------


class LogitsTrainer(SFTTrainer):
    def __init__(
        self,
        model,
        teacher_model=None,
        tokenizer=None,
        config=None,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, *args, **kwargs)
        self.config = config
        self.teacher_model = teacher_model
        self.processing_class = tokenizer
        self.use_reverse_kld = (
            True if config["distillation"]["kl_divergence"] == "reverse" else False
        )
        # Initialize accumulators for averaging losses over accumulation steps
        self._accumulated_loss_kd = 0.0
        self._accumulated_original_loss = 0.0
        self._accumulation_step_count = 0

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if hasattr(model, "device"):
            device = model.device
        elif hasattr(model, "module"):
            # For DataParallel models, use the device of the module inside
            device = model.module.device

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        # self.teacher_model = self.teacher_model.to(device) - commented out to avoid memory issues (cluster, 70B model)

        student_model = model.module if hasattr(model, "module") else model
        teacher_model = (
            self.teacher_model.module
            if hasattr(self.teacher_model, "module")
            else self.teacher_model
        )

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            
        # --- making sure everything is on the student's device -----------------
        teacher_logits = teacher_outputs.logits.to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        # ---------------------------------------------------------------------

        total_loss, loss_components = self.distillation_loss(
            student_outputs.logits, teacher_logits, inputs, student_outputs.loss
        )

        # Accumulate losses for logging average over gradient accumulation steps
        if student_model.training:
            self._accumulated_loss_kd += loss_components["loss_kd"].detach().item()
            self._accumulated_original_loss += (
                loss_components["original_loss"].detach().item()
            )
            self._accumulation_step_count += 1

        return (total_loss, student_outputs) if return_outputs else total_loss

    def distillation_loss(self, student_logits, teacher_logits, inputs, original_loss):

        student_logits_scaled = (
            student_logits / self.config["distillation"]["temperature"]
        )
        teacher_logits_scaled = (
            teacher_logits / self.config["distillation"]["temperature"]
        )

        kld = None
        if self.use_reverse_kld:
            kld = reverse_kld(
                student_logits_scaled, teacher_logits_scaled, reduction="none"
            )
        else:
            kld = F.kl_div(
                F.log_softmax(student_logits_scaled, dim=-1),
                F.softmax(teacher_logits_scaled, dim=-1),
                reduction="none",
            )

        if kld is None:
            raise ValueError("KL divergence is not calculated")

        kl_per_token = kld.sum(dim=-1)
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask = mask.to(kl_per_token.dtype)
            total_kl = (kl_per_token * mask).sum()
            num_tokens = mask.sum().clamp(min=1)
            loss_kd = (
                total_kl
                / num_tokens
                * (self.config["distillation"]["temperature"] ** 2)
            )
        else:
            loss_kd = kl_per_token.mean() * (
                self.config["distillation"]["temperature"] ** 2
            )

        total_loss = (
            self.config["distillation"]["alpha"] * loss_kd
            + (1 - self.config["distillation"]["alpha"]) * original_loss
        )

        return total_loss, {"loss_kd": loss_kd, "original_loss": original_loss}

    def log(self, logs, start_time=None):
        # Only process and reset when we have a real training loss to average
        if self._accumulation_step_count > 0 and "loss" in logs:
            logs["loss_kd"] = self._accumulated_loss_kd / self._accumulation_step_count
            logs["original_loss"] = (
                self._accumulated_original_loss / self._accumulation_step_count
            )
            logs["loss"] = logs["loss"] / self._accumulation_step_count
            # Reset accumulators for the next training cycle
            self._accumulated_loss_kd = 0.0
            self._accumulated_original_loss = 0.0
            self._accumulation_step_count = 0
        else:
            # In evaluation (or no training steps yet), only ensure our custom metrics exist
            if "loss_kd" not in logs:
                logs["loss_kd"] = float("nan")
            if "original_loss" not in logs:
                logs["original_loss"] = float("nan")

        super().log(logs)


def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    config = load_config()
    dataset_name = config["dataset"]
    dataset_config = load_config("datasets.yaml")[dataset_name]

    # Set up environment
    os.environ["WANDB_PROJECT"] = config["project_name"]
    accelerator = Accelerator()

    group_name = f"{config['dataset'].split('/')[-1].replace('-', '_')}_{config['distillation']['kl_divergence']}_{dataset_config['num_samples']}_samples_{config['training']['num_train_epochs']}_epochs_{time.strftime('%m_%d_%H_%M')}"
    run_name = f"process_{accelerator.process_index}_{time.strftime('%m_%d_%H_%M')}"

    # Output directory
    output_base = config["training"]["output_dir"]
    output_dir = os.path.join(output_base, group_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess dataset
    print(f"Process {accelerator.process_index}: Loading dataset...")
    dataset = (
        load_dataset(
            dataset_name,
            dataset_config["subset"],
            split=dataset_config["split"],
        )
        if dataset_config.get("subset")
        else load_dataset(dataset_name, split=dataset_config["split"])
    )

    # First take a subset of the dataset if specified, then shuffle
    # This is because I want to know which samples are used for training
    if "num_samples" in dataset_config:
        dataset = dataset.select(range(dataset_config["num_samples"]))
    dataset = dataset.shuffle(seed=dataset_config["seed"])

    # Load tokenizers
    print(f"Process {accelerator.process_index}: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], token=HF_TOKEN
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Preprocess and tokenize the dataset
    print(
        f"Process {accelerator.process_index}: Preprocessing and tokenizing dataset..."
    )
    original_columns = dataset.column_names
    dataset = dataset.map(code_alpaca_format, remove_columns=original_columns)
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, config),
        batched=True,
        num_proc=8,
        remove_columns=["text"],
    )
    if "test_size" in dataset_config:
        tokenized_dataset = tokenized_dataset.train_test_split(
            test_size=dataset_config["test_size"]
        )

    print(
        f"Process {accelerator.process_index}: Dataset preparation complete. Loading models..."
    )
    # Load models with configurable flash attention
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"Process {accelerator.process_index}: Loading teacher model...")
    # --- seems necessary for it to work on the cluster for the 70B model
    max_memory = {
        0: "38GiB",
        1: "38GiB",
        "cpu": "160GiB",
    }  # anything beyond 38 GiB goes to RAM

    teacher_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["teacher"],
        quantization_config=bnb_cfg,
        # device_map="balanced",  # fill GPUs evenly, overflow -> “cpu”
        device_map={"": "cpu"},  # <‑‑ force *all* layers to CPU
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    teacher_model.eval().requires_grad_(False)
    # ------------------------------------------------------------

    print(f"Process {accelerator.process_index}: Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"], **model_kwargs
    )

    training_args_dict = config["training"].copy()
    training_args_dict["output_dir"] = output_dir  # Ensure output_dir is set
    training_args_dict["report_to"] = ["wandb"]
    training_arguments = TrainingArguments(**training_args_dict)

    wandb.init(
        project=config["project_name"],
        name=run_name,
        group=group_name,
        config={
            "teacher_model": config["models"]["teacher"],
            "student_model": config["models"]["student"],
            "distillation_method": config["distillation"]["method"],
            "alpha": config["distillation"]["alpha"],
            "temperature": config["distillation"]["temperature"],
            "num_samples": dataset_config["num_samples"],
            "num_epochs": config["training"]["num_train_epochs"],
            "group_name": group_name,
            "training_args": training_args_dict,
        },
        reinit=False,
    )

    # Create the custom SFT Trainer
    trainer = LogitsTrainer(
        model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_arguments,
        config=config,
    )

    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
