import os
import torch
import torch.nn.functional as F
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


class MultiLayerAdaptationLayer(torch.nn.Module):
    def __init__(self, student_dim, teacher_dim, num_student_layers, num_teacher_layers, dtype=torch.bfloat16):
        super().__init__()
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers)
        self.dtype = dtype

    def create_layer_mapping(self, num_student_layers, num_teacher_layers):
        return {
            i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
            for i in range(num_student_layers)
        }

    def forward(self, student_hidden_states):
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states


class HiddenStatesTrainer(SFTTrainer):
    def __init__(
        self, model, teacher_model=None, adaptation_layer=None, tokenizer=None, config=None, *args, **kwargs
    ):
        super().__init__(model=model, *args, **kwargs)
        self.config = config
        self.teacher_model = teacher_model
        self.adaptation_layer = adaptation_layer
        self.tokenizer = tokenizer

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if hasattr(model, "device"):
            device = model.device
        elif hasattr(model, "module"):
            device = model.module.device

        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        self.teacher_model = self.teacher_model.to(device)
        self.adaptation_layer = self.adaptation_layer.to(device)

        student_model = model.module if hasattr(model, "module") else model
        teacher_model = (
            self.teacher_model.module
            if hasattr(self.teacher_model, "module")
            else self.teacher_model
        )

        student_outputs = student_model(**inputs, output_hidden_states=True)
        
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs, output_hidden_states=True)

        total_loss, loss_components = self.distillation_loss(
            student_outputs.hidden_states, 
            teacher_outputs.hidden_states, 
            inputs, 
            student_outputs.loss
        )

        self._current_loss_kd = loss_components["loss_kd"].detach().item()
        self._current_original_loss = loss_components["original_loss"].detach().item()

        return (total_loss, student_outputs) if return_outputs else total_loss

    def distillation_loss(self, student_hidden_states, teacher_hidden_states, inputs, original_loss):
        adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)

        total_loss_kd = 0
        for student_idx, teacher_idx in self.adaptation_layer.layer_mapping.items():
            teacher_hidden = teacher_hidden_states[teacher_idx]
            adapted_student_hidden = adapted_student_hidden_states[student_idx]
            
            if adapted_student_hidden.shape != teacher_hidden.shape:
                raise ValueError(f"Shape mismatch: student {adapted_student_hidden.shape} vs teacher {teacher_hidden.shape}")

            student_probs = F.softmax(adapted_student_hidden / self.config["distillation"]["temperature"], dim=-1)
            teacher_probs = F.softmax(teacher_hidden / self.config["distillation"]["temperature"], dim=-1)

            loss_kd = F.kl_div(
                F.log_softmax(adapted_student_hidden / self.config["distillation"]["temperature"], dim=-1),
                teacher_probs,
                reduction='batchmean'
            ) * (self.config["distillation"]["temperature"] ** 2)

            total_loss_kd += loss_kd

        avg_loss_kd = total_loss_kd / len(self.adaptation_layer.layer_mapping)
        hidden_dim = adapted_student_hidden_states[0].size(-1)
        scaled_loss_kd = avg_loss_kd / hidden_dim

        total_loss = self.config["distillation"]["alpha"] * scaled_loss_kd + (1 - self.config["distillation"]["alpha"]) * original_loss

        return total_loss, {"loss_kd": scaled_loss_kd, "original_loss": original_loss}

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
    config["distillation"]["method"] = "hidden_states"  # Ensure method is set correctly
    
    run_name = f"distill_hidden_{config['dataset']['name'].split('/')[-1].replace('-', '_')}_{config['dataset']['num_samples']}samples_{config['training']['num_train_epochs']}epochs_{time.strftime('%m-%d_%H-%M').replace('-', '_')}"

    # Output directory
    output_base = config["training"]["output_dir"]
    output_dir = os.path.join(
        output_base, run_name
    )
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
    dataset = dataset.map(medqa_format, remove_columns=original_columns)

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
        config["models"]["teacher"], token=HF_TOKEN, **model_kwargs
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"], token=HF_TOKEN, **model_kwargs
    )

    # Create adaptation layer
    adaptation_layer = MultiLayerAdaptationLayer(
        student_model.config.hidden_size,
        teacher_model.config.hidden_size,
        student_model.config.num_hidden_layers,
        teacher_model.config.num_hidden_layers,
        dtype=torch.bfloat16
    )

    training_config = config["training"].copy()
    training_config["output_dir"] = output_dir

    training_arguments = TrainingArguments(
        **training_config,
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
    trainer = HiddenStatesTrainer(
        model=student_model,
        teacher_model=teacher_model,
        adaptation_layer=adaptation_layer,
        tokenizer=student_tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_arguments,
        config=config,
    )

    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

    # Save the final model and adaptation layer
    trainer.save_model(output_dir)
    torch.save(adaptation_layer.state_dict(), os.path.join(output_dir, "adaptation_layer.pth"))

if __name__ == "__main__":
    main() 