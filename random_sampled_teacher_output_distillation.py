import os, time
import argparse
from dotenv import load_dotenv
from pathlib import Path

import torch, torch.nn.functional as F
from datetime import timedelta
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DefaultDataCollator,
    EarlyStoppingCallback,
)
from trl import SFTTrainer
from distillation_utils import (
    load_config,
    alpaca_format,
    dailydialog_format,
    pubmedqa_format,
    random_sampled_gsm8k_format,
    random_sampled_medqa_swe_format,
)
from distill_logits_final import tokenize_function
import wandb


# --------------------------------------------------------------------------- #
class SparseKDLossTrainer(SFTTrainer):
    """
    compute_loss() implements sparse KL on the fly:
      – each timestep uses the teacher's id list
      – gathers student logits for exactly those ids
      – computes forward‑KL
    """

    def __init__(
        self, *args, processing_class=None, distillation_config=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.processing_class = processing_class
        self.T = distillation_config["temperature"]
        print(f"Temperature is set to {self.T}")

        self._accumulated_loss = 0.0
        self._accumulation_step_count = 0

    def _answer_start(self, ids):
        eos = self.processing_class.eos_token_id
        for i, v in enumerate(ids):
            if v == eos:
                return i
        raise RuntimeError("EOS not found")

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        model = model.module if hasattr(model, "module") else model
        device = next(model.parameters()).device

        sparse_logits = inputs.pop("sparse_logits")  # list[list[list]]

        # ---- find start indices, filter invalid examples ---------------------
        # Check if EOS is present before calculating loss
        start_idxs = []
        valid_batch_indices = []
        for filtered_idx, ex_input_ids in enumerate(inputs["input_ids"]):
            try:
                start_idx = self._answer_start(ex_input_ids)
                start_idxs.append(start_idx)
                valid_batch_indices.append(filtered_idx)
            except RuntimeError as e:
                # Handle cases where EOS is not found more gracefully
                # This might happen due to truncation
                print(f"Warning: {e} in batch item {filtered_idx}. Skipping this item.")

        # Filter inputs to only include valid examples where EOS was found
        if len(valid_batch_indices) < len(inputs["input_ids"]):
            print(
                f"Filtering batch from {len(inputs['input_ids'])} to {len(valid_batch_indices)} due to missing EOS token."
            )
            # Select only the valid items from the batch tensors
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].shape[0] == len(
                    sparse_logits
                ):  # Check if it's a batch tensor
                    inputs[key] = inputs[key][valid_batch_indices]
            # Also filter the corresponding sparse_logits (requires it to be a list)
            sparse_logits = [sparse_logits[i] for i in valid_batch_indices]

        # If the entire batch becomes invalid after filtering, return a zero tensor loss
        if not valid_batch_indices:
            print(
                "Warning: Entire batch skipped due to missing EOS token in all items."
            )
            return torch.tensor(
                0.0, device=device, requires_grad=True
            )  # Requires grad to allow backward()
        # ---- filtering is done, only valid examples are kept -----------------

        # ---- get the student logits and compute the loss ---------------------
        inputs = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()
        }
        student_out = model(**inputs, use_cache=False)
        logits = student_out.logits  # Shape: (valid_batch_size, L, V)

        # sparse KD -> loop over batch & steps
        # initialize kd_loss as a tensor on the correct device
        kd_loss = torch.tensor(0.0, device=device, requires_grad=True)
        tot_tokens = 0

        # loop over the now-filtered batch
        for filtered_idx, original_index in enumerate(
            valid_batch_indices
        ):  # loop using filtered indices
            teacher_seq = sparse_logits[filtered_idx]
            start = start_idxs[
                filtered_idx
            ]  # Get the corresponding filtered start index

            # Add check: Ensure start index is actually within the sequence length
            # This could happen if EOS is the very last token at max_length
            if start >= logits.shape[1]:
                print(
                    f"Warning: Skipping item {original_index} (filtered index {filtered_idx}) because start index {start} >= seq length {logits.shape[1]}."
                )
                continue

            for t, token_pairs in enumerate(teacher_seq):
                current_logit_idx = start + t
                if current_logit_idx >= logits.shape[1]:
                    # Teacher sequence is longer than student's max length for this part
                    # print(f"Debug: Breaking loop for item {b}, timestep {t}. Index {current_logit_idx} >= {logits.shape[1]}")
                    break

                # unpack ids & logits
                # Check if token_pairs is empty (shouldn't happen)
                if not token_pairs:
                    print(
                        f"Warning: Empty token_pairs encountered for item {original_index}, timestep {t}. Skipping step."
                    )
                    continue

                ids, t_probs = zip(*token_pairs)

                # Check if ids is empty after unpacking (shouldn't happen)
                if not ids:
                    print(
                        f"Warning: Empty ids encountered for item {original_index}, timestep {t}. Skipping step."
                    )
                    continue

                ids = torch.tensor(ids, device=device, dtype=torch.long)
                t_probs = torch.tensor(t_probs, device=device)

                # student slice using the checked index
                # Need to use 'filtered_idx' for the batch dimension of logits now, as it's already filtered
                s_log = logits[filtered_idx, current_logit_idx, ids]

                # scale by temperature
                if not self.T == 1.0:
                    s_scaled = s_log / self.T
                    alpha = 1 / self.T
                    tmp = t_probs**alpha
                    t_probs_scaled = tmp / tmp.sum()
                else:
                    s_scaled = s_log
                    t_probs_scaled = t_probs

                # forward KL on this K‑vector
                kl = F.kl_div(
                    F.log_softmax(s_scaled, dim=-1),
                    t_probs_scaled,
                    reduction="sum",
                    log_target=False,
                )

                # Check for NaN/Inf KL divergence
                if torch.isnan(kl) or torch.isinf(kl):
                    print(
                        f"Warning: NaN/Inf KL divergence encountered for item {original_index}, timestep {t}. Skipping step."
                    )
                    # Potentially log more details: s_scaled, t_probs
                    continue

                kd_loss = kd_loss + kl  # Add tensors
                tot_tokens += 1

        # Avoid division by zero if no tokens were processed
        if tot_tokens > 0:
            kd_loss = kd_loss / tot_tokens * (self.T**2)
        else:
            # If no tokens were processed (e.g., all items skipped or had issues)
            # Return the zero tensor initialized earlier.
            print("Warning: No tokens processed in KD loss calculation for this batch.")

        if model.training:
            self._accumulated_loss += kd_loss.detach().item()
            self._accumulation_step_count += 1

        return (kd_loss, student_out) if return_outputs else kd_loss

    def log(self, logs, start_time=None):
        # For training steps with accumulated loss
        if self._accumulation_step_count > 0 and "loss" in logs:
            logs["loss"] = self._accumulated_loss / self._accumulation_step_count
            # Reset accumulators
            self._accumulated_loss = 0.0
            self._accumulation_step_count = 0

        # For evaluation, make sure eval_loss is captured
        is_eval = any(k.startswith("eval_") for k in logs.keys())

        if is_eval and "eval_loss" not in logs and "loss" in logs:
            logs["eval_loss"] = logs["loss"]

        super().log(logs)

    def prediction_step(self, model, inputs, prediction_loss_only, **_):
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)
        return (loss.detach(), None, None)  # we only care about the loss


# --------------------------------------------------------------------------- #
class SparseLogitsCollator(DefaultDataCollator):
    """keeps the nested list of sparse logits in the batch untouched"""

    def __call__(self, features):
        sparse = [f.pop("sparse_logits") for f in features]
        batch = super().__call__(features)
        batch["sparse_logits"] = sparse
        return batch


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train a student model with sparse logit distillation")
    parser.add_argument("--temperature", type=float, help="Temperature for distillation, overrides config")
    args = parser.parse_args()
    
    load_dotenv()
    cfg = load_config("random_sampling_config.yaml")
    dataset_name = cfg["dataset"]["name"]
    num_samples = cfg["dataset"].get("num_samples", None)
    dataset_config = load_config("datasets.yaml")[dataset_name]
    distillation_config = cfg["distillation"]
    
    # Override temperature if provided as argument
    if args.temperature is not None:
        print(f"Overriding temperature from config ({distillation_config['temperature']}) with command line value: {args.temperature}")
        distillation_config["temperature"] = args.temperature

    token = os.getenv("HF_TOKEN")
    accel = Accelerator(mixed_precision="bf16")

    # WARNING: currently only supports forward KL
    group_name = f"{dataset_name.split('/')[-1].replace('-', '_')}_"
    if num_samples is not None:
        group_name += f"{num_samples}s_"
    group_name += (
        f"{cfg['training']['num_train_epochs']}e_{distillation_config['temperature']}T_{time.strftime('%m_%d_%H_%M')}"
    )

    run_name = f"process_{accel.process_index}_{time.strftime('%m_%d_%H_%M')}"

    # Output directory
    output_base = cfg["training"]["output_dir"]
    output_dir = os.path.join(output_base, group_name)

    os.environ["WANDB_PROJECT"] = cfg["project_name"]

    # ---------- dataset & sparse logits ------------------------------------------------
    print("Loading dataset...")
    ds = (
        load_dataset(
            dataset_name,
            dataset_config.get("subset", None),
            split=dataset_config["split"],
        )
        if dataset_config.get("subset")
        else load_dataset(dataset_name, split=dataset_config["split"])
    )
    print("Dataset loaded, applying filtering if needed...")
    if dataset_name == "tatsu-lab/alpaca":
        ds = ds.filter(lambda x: x["input"] == "")  # keep alpaca‑style 0‑shot
    print("Loading generation config...")
    output_generation_config = cfg["output_generation"]
    print("Generation config loaded")
    
    print("Loading sparse logits...")
    if distillation_config.get("logits_path") is not None:
        print(f"Loading sparse logits from {distillation_config['logits_path']}")
        sparse = torch.load(distillation_config["logits_path"])
    else:
        sparse = torch.load(
            Path(output_generation_config["logits_dir"])
            / f"teacher_random_logits_{num_samples}_R{output_generation_config['draws']}_tau{output_generation_config['tau']}.pt"
        )
    print("Sparse logits loaded")
    if num_samples is not None:
        ds, sparse = ds.select(range(num_samples)), sparse[:num_samples]

    if len(ds) != len(sparse):
        raise ValueError("Dataset / sparse logits length mismatch")

    original_columns = ds.column_names
    print("Adding sparse logits to dataset...")
    ds = ds.add_column("sparse_logits", sparse)
    print("Sparse logits added to dataset")

    original_length = len(ds)
    print(f"Original dataset length: {original_length}")
    ds = ds.filter(lambda example: len(example["sparse_logits"]) > 0)
    filtered_length = len(ds)
    if original_length > filtered_length:
        print(
            f"Filtered dataset: Removed {original_length - filtered_length} examples with empty sparse logits."
        )
    print(f"Dataset length after filtering: {filtered_length}")

    tok = AutoTokenizer.from_pretrained(distillation_config["student"], token=token)
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"

    if dataset_name == "tatsu-lab/alpaca":
        ds = ds.map(
            lambda e: alpaca_format(e, tok),
            remove_columns=original_columns,
            load_from_cache_file=True,
        )  # use original columns for removal
    elif dataset_name == "roskoN/dailydialog":
        ds = ds.map(
            lambda e: dailydialog_format(e, tok),
            remove_columns=original_columns,
            load_from_cache_file=True,
        )
    elif dataset_name == "qiaojin/PubMedQA":
        ds = ds.map(
            lambda e: pubmedqa_format(e, tok),
            remove_columns=original_columns,
            load_from_cache_file=True,
        )
    elif dataset_name == "openai/gsm8k":
        ds = ds.map(
            lambda e: random_sampled_gsm8k_format(e, tok),
            remove_columns=original_columns,
            load_from_cache_file=False,
        )
    elif dataset_name == "kaans/medqa-swe-with-responses":
        ds = ds.map(
            lambda e: random_sampled_medqa_swe_format(e, tok),
            remove_columns=original_columns,
            load_from_cache_file=False,
        )
    
    print(" #### DEBUG: printing first three examples ################################")
    for i in range(3):
        print(ds[i]["text"])
        print("---------------------------------------------------------------")
    print(" ###################################################################")
    
    ds = ds.map(
        lambda e: tokenize_function(e, tok, distillation_config["max_length"]),
        batched=True,
        num_proc=8,
        remove_columns=[],
        load_from_cache_file=True,
    )

    # split
    test_size = 0.1
    ds = ds.train_test_split(test_size)

    # ---------- model & trainer --------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        distillation_config["student"],
        torch_dtype=torch.bfloat16,
        attn_implementation=(
            "flash_attention_2" if distillation_config["use_flash_attention"] else None
        ),
    )

    # reduce activation memory during backward pass
    model.gradient_checkpointing_enable()

    training_args_dict = cfg["training"].copy()
    training_args_dict["output_dir"] = output_dir
    training_args_dict["report_to"] = ["wandb"]
    # 8‑bit optimizer to keep states off the GPU
    training_args_dict["optim"] = "paged_adamw_8bit"
    early_stopping_patience = training_args_dict.pop("early_stopping_patience", 1)
    training_args = TrainingArguments(**training_args_dict, remove_unused_columns=False)

    # Configure callbacks
    callbacks = []

    # Add early stopping if configured
    if training_args.load_best_model_at_end:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )

    if accel.is_main_process:
        wandb.init(
            project=cfg["project_name"],
            name=run_name,
            group=group_name,
            config={
                "teacher_model": output_generation_config["model"],
                "student_model": distillation_config["student"],
                "temperature": distillation_config["temperature"],
                "kl_divergence": distillation_config["kl_divergence"],
                "num_samples": num_samples,
                "num_epochs": cfg["training"]["num_train_epochs"],
                "group_name": group_name,
                "training_args": training_args_dict,
                "early_stopping": {
                    "enabled": training_args.load_best_model_at_end,
                    "patience": early_stopping_patience,
                    "metric": training_args.metric_for_best_model,
                    "greater_is_better": training_args.greater_is_better,
                },
            },
            reinit=False,
        )

    trainer = SparseKDLossTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=SparseLogitsCollator(),
        args=training_args,
        distillation_config=distillation_config,
        callbacks=callbacks,
    )

    trainer = accel.prepare(trainer)
    trainer.train(resume_from_checkpoint=cfg["training"]["resume_from_checkpoint"])
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
