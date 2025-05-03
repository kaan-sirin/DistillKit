import os, time
from pathlib import Path
from dotenv import load_dotenv

import torch, torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DefaultDataCollator,
)
from trl import SFTTrainer
from distillation_utils import load_config, alpaca_format, dailydialog_format
from distill_logits_final import tokenize_function  


# --------------------------------------------------------------------------- #
class SparseKDLossTrainer(SFTTrainer):
    """
    compute_loss() implements sparse KL on the fly:
      – each timestep uses the teacher's id list
      – gathers student logits for exactly those ids
      – computes forward‑KL
    """

    def __init__(self, *args, processing_class=None, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_class = processing_class
        self.cfg = config
        self.T = config["distillation"]["temperature"]
        self.alpha = config["distillation"]["alpha"]

    # ------------------------------------------------------------------ utils
    def _answer_start(self, ids):
        eos = self.processing_class.eos_token_id
        for i, v in enumerate(ids):
            if v == eos:
                return i
        raise RuntimeError("EOS not found")

    # --------------------------------------------------------------- training
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        device = model.module.device  # Access device through .module

        sparse_logits = inputs.pop("sparse_logits")  # list[list[list]]
        
        # compute answer start indices
        # Check if EOS is present before calculating loss
        start_idxs = []
        valid_batch_indices = [] # Keep track of examples that are valid
        for i, ex_input_ids in enumerate(inputs["input_ids"]):
            try:
                start_idx = self._answer_start(ex_input_ids)
                start_idxs.append(start_idx)
                valid_batch_indices.append(i)
            except RuntimeError as e:
                # Handle cases where EOS is not found more gracefully
                # This might happen due to truncation
                print(f"Warning: {e} in batch item {i}. Skipping this item.")
                # We won't add an index to start_idxs or valid_batch_indices


        # Filter inputs to only include valid examples where EOS was found
        # This requires careful handling if using features other than input_ids/attention_mask
        if len(valid_batch_indices) < len(inputs["input_ids"]):
             print(f"Filtering batch from {len(inputs['input_ids'])} to {len(valid_batch_indices)} due to missing EOS.")
             # Select only the valid items from the batch tensors
             # Assuming standard inputs like input_ids, attention_mask
             for key in inputs:
                 if torch.is_tensor(inputs[key]) and inputs[key].shape[0] == len(sparse_logits): # Check if it's a batch tensor
                     inputs[key] = inputs[key][valid_batch_indices]
             # Also filter the corresponding sparse_logits (requires it to be a list)
             sparse_logits = [sparse_logits[i] for i in valid_batch_indices]

        # If the entire batch becomes invalid after filtering, return a zero tensor loss
        if not valid_batch_indices:
             print("Warning: Entire batch skipped due to missing EOS in all items.")
             # Need to return a tensor connected to the model's graph if possible,
             # otherwise, return a simple zero tensor. Getting a dummy output might be safest.
             # If model wasn't run, need a different approach. Let's just return 0 tensor for now.
             # This might still cause issues if no valid batch ever occurs.
             return torch.tensor(0.0, device=device, requires_grad=True) # Requires grad to allow backward()

        # Run the model only on the valid inputs
        student_out = model(**inputs, use_cache=False)
        logits = student_out.logits  # Shape: (valid_batch_size, L, V)

        # ----------------------------- sparse KD (loop over batch & steps)
        # Initialize kd_loss as a Tensor on the correct device
        kd_loss = torch.tensor(0.0, device=device, requires_grad=True) 
        tot_tokens = 0
        
        # Loop over the now-filtered batch
        for i, b in enumerate(valid_batch_indices): # Loop using filtered indices
            teacher_seq = sparse_logits[i] # Get the corresponding filtered sparse logits
            start = start_idxs[i]          # Get the corresponding filtered start index

            # Add check: Ensure start index is actually within the sequence length
            # This could happen if EOS is the very last token at max_length
            if start >= logits.shape[1]: 
                 print(f"Warning: Skipping item {b} (filtered index {i}) because start index {start} >= seq length {logits.shape[1]}.")
                 continue

            for t, token_pairs in enumerate(teacher_seq):
                # <<< --- Boundary Check --- >>>
                current_logit_idx = start + t
                # Check if the target index is within the student's logit sequence length
                if current_logit_idx >= logits.shape[1]: 
                    # Teacher sequence is longer than student's max length for this part
                    # print(f"Debug: Breaking loop for item {b}, timestep {t}. Index {current_logit_idx} >= {logits.shape[1]}")
                    break 
                # <<< -------------------- >>>

                # unpack ids & logits
                # Check if token_pairs is empty (shouldn't happen if pre-filtered)
                if not token_pairs:
                     print(f"Warning: Empty token_pairs encountered for item {b}, timestep {t}. Skipping step.")
                     continue
                
                ids, t_probs = zip(*token_pairs)
                
                # Check if ids is empty after unpacking (if token_pairs was [()])
                if not ids:
                    print(f"Warning: Empty ids encountered for item {b}, timestep {t}. Skipping step.")
                    continue

                ids = torch.tensor(ids, device=device, dtype=torch.long)
                t_probs = torch.tensor(t_probs, device=device)

                # student slice using the checked index
                # Need to use 'i' for the batch dimension of logits now, as it's filtered
                s_log = logits[i, current_logit_idx, ids] 

                # scale by temperature
                s_scaled = s_log / self.T

                # forward KL on this K‑vector
                kl = F.kl_div(
                    F.log_softmax(s_scaled, dim=-1),
                    t_probs,
                    reduction="sum",
                    log_target=False,
                )
                
                # Check for NaN/Inf KL divergence
                if torch.isnan(kl) or torch.isinf(kl):
                    print(f"Warning: NaN/Inf KL divergence encountered for item {b}, timestep {t}. Skipping step.")
                    # Potentially log more details: s_scaled, t_probs
                    continue

                kd_loss = kd_loss + kl # Add tensors
                tot_tokens += 1

        # Avoid division by zero if no tokens were processed
        if tot_tokens > 0:
             kd_loss = kd_loss / tot_tokens * (self.T**2)
        else:
             # If no tokens were processed (e.g., all items skipped or had issues)
             # Return the zero tensor initialized earlier.
             print("Warning: No tokens processed in KD loss calculation for this batch.")
             # kd_loss is already torch.tensor(0.0, ...)

        total = kd_loss
        return (total, student_out) if return_outputs else total


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
    load_dotenv()
    cfg = load_config("random_sampling_config.yaml")
    token = os.getenv("HF_TOKEN")
    accel = Accelerator()

    # ---------- dataset & sparse logits ------------------------------------------------
    ds = (
        load_dataset(
            cfg["dataset"]["name"],
            cfg["dataset"]["subset"],
            split=cfg["dataset"]["split"],
        )
        if cfg["dataset"].get("subset")
        else load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])
    )
    if cfg["dataset"]["name"] == "tatsu-lab/alpaca":    
        ds = ds.filter(lambda x: x["input"] == "")  # keep alpaca‑style 0‑shot

    sparse = torch.load(cfg["dataset"]["teacher_data"]["logits_path"])
    if "num_samples" in cfg["dataset"]:
        n = cfg["dataset"]["num_samples"]
        ds, sparse = ds.select(range(n)), sparse[:n]

    if len(ds) != len(sparse):
        raise ValueError("Dataset / sparse logits length mismatch")

    original_columns = ds.column_names
    ds = ds.add_column("sparse_logits", sparse)
    
    original_length = len(ds)
    print(f"Original dataset length: {original_length}")
    ds = ds.filter(lambda example: len(example['sparse_logits']) > 0)
    filtered_length = len(ds)
    if original_length > filtered_length:
        print(f"Filtered dataset: Removed {original_length - filtered_length} examples with empty sparse logits.")
    print(f"Dataset length after filtering: {filtered_length}")
    
    
    tok = AutoTokenizer.from_pretrained(cfg["models"]["student"], token=token)
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"


    if cfg["dataset"]["name"] == "tatsu-lab/alpaca":
        ds = ds.map(
            lambda e: alpaca_format(e, tok),
            remove_columns=original_columns,
            load_from_cache_file=True,
        )  # use original columns for removal
    elif cfg["dataset"]["name"] == "roskoN/dailydialog":
        ds = ds.map(
            lambda e: dailydialog_format(e, tok),
            remove_columns=original_columns,
            load_from_cache_file=True,
        )
    ds = ds.map(
        lambda e: tokenize_function(e, tok, cfg),
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
        cfg["models"]["student"],
        torch_dtype=torch.bfloat16,
        attn_implementation=(
            "flash_attention_2" if cfg["model_config"]["use_flash_attention"] else None
        ),
    )
    args = TrainingArguments(
        **cfg["training"], remove_unused_columns=False, report_to="wandb"
    )

    trainer = SparseKDLossTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=SparseLogitsCollator(),
        args=args,
        config=cfg,
    )

    trainer = accel.prepare(trainer)
    trainer.train(resume_from_checkpoint=cfg["training"]["resume_from_checkpoint"])
    trainer.save_model(
        Path(cfg["training"]["output_dir"])
        / f"sparse_kd_student_{time.strftime('%Y%m%d_%H%M%S')}"
    )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
