###############################################################################
#  distill_teacher_outputs_sparse.py
#  ---------------------------------
#  Train a student LLM from RANDOM‑SAMPLING teacher logits.
#  Differences vs. your original file:
#    • no expand_teacher_sparse_representation
#    • no match_student_logits_to_teacher
#    • KL is computed step‑by‑step on the sparse id set
###############################################################################
import os, time
from pathlib import Path
from dotenv import load_dotenv

import torch, torch.nn.functional as F
from torch.utils.data import random_split
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DefaultDataCollator,
)
from trl import SFTTrainer

from distillation_utils import load_config, alpaca_format
from distill_logits_final import tokenize_function  # same helper as before


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

        # print(f"\nPRINTING INPUTS")
        # print(inputs['input_ids'][0, :100])
        # for token in inputs['input_ids'][0, :10]:
        #     print(self.processing_class.decode(token, skip_special_tokens=False), end="")
        # print()

        sparse_logits = inputs.pop("sparse_logits")  # list[list[list]]
        labels = inputs["input_ids"].clone()

        # predict‑next‑token shift & ignore last
        labels[:, :-1] = inputs["input_ids"][:, 1:]
        labels[:, -1] = -100

        # compute answer start indices
        start_idxs = [self._answer_start(ex) for ex in inputs["input_ids"]]

        # mask CE so it is computed only on teacher‑provided positions
        for b, teacher_seq in enumerate(sparse_logits):
            s = start_idxs[b]
            t_len = len(teacher_seq)
            labels[b, :s] = -100
            labels[b, s + t_len :] = -100

        inputs["labels"] = labels

        # print(inputs['labels'][0, :100])
        # print()
        # print(inputs['attention_mask'][0][:100])
        # print(inputs['attention_mask'][0][-100:])

        student_out = model(**inputs, use_cache=False)
        ce_loss = student_out.loss  # already mean‑reduced
        logits = student_out.logits  # B × L × V

        # ----------------------------- sparse KD (loop over batch & steps)
        kd_loss, tot_tokens = 0.0, 0
        for b, teacher_seq in enumerate(sparse_logits):
            start = start_idxs[b]
            for t, token_pairs in enumerate(teacher_seq):
                # unpack ids & logits
                ids, t_probs = zip(*token_pairs)
                ids = torch.tensor(ids, device=device, dtype=torch.long)
                t_probs = torch.tensor(t_probs, device=device)

                # for i in range(5):
                #     s_log_full = logits[b, start + t + i, :]
                #     s_log_full_top_3_vals = torch.topk(s_log_full, 3, dim=-1).values
                #     s_log_full_top_3_ids = torch.topk(s_log_full, 3, dim=-1).indices
                #     print(f"-> s_log_full_top_3: {s_log_full_top_3_vals}")
                #     # get the token ids and print the decoded tokens
                #     for id in s_log_full_top_3_ids:
                #         print(self.processing_class.decode(id, skip_special_tokens=False), end=", ")
                #     print()

                # student slice
                s_log = logits[b, start + t, ids]

                # scale by temperature
                s_scaled = s_log / self.T

                # reductions = ["batchmean", "sum", "mean", "none"]
                # for reduction in reductions:
                #     kl_test = F.kl_div(
                #         F.log_softmax(s_scaled, dim=-1),
                #         t_probs,
                #         reduction=reduction,
                #         log_target=False,
                #     )
                #     print(f"Reduction '{reduction}': {kl_test}")
                #     print("-" * 100)
                

                # forward KL on this K‑vector
                kl = F.kl_div(
                    F.log_softmax(s_scaled, dim=-1),
                    t_probs,
                    reduction="sum",
                    log_target=False,
                )
                kd_loss += kl
                tot_tokens += 1

        kd_loss = kd_loss / max(1, tot_tokens) * (self.T**2)
       
        # total = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        total = kd_loss  # TODO: WARNING, FOR DEBUGGING ONLY
        return (total, student_out) if return_outputs else total


# --------------------------------------------------------------------------- #
class SparseLogitsCollator(DefaultDataCollator):
    """keeps the list‑of‑lists in the batch untouched"""

    def __call__(self, features):
        sparse = [f.pop("sparse_logits") for f in features]
        batch = super().__call__(features)
        batch["sparse_logits"] = sparse
        return batch


# --------------------------------------------------------------------------- #
def main():
    load_dotenv()
    cfg = load_config("experimental_config.yaml")
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
    ds = ds.filter(lambda x: x["input"] == "")  # keep alpaca‑style 0‑shot

    sparse = torch.load(cfg["dataset"]["teacher_data"]["logits_path"])
    if "num_samples" in cfg["dataset"]:
        n = cfg["dataset"]["num_samples"]
        ds, sparse = ds.select(range(n)), sparse[:n]

    if len(ds) != len(sparse):
        raise ValueError("Dataset / sparse logits length mismatch")

    original_columns = ds.column_names
    ds = ds.add_column("sparse_logits", sparse)
    tok = AutoTokenizer.from_pretrained(cfg["models"]["student"], token=token)
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"

    # alpaca formatting & tokenisation
    ds = ds.map(
        lambda e: alpaca_format(e, tok),
        remove_columns=original_columns,
        load_from_cache_file=False,
    )  # use original columns for removal
    ds = ds.map(
        lambda e: tokenize_function(e, tok, cfg),
        batched=True,
        num_proc=8,
        remove_columns=[],
        load_from_cache_file=False,
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
