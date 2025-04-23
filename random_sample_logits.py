###############################################################################
#  generate_random_sampled_logits.py
#  -------------------------------------------------
#  Save ≈K‑unique token–probability pairs per generated step using the
#  unbiased importance‑sampling method described in “Sparse‑Logit Sampling”.
#  ✔ unbiased  ✔ constant memory (no full‑vocab softmax on GPU)
###############################################################################
from pathlib import Path
import os
from tqdm import tqdm
import torch
from torch import amp
from datasets import load_dataset
from batch_comparison import setup, load_model
from distillation_utils import load_config

# --------------------------------------------------------------------------- #
def random_sample_distribution(logits, draws=50, tau=1.0):
    """
    Args
    ----
    logits : 1‑D tensor (vocab,)
    draws  : R  – number of proposal samples (with replacement)
    tau    : τ – sampling temperature for proposal q_i ∝ p_i^τ
    Returns
    -------
    ids    : tensor(K)          unique token ids
    probs  : tensor(K)          unbiased teacher probs  (sum = 1)
    """
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)                    # teacher p_i
        q = torch.pow(p, tau)                                # proposal
        q = q / q.sum()                                      # normalize q

        idx = torch.multinomial(q, draws, replacement=True)  # R samples with replacement
        # idx is a tensor of shape (R,) containing the indices of the sampled tokens
        # accumulate importance weights 
        # when tau = 1, this will be equal to how many times the token was sampled
        w = torch.zeros_like(p).scatter_add_(0, idx, p[idx] / q[idx])
        ids = (w > 0).nonzero(as_tuple=True)[0]              # unique ids
        probs = w[ids]
        probs = probs / probs.sum()                          # normalise
        return ids, probs

# --------------------------------------------------------------------------- #
def generate_and_save_random_sampled_logits(
        model_name: str,
        dataset_name: str,
        *,
        dataset_subset=None,
        dataset_split="train",
        system_prompt=None,
        num_samples=None,
        max_new_tokens=256,
        batch_size=1,
        draws=50,        # R – controls expected ~K ≃ 12 unique tokens
        tau=1.0,
        output_dir=None,
        debug=False,
):
    logger = setup()
    logger.info(f"[RSKD] Generating sampled logits from {model_name}")

    # ------------------------------------------------------------------ data
    ds = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    ds = ds.filter(lambda x: x["input"] == "")               # alpaca quirk
    num_samples = min(len(ds), num_samples or len(ds))

    model, tok = load_model(model_name)
    tok.pad_token = tok.pad_token or tok.eos_token
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")

    # ------------------------------------------------------------------ out
    out_dir = Path(output_dir or
                   f"distillation_data/{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_steps = []  # list[list[ list[(id,prob)] ]]

    # ------------------------------------------------------------------ loop
    for b0 in tqdm(range(0, num_samples, batch_size),
                   desc=f"Random‑sampling {num_samples} ex"): #b0 is the batch start index
        b1 = min(b0 + batch_size, num_samples) #b1 is the batch end index
        prompts = []
        end_marker = "<|eot_id|>"
        for ex in ds.select(range(b0, b1)):
            prompt = (system_prompt or "") + ex["instruction"] + end_marker
            prompts.append(prompt)

        inp = tok(prompts, padding=True, return_tensors="pt").to(model.device)

        with amp.autocast("cuda"):
            gen = model.generate(**inp,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False, num_beams=1,
                                 return_dict_in_generate=True, output_scores=True)
            
        
        
        for seq_i in range(len(prompts)):
            step_pairs = []                               # this sequence
            for t, score in enumerate(gen.scores):        # t = token position, score = logits (batch_size, vocab_size) ie. all logits for this token position
                if score[seq_i].argmax().item() == eot_id:
                    break
                ids, probs = random_sample_distribution(
                    score[seq_i], draws=draws, tau=tau)
                step_pairs.append([(int(i), float(p)) for i, p in zip(ids, probs)]) # ids = unique token ids, probs = unbiased teacher probs (sum = 1)
            
            if debug and b0 == 0:
                print(f"\n############# DEBUGGING SEQ {seq_i} #############")
                for token_prob_pairs in (step_pairs):
                    token_prob_pairs.sort(key=lambda x: x[1], reverse=True)
                    print(f"{tok.decode(token_prob_pairs[0][0], skip_special_tokens=False)}", end="")
                print()
                print("-"*100)
                token_num = 0
                for token_prob_pairs in (step_pairs):
                    token_num += len(token_prob_pairs)
                print(f"Average number of tokens: {token_num / len(step_pairs)}")
                print(f"###########################################\n")
            all_steps.append(step_pairs) 
            
            

    # ------------------------------------------------------------------ save
    save_path = out_dir / f"teacher_random_logits_{num_samples}_R{draws}_tau{tau}.pt"
    torch.save(all_steps, save_path)
    logger.info(f"[RSKD] wrote {len(all_steps)} sequences → {save_path}")
    return all_steps

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = load_config("experimental_config.yaml")
    generate_and_save_random_sampled_logits(
        model_name=cfg["models"]["teacher"],
        dataset_name=cfg["dataset"]["name"],
        dataset_split=cfg["dataset"]["split"],
        system_prompt=cfg["dataset"]["teacher_data"].get("system_prompt", None),
        num_samples=cfg["dataset"]["num_samples"],
        max_new_tokens=cfg["dataset"]["teacher_data"]["max_new_tokens"],
        batch_size=8,
        draws=10000, # 300 averages out to 12 unique tokens per step
        tau=1.0,
        output_dir=Path(cfg["dataset"]["teacher_data"]["logits_path"]).parent,
        debug=True,
    )