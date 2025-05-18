from pathlib import Path
from tqdm import tqdm
import torch
from torch import amp
from datasets import load_dataset
from batch_comparison import setup, load_model
from distillation_utils import load_config, format_dialog
import argparse

# --------------------------------------------------------------------------- #
def random_sample_distribution(logits, draws=50, tau=1.0):
    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)  # teacher p_i
        q = torch.pow(p, tau)  # proposal
        q = q / q.sum()  # normalize q

        idx = torch.multinomial(
            q, draws, replacement=True
        )  # R samples with replacement
        # idx is a tensor of shape (R,) containing the indices of the sampled tokens
        # accumulate weights
        # when tau = 1, this will be equal to how many times the token was sampled
        w = torch.zeros_like(p).scatter_add_(0, idx, p[idx] / q[idx])
        ids = (w > 0).nonzero(as_tuple=True)[0]  # unique ids
        probs = w[ids]
        probs = probs / probs.sum()  # normalise
        return ids, probs


# --------------------------------------------------------------------------- #
def generate_and_save_random_sampled_logits(
    model_name: str,
    dataset_name: str,
    *,
    seed = 42,
    dataset_subset=None,
    dataset_split="train",
    system_prompt=None,
    num_samples=None,
    start_idx=0,
    max_new_tokens=256,
    batch_size=1,
    draws=50,  # R – controls expected ~K ≃ 12 unique tokens
    tau=1.0,
    output_dir=None,
    debug=False,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger = setup()
    logger.info(f"[RSKD] Generating sampled logits from {model_name}")

    # ------------------------------------------------------------------ data
    ds = load_dataset(dataset_name, dataset_subset, split=dataset_split)
    if dataset_name == "tatsu-lab/alpaca":
        ds = ds.filter(lambda x: x["input"] == "")  # alpaca quirk
    
    end_idx = min(len(ds), start_idx + (num_samples or len(ds)))
    actual_num_samples = end_idx - start_idx

    model, tok = load_model(model_name)
    tok.pad_token = tok.pad_token or tok.eos_token
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")

    # ------------------------------------------------------------------ out
    out_dir = Path(
        output_dir
        or f"generated_tokens/{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    all_steps = []  # list[list[ list[(id,prob)] ]]

    # ------------------------------------------------------------------ loop
    for b0 in tqdm(
        range(start_idx, end_idx, batch_size), 
        desc=f"Random‑sampling {actual_num_samples} ex from {start_idx} to {end_idx-1}"
    ):
        b1 = min(b0 + batch_size, end_idx)  # b1 is the batch end index
        prompts = []
        end_marker = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        for ex in ds.select(range(b0, b1)):
            if dataset_name == "tatsu-lab/alpaca":
                prompt = (system_prompt or "") + ex["instruction"] + end_marker
            elif dataset_name == "roskoN/dailydialog":
                dialog = format_dialog(ex["utterances"])
                prompt = (system_prompt or "") + dialog + end_marker
            elif dataset_name == "qiaojin/PubMedQA":
                final_decision = ex["final_decision"]
                formatted_final_decision = (
                    final_decision[0].upper() + final_decision[1:]
                )
                user_prompt = f"Question: {ex['question']}\n\nContext: {formatted_final_decision} - {ex['long_answer']}"
                prompt = (system_prompt or "") + user_prompt + end_marker
            elif dataset_name == "openai/gsm8k":
                prompt = (system_prompt or "") + ex["question"] + end_marker
            prompts.append(prompt)

        inp = tok(prompts, padding=True, return_tensors="pt").to(model.device)

        with amp.autocast("cuda"):
            gen = model.generate(**inp,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False, num_beams=1,
                                 return_dict_in_generate=True, output_scores=True)
            
        
        
        for seq_i in range(len(prompts)):
            step_pairs = []  # this sequence
            for t, score in enumerate(
                gen.scores
            ):  # t = token position, score = logits (batch_size, vocab_size) ie. all logits for this token position
                if score[seq_i].argmax().item() == eot_id:
                    break
                ids, probs = random_sample_distribution(
                    score[seq_i], draws=draws, tau=tau
                )
                step_pairs.append(
                    [(int(i), float(p)) for i, p in zip(ids, probs)]
                )  # ids = unique token ids, probs = unbiased teacher probs (sum = 1)

            if debug and b0 == start_idx:
                print(f"\n############# DEBUGGING SEQ {seq_i} #############")
                for token_prob_pairs in step_pairs:
                    token_prob_pairs.sort(key=lambda x: x[1], reverse=True)
                    print(
                        f"{tok.decode(token_prob_pairs[0][0], skip_special_tokens=False)}",
                        end="",
                    )
                print()
                print("-" * 100)
                token_num = 0
                for token_prob_pairs in step_pairs:
                    token_num += len(token_prob_pairs)
                if len(step_pairs) > 0:
                    print(f"Average number of tokens: {token_num / len(step_pairs)}")
                else:
                    print("Sequence was empty.")
                print(f"###########################################\n")
            all_steps.append(step_pairs)

    # ------------------------------------------------------------------ save
    save_path = out_dir / f"teacher_random_logits_{start_idx}-{end_idx-1}_R{draws}_tau{tau}.pt"
    torch.save(all_steps, save_path)
    logger.info(f"[RSKD] wrote {len(all_steps)} sequences ({start_idx}-{end_idx-1}) → {save_path}")
    return all_steps


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random sampled logits from teacher model")
    parser.add_argument("--start_idx", type=int, help="Starting index for dataset processing, overrides config")
    parser.add_argument("--num_samples", type=int, help="Number of samples to process, overrides config")
    args = parser.parse_args()
    
    general_config = load_config("random_sampling_config.yaml")
    config = general_config["output_generation"]
    dataset_name = general_config["dataset"]['name']
    num_samples = general_config["dataset"].get('num_samples', None)

    dataset_config = load_config("datasets.yaml")[dataset_name]

    if args.start_idx is not None:
        config["start_idx"] = args.start_idx
    if args.num_samples is not None:
        num_samples = args.num_samples

    generate_and_save_random_sampled_logits(
        model_name=config["model"],
        dataset_name=dataset_name,
        seed=42,
        dataset_subset=dataset_config.get("subset", None),
        dataset_split=dataset_config["split"],
        system_prompt=dataset_config.get("system_prompt", None),
        num_samples=num_samples,
        start_idx=config.get("start_idx", 0),
        max_new_tokens=config["max_new_tokens"],
        batch_size=config["batch_size"],
        draws=config["draws"],
        tau=config["tau"],
        output_dir=Path(config["logits_dir"]),
        debug=True,
    )
