import torch
import logging
from pathlib import Path
from tqdm import tqdm
from distillation_utils import load_config
from batch_comparison import setup
from transformers import AutoTokenizer

def trim_eot_tokens(tokens, eot_token_id):
    """Trim trailing EOT tokens from a sequence, keeping exactly one at the end."""
    # Find the last non-EOT token
    non_eot_indices = (tokens != eot_token_id).nonzero()
    if len(non_eot_indices) == 0:
        return tokens[:1]  # Return just the first token if all are EOT
    last_non_eot = non_eot_indices[-1].item()
    print(f"Last non-EOT token index: {last_non_eot}")
    print(f"3 tokens before last non-EOT token: {tokens[last_non_eot-3:last_non_eot+1]}")
    # Keep one EOT token after the last non-EOT token
    return tokens[:last_non_eot + 2]  # +2 to include the last non-EOT and one EOT


def process_tokens_file(input_path, output_path, eot_token_id):
    """Process the tokens file and save trimmed version."""
    logger = setup()
    logger.info(f"Loading tokens from {input_path}")
    
    # Load the tokens
    tokens = torch.load(input_path)
    logger.info(f"Loaded {len(tokens)} sequences")
    
    # Trim each sequence
    trimmed_tokens = []
    original_lengths = []
    trimmed_lengths = []
    
    for seq in tqdm(tokens, desc="Trimming sequences"):
        original_lengths.append(len(seq))
        trimmed_seq = trim_eot_tokens(seq, eot_token_id)
        trimmed_lengths.append(len(trimmed_seq))
        trimmed_tokens.append(trimmed_seq)
    
    # Calculate statistics
    avg_original = sum(original_lengths) / len(original_lengths)
    avg_trimmed = sum(trimmed_lengths) / len(trimmed_lengths)
    total_saved = sum(original_lengths) - sum(trimmed_lengths)
    
    logger.info(f"Average sequence length before trimming: {avg_original:.2f}")
    logger.info(f"Average sequence length after trimming: {avg_trimmed:.2f}")
    logger.info(f"Total tokens saved: {total_saved}")
    
    # Save trimmed tokens
    torch.save(trimmed_tokens, output_path)
    logger.info(f"Saved trimmed tokens to {output_path}")


if __name__ == "__main__":
    config = load_config()
    tokens_path = Path(config["dataset"]["teacher_data"]["tokens_path"])
    output_path = tokens_path.parent / f"{tokens_path.stem}_trimmed.pt"
    # Get EOT token ID from config or use default
    process_tokens_file(tokens_path, output_path, 128001) 