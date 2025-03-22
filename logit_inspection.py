import torch
import os
from transformers import AutoTokenizer
from dotenv import load_dotenv
import numpy as np
from pprint import pprint

# Load environment variables (for HF_TOKEN if needed)
load_dotenv()

# Define paths
output_dir = "results/teacher_outputs_03_22_14_09"
indices_file = os.path.join(output_dir, "indices.pt")
logits_file = os.path.join(output_dir, "sparse_logits.pt")
tokens_file = os.path.join(output_dir, "generated_tokens.pt")

# Load the data
print(f"Loading data from {output_dir}...")
indices = torch.load(indices_file)
sparse_logits = torch.load(logits_file)
token_tensors = torch.load(tokens_file)

# Print basic information
print("\n===== Basic Information =====")
print(f"Number of samples: {len(token_tensors)}")
print(f"First 10 indices: {indices[:10]}")
print(f"Indices shape: {len(indices)}")


# Analyze token tensors
print("\n===== Token Tensors Analysis =====")
token_shapes = [tensor.shape for tensor in token_tensors]
print(f"Number of token tensors: {len(token_tensors)}")
print(f"First 5 token tensor shapes: {token_shapes[:5]}")

# Get statistics on sequence lengths
sequence_lengths = [tensor.size(0) for tensor in token_tensors]
avg_seq_length = sum(sequence_lengths) / len(sequence_lengths)
max_seq_length = max(sequence_lengths)
min_seq_length = min(sequence_lengths)

print(f"Average sequence length: {avg_seq_length:.2f} tokens")
print(f"Maximum sequence length: {max_seq_length} tokens")
print(f"Minimum sequence length: {min_seq_length} tokens")

# Analyze logit tensors
print("\n===== Sparse Logit Tensors Analysis =====")
print(f"Number of sparse logit tensor lists: {len(sparse_logits)}")

# Check the first sample's logits
first_sample_logits = sparse_logits[0]
print(f"First sample logits count: {len(first_sample_logits)}")
print(f"First sample logits shapes: {[logit.shape for logit in first_sample_logits[:5]]}")

# Get the number of values kept per token
values_per_token = [[logit.shape[0] for logit in sample_logits] for sample_logits in sparse_logits]
avg_values = np.mean([np.mean(sample_values) for sample_values in values_per_token])
print(f"Average values kept per token: {avg_values:.2f}")

# Load the tokenizer to decode tokens
# Try to infer the model name from the data directories or files
try:
    # Import the config loading function from the generation script
    from distillation_with_diff_prompts_final import load_config
    config = load_config()
    model_name = config["models"]["teacher"]
    print(f"\nUsing teacher model: {model_name}")
except (ImportError, KeyError):
    print("\nCould not load config. Using default model.")
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Default model

# Get HF token if it exists
HF_TOKEN = os.getenv("HF_TOKEN")

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        
    # Decode the first few generated token sequences
    print("\n===== Decoded Tokens =====")
    for i in range(min(5, len(token_tensors))):
        decoded = tokenizer.decode(token_tensors[i], skip_special_tokens=True)
        print(f"\nSample {i}, Original index: {indices[i]}")
        print(f"Tokens: {token_tensors[i][:10].tolist()}... (total: {len(token_tensors[i])})")
        print(f"Decoded text (first 100 chars): {decoded[:100]}...")
except Exception as e:
    print(f"Error loading tokenizer or decoding tokens: {e}")
    print("Could not decode tokens")

# Verify alignment between tokens and logits
print("\n===== Alignment Verification =====")
alignment_check = all(len(token_tensors[i]) == len(sparse_logits[i]) for i in range(len(token_tensors)))
print(f"Tokens and logits are aligned: {alignment_check}")

if not alignment_check:
    misaligned = [(i, len(token_tensors[i]), len(sparse_logits[i])) 
                  for i in range(len(token_tensors)) 
                  if len(token_tensors[i]) != len(sparse_logits[i])]
    print(f"Misaligned samples (idx, tokens_len, logits_len): {misaligned[:5]}")

# Print a detailed view of the first sample
print("\n===== Detailed First Sample =====")
if len(token_tensors) > 0 and len(sparse_logits) > 0:
    first_tokens = token_tensors[0]
    first_logits = sparse_logits[0]
    
    print(f"First sample tokens shape: {first_tokens.shape}")
    print(f"First 5 tokens: {first_tokens[:5].tolist()}")
    
    for i in range(min(5, len(first_logits))):
        print(f"Token {i} logits shape: {first_logits[i].shape}")
        print(f"Token {i} top 5 logit values: {first_logits[i][:5].tolist()}")
        
        if 'tokenizer' in locals():
            try:
                token_id = first_tokens[i].item()
                token_text = tokenizer.decode([token_id])
                print(f"Token {i} text: '{token_text}'")
            except:
                print(f"Could not decode token {i}")
