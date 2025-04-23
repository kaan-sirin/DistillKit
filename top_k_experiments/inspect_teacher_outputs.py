import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from batch_comparison import load_model

def inspect_generated_tokens(tokens_path, model_name, dataset_name=None, 
                             dataset_subset=None, dataset_split="train", 
                             num_samples=5):
    
    print(f"\n{'='*50}")
    print(f"INSPECTING TOKENS: {os.path.basename(tokens_path)}")
    print(f"{'='*50}\n")
    
    # Load token tensors
    tokens = torch.load(tokens_path)
    print(f"✓ Loaded {len(tokens)} examples from {tokens_path}")
    

    # Basic statistics
    num_examples = len(tokens)
    token_lengths = [len(seq) for seq in tokens]
    avg_length = sum(token_lengths) / num_examples if num_examples > 0 else 0
    max_length = max(token_lengths) if token_lengths else 0
    min_length = min(token_lengths) if token_lengths else 0
    
    print(f"\n📊 STATISTICS:")
    print(f"  • Number of examples: {num_examples}")
    print(f"  • Average token length: {avg_length:.1f}")
    print(f"  • Max token length: {max_length}")
    print(f"  • Min token length: {min_length}")
    
    # Sample tensor details
    print("\n🔢 TENSOR DETAILS:")
    for i in range(min(3, num_examples)):
        tensor = tokens[i]
        print(f"  • Example {i}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # Load tokenizer and dataset
    _, tokenizer = load_model(model_name)
    
    original_data = None
    if dataset_name:
        original_data = (
            load_dataset(dataset_name, dataset_subset, split=dataset_split)
            if dataset_subset
            else load_dataset(dataset_name, split=dataset_split)
        )
    
    # Decode and display samples
    print(f"\n📝 SAMPLE OUTPUTS ({min(num_samples, num_examples)}):")
    
    for i in range(min(num_samples, num_examples)):
        tensor = tokens[i]
        decoded = tokenizer.decode(tensor, skip_special_tokens=False)
        
        prompt = "N/A"
        if original_data and i < len(original_data):
            prompt = original_data[i].get("prompt", "N/A")
            
        print(f"\n----- EXAMPLE {i} -----")
        if prompt != "N/A":
            print(f"PROMPT: {prompt[:100]}..." if len(prompt) > 100 else f"PROMPT: {prompt}")
        print(f"TOKENS: {len(tensor)} tokens")
        print(f"OUTPUT: {decoded}")
    
    # Generate histogram
    plt.figure(figsize=(10, 5))
    plt.hist(token_lengths, bins=20)
    plt.title("Distribution of Generated Token Lengths")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.annotate(f"Mean: {avg_length:.1f}", xy=(0.75, 0.85), xycoords='axes fraction')
    
    # Save and display
    save_path = os.path.join(os.path.dirname(tokens_path), "token_distribution.png")
    plt.savefig(save_path)
    print(f"\n📊 Saved distribution plot to {save_path}")

def inspect_sparse_logits(logits_path, model_name, dataset_name=None, 
                         dataset_subset=None, dataset_split="train", 
                         num_samples=5):
    """
    Inspect sparse logits saved for knowledge distillation.
    Each example is a list of (indices, values) tuples for each token position.
    """
    print(f"\n{'='*50}")
    print(f"INSPECTING SPARSE LOGITS: {os.path.basename(logits_path)}")
    print(f"{'='*50}\n")
    
    # Load sparse logits
    sparse_logits = torch.load(logits_path)
    print(f"✓ Loaded {len(sparse_logits)} examples from {logits_path}")
    
    # Basic statistics
    num_examples = len(sparse_logits)
    sequence_lengths = [len(seq) for seq in sparse_logits]
    avg_length = sum(sequence_lengths) / num_examples if num_examples > 0 else 0
    max_length = max(sequence_lengths) if sequence_lengths else 0
    min_length = min(sequence_lengths) if sequence_lengths else 0
    
    # Count how many tokens we saved per position (should be consistent)
    if num_examples > 0 and sequence_lengths[0] > 0:
        sample_tokens_saved = sparse_logits[0][0][0].shape[0]
    else:
        sample_tokens_saved = 0
    
    print(f"\n📊 STATISTICS:")
    print(f"  • Number of examples: {num_examples}")
    print(f"  • Average sequence length: {avg_length:.1f}")
    print(f"  • Max sequence length: {max_length}")
    print(f"  • Min sequence length: {min_length}")
    print(f"  • Tokens saved per position: {sample_tokens_saved}")
    
    # Sample tensor details
    print("\n🔢 TENSOR DETAILS:")
    for i in range(min(3, num_examples)):
        sequence = sparse_logits[i]
        if len(sequence) > 0:
            indices, values = sequence[0]  # First position
            print(f"  • Example {i}: positions={len(sequence)}, tokens_per_pos={indices.shape[0]}")
            print(f"    • Indices shape: {indices.shape}, Values shape: {values.shape}")
        else:
            print(f"  • Example {i}: Empty sequence")
    
    # Load tokenizer and dataset
    _, tokenizer = load_model(model_name)
    
    # Get important token IDs for inspection
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    print(f"Using EOT token ID: {eot_token_id}")
    
    original_data = None
    if dataset_name:
        try:
            original_data = (
                load_dataset(dataset_name, dataset_subset, split=dataset_split)
                if dataset_subset
                else load_dataset(dataset_name, split=dataset_split)
            )
            # Apply the same filtering logic as in generate_teacher_outputs.py
            if "input" in original_data.column_names:
                original_data = original_data.filter(lambda x: x["input"] == "")
            print(f"✓ Loaded {len(original_data)} examples from {dataset_name}")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
    
    # Decode and display samples
    print(f"\n📝 SAMPLE OUTPUTS ({min(num_samples, num_examples)}):")
    
    for i in range(min(num_samples, num_examples)):
        sequence = sparse_logits[i]
        
        # Get the top token at each position (index 0 of each position's values)
        if len(sequence) > 0:
            top_tokens = []
            for position in sequence:
                indices, values = position
                top_token_idx = indices[0].item()  # Get the highest probability token
                top_tokens.append(top_token_idx)
            
            # Convert to tensor and decode
            top_tokens_tensor = torch.tensor(top_tokens)
            decoded = tokenizer.decode(top_tokens_tensor, skip_special_tokens=False)
            
            # Get some statistics about the logits
            all_top_values = [pos[1][0].item() for pos in sequence]  # Top logit value at each position
            avg_confidence = torch.mean(torch.tensor(all_top_values))
            
            # Check if the sequence ends with EOT
            has_eot = eot_token_id in top_tokens
            last_token_id = top_tokens[-1] if top_tokens else None
            ends_with_eot = last_token_id == eot_token_id
            
            # Get EOT positions
            eot_positions = [i for i, token in enumerate(top_tokens) if token == eot_token_id]
            
            ending_description = "Unknown"
            if has_eot:
                if ends_with_eot:
                    ending_description = "Ends with EOT (correct)"
                else:
                    ending_description = f"Has EOT at position(s) {eot_positions} but not at the end"
            else:
                ending_description = "Missing EOT token"
            
            # Get prompt if available
            prompt = "N/A"
            if original_data and i < len(original_data):
                prompt = original_data[i].get("instruction", original_data[i].get("prompt", "N/A"))
                
            # Print sample information
            print(f"\n----- EXAMPLE {i} -----")
            if prompt != "N/A":
                print(f"PROMPT: {prompt[:100]}..." if len(prompt) > 100 else f"PROMPT: {prompt}")
            print(f"SEQUENCE LENGTH: {len(sequence)} positions")
            print(f"AVG TOP LOGIT VALUE: {avg_confidence:.4f}")
            print(f"ENDING ANALYSIS: {ending_description}")
            
            # Show the token IDs for better debugging
            if len(top_tokens) > 5:
                print(f"FIRST 5 TOKEN IDs: {top_tokens[:5]}")
                print(f"LAST 5 TOKEN IDs: {top_tokens[-5:]}")
            else:
                print(f"ALL TOKEN IDs: {top_tokens}")
                
            print(f"OUTPUT (from top tokens): {decoded}")
            
            # Show distribution of top few tokens for 3 sample positions (first, middle, and last)
            if len(sequence) >= 3:
                positions_to_check = [0, len(sequence) // 2, -1]  # First, middle, last
                position_names = ["First", "Middle", "Last"] 
                
                for pos_idx, pos in enumerate(positions_to_check):
                    indices, values = sequence[pos]
                    top_5_tokens = [tokenizer.decode([idx.item()]) for idx in indices[:5]]
                    top_5_values = [val.item() for val in values[:5]]
                    
                    print(f"\n{position_names[pos_idx]} position tokens:")
                    for t, v in zip(top_5_tokens, top_5_values):
                        print(f"  • Token: '{t}', Logit: {v:.4f}")
            elif len(sequence) > 0:
                # If sequence is very short, just show the first position
                indices, values = sequence[0] 
                top_5_tokens = [tokenizer.decode([idx.item()]) for idx in indices[:5]]
                top_5_values = [val.item() for val in values[:5]]
                
                print("\nFirst position tokens:")
                for t, v in zip(top_5_tokens, top_5_values):
                    print(f"  • Token: '{t}', Logit: {v:.4f}")
        else:
            print(f"\n----- EXAMPLE {i} -----")
            print("Empty sequence - no tokens generated")
    
    # Generate histogram of sequence lengths
    plt.figure(figsize=(10, 5))
    plt.hist(sequence_lengths, bins=20)
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Number of Token Positions")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.annotate(f"Mean: {avg_length:.1f}", xy=(0.75, 0.85), xycoords='axes fraction')
    
    # Save and display
    save_path = os.path.join(os.path.dirname(logits_path), "sequence_length_distribution.png")
    plt.savefig(save_path)
    print(f"\n📊 Saved sequence length distribution plot to {save_path}")
    
    # Generate logit value distribution (for top tokens)
    if num_examples > 0 and any(sequence_lengths):
        # Collect top logit values from all positions in all sequences
        all_top_logits = []
        for seq in sparse_logits[:min(100, num_examples)]:  # Limit to first 100 examples
            for pos in seq:
                if len(pos) > 0:
                    indices, values = pos
                    all_top_logits.append(values[0].item())  # Top logit value
        
        plt.figure(figsize=(10, 5))
        plt.hist(all_top_logits, bins=50)
        plt.title("Distribution of Top Logit Values")
        plt.xlabel("Logit Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        # Save and display
        save_path = os.path.join(os.path.dirname(logits_path), "logit_value_distribution.png")
        plt.savefig(save_path)
        print(f"📊 Saved logit value distribution plot to {save_path}")
        
        # Generate confidence over position plot (for first few examples)
        plt.figure(figsize=(12, 6))
        for i in range(min(5, num_examples)):
            if i < len(sparse_logits) and len(sparse_logits[i]) > 0:
                # Get logit values for each position
                logit_values = [pos[1][0].item() for pos in sparse_logits[i]]
                plt.plot(logit_values, label=f"Example {i}")
        
        plt.title("Confidence (Top Logit Value) by Position")
        plt.xlabel("Token Position")
        plt.ylabel("Top Logit Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save and display
        save_path = os.path.join(os.path.dirname(logits_path), "confidence_by_position.png")
        plt.savefig(save_path)
        print(f"📊 Saved confidence by position plot to {save_path}")

if __name__ == "__main__":
    # Configuration - change these values as needed
    
    # For token-based output inspection
    # inspect_generated_tokens(
    #     tokens_path="generated_tokens/sassy/teacher_tokens_sassy_batched.pt",
    #     model_name="meta-llama/Llama-3.2-3B-Instruct",
    #     dataset_name="kaans/rule_qa",
    #     dataset_split="train",
    #     num_samples=5
    # )
    
    # For sparse logits inspection
    inspect_sparse_logits(
        logits_path="generated_tokens/sassy/teacher_sparse_logits.pt",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        dataset_name="tatsu-lab/alpaca",
        dataset_split="train",
        num_samples=5
    ) 