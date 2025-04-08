import os
import torch
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

if __name__ == "__main__":
    # Configuration - change these values as needed
    inspect_generated_tokens(
        tokens_path="generated_tokens/sassy/teacher_tokens_sassy_batched.pt",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        dataset_name="kaans/rule_qa",
        dataset_split="train",
        num_samples=5
    ) 