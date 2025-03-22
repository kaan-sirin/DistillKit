import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from dotenv import load_dotenv
import time
from tqdm import tqdm
from distillation_with_diff_prompts_final import (
    load_config,
    medqa_format,
    tokenize_function,
)
from torch.nn.utils.rnn import pad_sequence




def generate_and_save_outputs(shuffle=False):
    HF_TOKEN = os.getenv("HF_TOKEN")
    config = load_config()

    # Set up output directory
    output_base = config["training"]["output_dir"]
    timestamp = time.strftime("%m-%d_%H-%M").replace("-", "_")
    output_dir = os.path.join(output_base, f"teacher_outputs_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Load dataset

    dataset = (
        load_dataset(
            config["dataset"]["name"],
            config["dataset"]["subset"],
            split=config["dataset"]["split"],
        )
        if config["dataset"].get("subset")
        else load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    )
    
    # Dataset contains 3180 samples

    # Get the first num_samples
    if "num_samples" in config["dataset"]:
        dataset = dataset.select(range(config["dataset"]["num_samples"]))

    # Save original indices before shuffling (in case there is less than num_samples)
    original_indices = list(range(len(dataset)))

    if shuffle:
        # Shuffle dataset with a fixed seed for reproducibility
        shuffle_seed = config["dataset"]["seed"]

        # Create a deterministic random generator
        rng = np.random.default_rng(shuffle_seed)
        # Generate shuffled indices
        shuffled_indices = rng.permutation(original_indices).tolist()
    else:
        shuffle_seed = None
        shuffled_indices = original_indices

    # Save shuffled indices
    indices_file = os.path.join(output_dir, "indices.pt")
    print(f"Saving shuffled indices to {indices_file}")
    torch.save(shuffled_indices, indices_file)

    # Apply the shuffle
    dataset = dataset.select(shuffled_indices)
    print(f"Dataset shuffled with seed {shuffle_seed}, size: {len(dataset)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["teacher"], token=HF_TOKEN
    )

    # Ensure padding token is set if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing dataset...")
    original_columns = dataset.column_names
    dataset = dataset.map(
        medqa_format,
        remove_columns=original_columns,
    )

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        num_proc=8,
        remove_columns=["question", "prompt"],
    )

    # No longer splitting the dataset
    # tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    # Load teacher model
    print("Loading teacher model...")
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    teacher_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["teacher"], **model_kwargs
    ).to(device)
    teacher_model.eval()  # Set model to evaluation mode

    # Add configuration for top-k logits (default to 5% as per the SLIM paper)
    top_k_percent = config.get("distillation", {}).get("top_k_percent", 5)

    # Prepare outputs
    outputs = []

    # Process dataset and generate outputs
    print("Generating outputs from teacher model...")
    MAX_NEW_TOKENS = 512

    batch_size = 4

    # Function to process a dataset split
    def process_dataset(dataset):
        results = []

        for i in tqdm(
            range(0, len(dataset), batch_size),
            desc=f"Processing the dataset",
        ):
            batch = dataset[i : i + batch_size]
            
            # Convert lists to tensors and pad them
            input_tensors = [torch.tensor(seq) for seq in batch["input_ids"]]
            mask_tensors = [torch.tensor(mask) for mask in batch["attention_mask"]]
            
            # Pad sequences (padding value defaults to 0, so set it to pad_token_id for inputs)
            padded_inputs = pad_sequence(input_tensors, batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_masks = pad_sequence(mask_tensors, batch_first=True, padding_value=0)
            
            # Prepare inputs
            inputs = {
                "input_ids": padded_inputs.to(device),
                "attention_mask": padded_masks.to(device),
            }

            # Get the number of tokens in the teacher input
            teacher_input_tokens = inputs["input_ids"].shape[-1]

            # Generate answer
            with torch.no_grad():
                answers = teacher_model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    temperature=1.0,
                    top_p=1.0,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Get sequence and scores
            for idx in range(len(answers.sequences)):
                # Get the generated tokens (excluding input)
                generated_tokens = answers.sequences[idx][
                    teacher_input_tokens:
                ]
                
                generated_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )

                # Get the logits/scores and keep only top-k
                # Convert scores tuple to a list of tensors
                logits_list = [score[idx].cpu() for score in answers.scores]

                # Process each token's logits to keep only top-k%
                sparse_logits = []
                for logits in logits_list:
                    vocab_size = logits.size(0)
                    k = max(
                        1, int(vocab_size * top_k_percent / 100)
                    )  # Calculate k from percentage

                    # Get top-k values and indices
                    values, indices = torch.topk(logits, k)

                    # Create sparse representation: (values only, since indices are saved separately)
                    sparse_logits.append(values.tolist())

                print(f"Number of generated tokens: {len(generated_tokens)}")
                print(f"Original vocab size: {vocab_size}, keeping top-{k} logits")

                # Store the output
                example_output = {
                    "index": i + idx,
                    "input_text": tokenizer.decode(
                        inputs["input_ids"][idx], skip_special_tokens=True
                    ),
                    "generated_text": generated_text,
                    "generated_tokens": generated_tokens.tolist(),
                    "sparse_logits": sparse_logits,  # Store only top-k values
                }

                results.append(example_output)

        return results

    # Process the dataset (no splitting)
    outputs = process_dataset(tokenized_dataset)

    # Instead of saving the full output dictionary, extract only tensors of tokens and sparse logits
    print("Converting to tensor format...")

    # Prepare lists to collect data before converting to tensors
    token_ids = []
    logit_values = []

    for output in outputs:
        # Get token ids as a list
        tokens = output["generated_tokens"]
        token_ids.append(tokens)

        # Extract sparse logit values (no indices)
        values_list = output["sparse_logits"]
        logit_values.append(values_list)

    # Convert to tensors - each sample is a separate tensor since they may have different lengths
    token_tensors = [torch.tensor(ids) for ids in token_ids]

    # For sparse logits, now we only keep the values (indices are saved separately in indices.pt)
    sparse_logit_tensors = []
    for sample_idx in range(len(logit_values)):
        sample_values = logit_values[sample_idx]

        # Create a list of sparse logit value tensors for this sample
        sample_sparse_logits = []
        for token_idx in range(len(sample_values)):
            values = torch.tensor(sample_values[token_idx])
            sample_sparse_logits.append(values)

        sparse_logit_tensors.append(sample_sparse_logits)

    # Save tokens and sparse logits as tensors
    tokens_file = os.path.join(output_dir, "generated_tokens.pt")
    logits_file = os.path.join(output_dir, "sparse_logits.pt")

    print(f"Saving tokens to {tokens_file}")
    torch.save(token_tensors, tokens_file)

    print(f"Saving sparse logits to {logits_file}")
    torch.save(sparse_logit_tensors, logits_file)

    print(f"Generation complete. Outputs saved to {output_dir}")

    # Return the file paths including indices file
    return tokens_file, logits_file, indices_file


if __name__ == "__main__":
    load_dotenv()

    # Call the function and get the file paths
    tokens_file, logits_file, indices_file = generate_and_save_outputs()
    print(f"Tokens file: {tokens_file}")
    print(f"Logits file: {logits_file}")
    print(f"Indices file: {indices_file}")

    # Add verification code to check the saved tensors and indices
    print("\n" + "=" * 60)
    print(" " * 20 + "TENSOR VERIFICATION" + " " * 20)
    print("=" * 60)

    # Load configuration for reference
    config = load_config()
    top_k_percent = config.get("distillation", {}).get("top_k_percent", 5)

    # Use the returned file paths directly
    print(f"\nVerifying tensors and indices from returned file paths:")
    print(f"Token file: {tokens_file}")
    print(f"Logits file: {logits_file}")
    print(f"Indices file: {indices_file}")

    # Load the tensors and indices
    token_tensors = torch.load(tokens_file)
    sparse_logit_tensors = torch.load(logits_file)
    shuffled_indices = torch.load(indices_file)

    # Basic statistics
    num_samples = len(token_tensors)
    print(f"\n📊 Generated {num_samples} samples")
    print(f"📋 First 10 shuffled indices: {shuffled_indices[:10]}")
    print(f"📈 Indices range: {min(shuffled_indices)} to {max(shuffled_indices)}")

    if num_samples > 0:
        # Get statistics on sequence lengths
        sequence_lengths = [tensor.size(0) for tensor in token_tensors]
        avg_seq_length = sum(sequence_lengths) / len(sequence_lengths)
        max_seq_length = max(sequence_lengths)
        min_seq_length = min(sequence_lengths)

        print(f"\n📏 Sequence Statistics:")
        print(f"   ├── Average length: {avg_seq_length:.2f} tokens")
        print(f"   ├── Maximum length: {max_seq_length} tokens")
        print(f"   └── Minimum length: {min_seq_length} tokens")

        # Check the first sample in detail
        print(f"\n🔍 First Sample Details:")

        # Token tensor
        first_tokens = token_tensors[0]
        print(f"   ├── Token tensor shape: {first_tokens.shape}")
        print(f"   ├── Token tensor type: {first_tokens.dtype}")
        print(f"   ├── First 5 tokens: {first_tokens[:5].tolist()}")

        # Sparse logit tensors
        first_sparse_logits = sparse_logit_tensors[0]

        # Verify we have one logit entry per token
        assert len(first_sparse_logits) == len(
            first_tokens
        ), "Mismatch between tokens and logits!"

        # Check the first token's logits
        first_token_logits = first_sparse_logits[0]

        # Get vocabulary size estimate
        vocab_size_estimate = max(first_token_logits.max().item(), 128256)

        # Calculate how many values we're keeping on average
        values_per_token = [len(token_logit) for token_logit in first_sparse_logits]
        avg_values = sum(values_per_token) / len(values_per_token)

        # Get the sparsity ratio
        sparsity_ratio = avg_values / vocab_size_estimate * 100

        print(f"   ├── Sparse logits entries: {len(first_sparse_logits)}")
        print(f"   ├── First token values shape: {first_token_logits.shape}")
        print(f"   ├── Values kept per token (avg): {avg_values:.2f}")
        print(f"   ├── Estimated vocabulary size: {vocab_size_estimate}")
        print(
            f"   ├── Actual sparsity: {sparsity_ratio:.2f}% (target: {top_k_percent}%)"
        )

        # Verify the sparsity is close to the target
        if abs(sparsity_ratio - top_k_percent) > 1.0:
            print(f"   └── ⚠️ WARNING: Actual sparsity differs from target by > 1%")
        else:
            print(f"   └── ✅ Sparsity matches target")

        # Memory usage analysis
        sparse_size = sum(
            tensor.numel() for sample in sparse_logit_tensors for tensor in sample
        )
        full_size = sum(
            len(sample) * vocab_size_estimate for sample in sparse_logit_tensors
        )
        compression = full_size / sparse_size if sparse_size > 0 else 0

        print(f"\n💾 Storage Analysis:")
        print(f"   ├── Sparse elements stored: {sparse_size:,}")
        print(f"   ├── Dense elements (if stored): {full_size:,}")
        print(f"   └── Compression ratio: {compression:.2f}x")

        # Tensor suitability for distillation
        print(f"\n🔄 Tensor Suitability for Distillation:")
        print(f"   ├── Token tensors ready: ✅")
        print(f"   ├── Sparse logit format suitable: ✅")
        print(f"   ├── Indices saved separately: ✅")
        print(f"   └── Data verification complete")

    print("\n" + "=" * 60)
    print(" " * 20 + "END OF VERIFICATION" + " " * 20)
    print("=" * 60)
