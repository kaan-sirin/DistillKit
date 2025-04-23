from datetime import datetime
import os
from pathlib import Path
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch import amp
from batch_comparison import setup, load_model
from distillation_utils import load_config
import numpy as np

def generate_and_save_tokens(
    model_name,
    dataset_name,
    dataset_subset=None,
    dataset_split="train",
    system_prompt=None,
    num_samples=None,
    max_new_tokens=512,
    batch_size=1,
    output_dir=None,
    debug=False,
):
    logger = setup()
    logger.info(f"Generating outputs from {model_name} on dataset {dataset_name}")

    # Load dataset
    try:
        dataset = (
            load_dataset(dataset_name, dataset_subset, split=dataset_split)
            if dataset_subset
            else load_dataset(dataset_name, split=dataset_split)
        )
        logger.info(f"Loaded {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # WARNING: This is only for the Alpaca dataset - remove for other datasets
    logger.info(f"Filtering out examples where input column isn't empty")
    dataset = dataset.filter(lambda x: x["input"] == "")
    logger.info(f"Filtered dataset to {len(dataset)} examples")

    # Load model
    try:
        model, tokenizer = load_model(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Create output directory
    if output_dir is None:
        output_dir = f"distillation_data/{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)

    # Create a single tensor list to store all tokens
    all_generated_tokens = []
    num_samples = num_samples if num_samples < len(dataset) else len(dataset)

    for batch_start in tqdm(
        range(0, num_samples, batch_size),
        desc=f"Processing samples ({num_samples} samples)",
    ):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_examples = [dataset[i] for i in range(batch_start, batch_end)]
        batch_prompts = []
        end_prompt = "<|eot_id|>"
        for example in batch_examples:
            # Format with system prompt if provided
            if system_prompt:
                # Combine system prompt and user question
                full_prompt = f"{system_prompt}{example['instruction']}{end_prompt}"
            else:
                full_prompt = example["instruction"]

            batch_prompts.append(full_prompt)

        # Tokenize inputs
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(
            model.device
        )

        # Generate with token tracking
        with amp.autocast("cuda"):
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                # return_dict_in_generate=True,
                # output_scores=True,
            )

        torch.cuda.synchronize()

        # Extract only the generated tokens (not the prompt tokens)
        for i, (example, sequence) in enumerate(zip(batch_examples, outputs)):
            prompt_length = len(batch_inputs["input_ids"][i])
            generated_tokens = sequence[prompt_length:]  # Keep as tensor

            # Debug information
            if debug:
                print(f"\n===== GENERATED OUTPUT {batch_start + i} =====")
                print(
                    f"Prompt length: {prompt_length}, Sequence length: {len(sequence)}"
                )

            # Check if we have any generated tokens
            if len(generated_tokens) > 0:
                decoded_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                print(f"Generated text ({len(generated_tokens)} tokens):")
                print(decoded_text)
            else:
                print("WARNING: No tokens were generated beyond the prompt!")
                # Try decoding the full sequence to see what's there
                if debug:
                    print("Full sequence:")
                    print(tokenizer.decode(sequence, skip_special_tokens=True))

            if debug:
                print("=======================================\n")
            all_generated_tokens.append(generated_tokens)

            # Optional: print a sample of decoded text occasionally for verification
            if (batch_start + i) % 20 == 0:
                sample_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                logger.info(
                    f"Sample generation ({batch_start + i}): {sample_text[:50]}..."
                )

        # Save checkpoint every 100 examples to avoid data loss
        if (batch_end) % 100 == 0 or batch_end == num_samples:
            checkpoint_path = f"{output_dir}/tokens_checkpoint_{batch_end}.pt"
            torch.save(all_generated_tokens, checkpoint_path)
            logger.info(
                f"Saved checkpoint with {len(all_generated_tokens)} examples to {checkpoint_path}"
            )

    # Save final complete tensor file
    final_path = f"{output_dir}/teacher_tokens.pt"
    torch.save(all_generated_tokens, final_path)
    logger.info(
        f"Generation complete! Saved {len(all_generated_tokens)} examples to {final_path}"
    )

    return all_generated_tokens


def generate_and_save_sparse_logits(
    model_name,
    dataset_name,
    dataset_subset=None,
    dataset_split="train",
    system_prompt=None,
    num_samples=None,
    max_new_tokens=256,
    batch_size=1,
    top_k=3,  
    output_dir=None,
    debug=False,
):
    logger = setup()
    logger.info(f"Generating sparse logits from {model_name} on dataset {dataset_name}")

    # Load dataset
    try:
        dataset = (
            load_dataset(dataset_name, dataset_subset, split=dataset_split)
            if dataset_subset
            else load_dataset(dataset_name, split=dataset_split)
        )
        logger.info(f"Loaded {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # WARNING: This is only for the Alpaca dataset - remove for other datasets
    logger.info(f"Filtering out examples where input column isn't empty")
    dataset = dataset.filter(lambda x: x["input"] == "")
    logger.info(f"Filtered dataset to {len(dataset)} examples")

    # Load model
    try:
        model, tokenizer = load_model(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Get the EOT token ID for detecting the end of turn
        eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        logger.info(f"Using EOT token ID: {eot_token_id}")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Create output directory
    if output_dir is None:
        output_dir = f"distillation_data/{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate top-k based on vocabulary size
    vocab_size = len(tokenizer)
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(
        f"Saving top {top_k} logits out of {vocab_size} total"
    )

    # Create a list to store all sparse logits
    all_sparse_logits = []
    num_samples = min(num_samples, len(dataset)) if num_samples else len(dataset)

    for batch_start in tqdm(
        range(0, num_samples, batch_size),
        desc=f"Processing samples ({num_samples} samples)",
    ):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_examples = [dataset[i] for i in range(batch_start, batch_end)]
        batch_prompts = []
        end_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        for example in batch_examples:
            # Format with system prompt if provided
            if system_prompt:
                # Combine system prompt and user question
                full_prompt = f"{system_prompt}{example['instruction']}{end_prompt}"
            else:
                full_prompt = example["instruction"]

            batch_prompts.append(full_prompt)

        # Tokenize inputs
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(
            model.device
        )

        # Generate with logit tracking
        with amp.autocast("cuda"):
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )

        torch.cuda.synchronize()
        
        output_sequences = {f"sequence_{i}": {"decoded_sequence": "", "eot_found_index": -1, "top_logits": []} for i in range(batch_size)}
        for step, score_tensor in enumerate(outputs.scores):
            # score_tensor has shape (batch_size, vocab_size)
            if step % 100 == 0:
                print(f"STEP: {step}, SHAPE: {score_tensor.shape}")
            for i in range(score_tensor.size(0)):  # i goes from 0..(batch_size-1) ie. for each sequence in the batch
                if output_sequences[f"sequence_{i}"]["eot_found_index"] != -1:
                    continue
                score_vector = score_tensor[i]
                top_token = score_vector.argmax().item()
                output_sequences[f"sequence_{i}"]["decoded_sequence"] += tokenizer.decode([top_token])
                values, indices = torch.topk(score_vector, top_k)
                top_k_list = [(indices[j].item(), values[j].item()) for j in range(top_k)]
                output_sequences[f"sequence_{i}"]["top_logits"].append(top_k_list)
                
                if top_token == eot_token_id:
                    print(f"Found EOT token at generation step {step} for sequence {i}")
                    output_sequences[f"sequence_{i}"]["eot_found_index"] = step
        
        if debug:
            for i in range(batch_size):
                seq_info = output_sequences[f"sequence_{i}"]
                print(f"Sequence {i} decoded: {seq_info['decoded_sequence']}")
                print(f"EOT found step: {seq_info['eot_found_index']}")
                print(f"Number of steps we collected: {len(seq_info['top_logits'])}")
                # Inspect the top logits for this sequence (only the first 3)
                # For each element in the top_logits list, print the token, it's decoded value and the logit
                # for j in range(len(seq_info['top_logits'])):
                #     top_logits = seq_info['top_logits'][j]
                #     for token_id, logit in top_logits:
                #         token_decoded = tokenizer.decode([token_id])
                #         print(f"Token {j}: {token_id} ({token_decoded}), Logit: {logit}")
                #     print("-----")
                # print("\n\n")
                
        # Now we can save the top logits for each sequence
        for i in range(batch_size):
            all_sparse_logits.append(output_sequences[f"sequence_{i}"]["top_logits"])   

    # Save all sparse logits
    final_path = f"{output_dir}/teacher_sparse_logits_{dataset_name.split('/')[-1]}_{num_samples}_samples_top_{top_k}.pt"
    torch.save(all_sparse_logits, final_path)
    logger.info(
        f"Generation complete! Saved {len(all_sparse_logits)} examples to {final_path}"
    )
    
    
    return all_sparse_logits


if __name__ == "__main__":
    # Example usage
    config = load_config()

    # Use the new sparse logits function instead of the token function
    generate_and_save_sparse_logits(
        model_name=config["models"]["teacher"],
        dataset_name=config["dataset"]["name"],
        dataset_split=config["dataset"]["split"],
        system_prompt=config["dataset"]["teacher_data"].get("system_prompt", None),
        num_samples=config["dataset"]["num_samples"],
        max_new_tokens=config["dataset"]["teacher_data"]["max_new_tokens"],
        batch_size=8,  # Using batch size 1 for clearer logit tracking
        top_k=config["dataset"]["teacher_data"]["top_k"],  # Save top 1% of logits # TODO: Currently overriden by top_k=3
        output_dir=Path(config["dataset"]["teacher_data"]["tokens_path"]).parent,
        debug=True,  # Enable debug output initially to verify everything works
    )
