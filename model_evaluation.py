import argparse
import os
import random
import json
from datetime import datetime
import re
import torch
from tqdm import tqdm
from datasets import load_dataset
from batch_comparison import format_examples, setup, load_model
from torch import amp

from distillation_utils import load_config


def generate_responses(
    model, tokenizer, prompts, max_new_tokens=512, skip_special_tokens=True
):
    """Generate responses for a list of prompts"""
    batch_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(
        model.device
    )

    with amp.autocast("cuda"):
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            top_p=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

    torch.cuda.synchronize()

    # Decode all responses
    responses = []
    for i, output in enumerate(outputs):
        prompt_tokens = len(batch_inputs["input_ids"][i])
        response = tokenizer.decode(
            output[prompt_tokens:], skip_special_tokens=skip_special_tokens
        )
        responses.append(response)

    return responses


def evaluate_model(
    model_path,
    dataset_name,
    dataset_subset=None,
    dataset_split="test",
    num_samples=8,
    random_sampling=True,
    start_index=0,
    max_new_tokens=512,
    batch_size=4,
    output_filename=None,
    enable_logging: bool = False,
):
    """Evaluate a model on a dataset"""
    logger = setup() if enable_logging else None
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    if logger:
        logger.info(f"Evaluating model {model_path} on dataset {dataset_name}")

    # Load dataset
    try:
        dataset = (
            load_dataset(dataset_name, dataset_subset, split=dataset_split)
            if dataset_subset
            else load_dataset(dataset_name, split=dataset_split)
        )
        if logger:
            logger.info(f"Loaded {len(dataset)} examples")
    except Exception as e:
        if logger:
            logger.error(f"Error loading dataset: {e}")
        raise
    
    if dataset_name == "Vezora/Tested-143k-Python-Alpaca": # filter out examples with input
        dataset = dataset.filter(lambda x: x["input"] == "")

    try:
        model, tokenizer = load_model(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        if logger:
            logger.error(f"Error loading model: {e}")
        raise

    # Select examples
    examples = [dataset[i] for i in range(start_index, dataset.num_rows)]
    if random_sampling:
        if logger:
            logger.info(f"Randomly sampling {num_samples} examples")
        selected_examples = random.sample(examples, min(num_samples, len(examples)))
    else:
        if logger:
            logger.info(f"Using first {num_samples} examples after index {start_index}")
        selected_examples = examples[: min(num_samples, len(examples))]

    # Format examples
    formatted_examples, _ = format_examples(selected_examples, dataset_name)

    # Create output directory
    if output_filename:
        output_dir = f"evaluations/{model_path.split('/')[-1]}_{dataset_name.split('/')[-1]}_{output_filename}"
    else:
        output_dir = (
            f"evaluations/{model_path.split('/')[-1]}_{dataset_name.split('/')[-1]}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Generate responses in batches
    all_results = []

    for batch_start in tqdm(
        range(0, len(formatted_examples), batch_size), desc="Processing batches"
    ):
        batch_end = min(batch_start + batch_size, len(formatted_examples))
        current_batch = formatted_examples[batch_start:batch_end]

        start_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        end_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        prompts = [
            f"{start_prompt}{example['prompt']}{end_prompt}"
            for example in current_batch
        ]
        responses = generate_responses(model, tokenizer, prompts, max_new_tokens, skip_special_tokens=True)

        # Store results
        for i, example in enumerate(current_batch):
            result = {
                "prompt": example["prompt"],
                "reference_answer": example["reference_answer"],
                "model_response": responses[i],
            }
            all_results.append(result)

    # Save results
    with open(f"{output_dir}/results_{timestamp}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate markdown report
    with open(f"{output_dir}/report_{timestamp}.md", "w") as f:
        f.write(f"# Model Evaluation: {model_path} on {dataset_name}\n\n")

        for i, result in enumerate(all_results):
            f.write(f"## Example {i+1}\n\n")
            f.write(f"### Prompt\n\n{result['prompt']}\n\n")
            f.write(f"### Reference Answer\n\n{result['reference_answer']}\n\n")
            f.write(f"### Model Response\n\n{result['model_response']}\n\n")
            f.write("---\n\n")

    if logger:
        logger.info(f"Evaluation complete! Results saved to {output_dir}")
    return all_results


# Quick test: Ask the model a single question
def ask_model_question(
    model_name,
    question,
    max_new_tokens=512,
    enable_logging: bool = False,
    skip_special_tokens: bool = True,
):
    """Ask a single question to the model and print the response"""
    logger = setup() if enable_logging else None
    if logger:
        logger.info(f"Loading model: {model_name}")

    # Load model
    model, tokenizer = load_model(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format question
    start_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    end_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    formatted_question = f"{start_prompt}{question}{end_prompt}"

    # Generate response
    if logger:
        logger.info("Generating response...")
    response = generate_responses(
        model, tokenizer, [formatted_question], max_new_tokens, skip_special_tokens
    )[0]

    return response


if __name__ == "__main__":
    general_config = load_config("random_sampling_config.yaml")
    output_generation_config = general_config["output_generation"]
    dataset_name = general_config["dataset"]['name']
    num_samples = general_config["dataset"].get("num_samples", None)
    dataset_config = load_config("datasets.yaml")[dataset_name]
    
    random.seed(42)
    
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--num_samples", type=int, help="Number of samples to process")
    parser.add_argument("--random_sampling", type=bool, default=False, help="Random sampling")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for dataset processing")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--enable_logging", type=bool, default=False, help="Enable logging")
    
    args = parser.parse_args()
    

    if args.model_path is not None:
        output_generation_config["model"] = args.model_path
    if args.batch_size is not None:
        output_generation_config["batch_size"] = args.batch_size
    
    if args.num_samples is not None:
        num_samples = args.num_samples
    elif num_samples is not None:
        num_samples = num_samples
    else:
        num_samples = 100
    
        
    evaluate_model(
        model_path=output_generation_config["model"],
        dataset_name=dataset_name,
        dataset_subset=dataset_config.get("subset", None),
        dataset_split=dataset_config.get("split", "train"),
        num_samples=num_samples,
        random_sampling=args.random_sampling,
        start_index=args.start_index,
        max_new_tokens=output_generation_config["max_new_tokens"],
        batch_size=output_generation_config["batch_size"],
        output_filename=f"{output_generation_config['model']}_{dataset_name}_{num_samples}_generations".replace("/", "_"),
        enable_logging=args.enable_logging
    )
    
