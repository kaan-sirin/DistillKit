import os
import random
import json
from datetime import datetime
import torch
from tqdm import tqdm
from datasets import load_dataset
from batch_comparison import format_examples, setup, load_model
from torch import amp


def generate_responses(model, tokenizer, prompts, max_new_tokens=512):
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
            temperature=1.0,
            top_p=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

    torch.cuda.synchronize()

    # Decode all responses
    responses = []
    for i, output in enumerate(outputs):
        prompt_tokens = len(batch_inputs["input_ids"][i])
        response = tokenizer.decode(output[prompt_tokens:], skip_special_tokens=True)
        responses.append(response)

    return responses


def evaluate_model(
    model_name,
    dataset_name,
    dataset_subset=None,
    dataset_split="test",
    num_samples=10,
    random_sampling=True,
    start_index=0,
    max_new_tokens=512,
    batch_size=4,
    output_filename=None,
):
    """Evaluate a model on a dataset"""
    logger = setup()
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    logger.info(f"Evaluating model {model_name} on dataset {dataset_name}")

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

    # Load model
    try:
        model, tokenizer = load_model(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Select examples
    examples = [dataset[i] for i in range(start_index, dataset.num_rows)]
    if random_sampling:
        logger.info(f"Randomly sampling {num_samples} examples")
        selected_examples = random.sample(examples, min(num_samples, len(examples)))
    else:
        logger.info(f"Using first {num_samples} examples after index {start_index}")
        selected_examples = examples[: min(num_samples, len(examples))]

    # Format examples
    formatted_examples, _ = format_examples(selected_examples, dataset_name)

    # Create output directory
    if output_filename:
        output_dir = f"evaluations/{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}_{output_filename}"
    else:
        output_dir = (
            f"evaluations/{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Generate responses in batches
    all_results = []

    for batch_start in tqdm(
        range(0, len(formatted_examples), batch_size), desc="Processing batches"
    ):
        batch_end = min(batch_start + batch_size, len(formatted_examples))
        current_batch = formatted_examples[batch_start:batch_end]

        cot_prompt = """
        **Question:** The girls are trying to raise money for a carnival. Kim raises $320 more than Alexandra, who raises $430, and Maryam raises $400 more than Sarah, who raises $300. How much money, in dollars, did they all raise in total?

        **Answer:**
        Alexandra raises 430 dollars.
        Kim raises 320+430=750 dollars.
        Sarah raises 300 dollars.
        Maryam raises 400+300=700 dollars.
        In total, they raise 750+430+400+700=2280 dollars.
        

        """
        style_prompt_swe = "[INST] Svara kort och tydligt på frågan nedan. Det är **viktigt** att svaret låter helt naturligt, precis som om det vore skrivet av en person som har svenska som modersmål. [/INST]"
        style_prompt_sassy = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You're a sassy teenager. Answer the following question in a sassy way.
        <|eot_id|><|start_header_id|>user<|end_header_id|>"""
        
        style_prompt_cat = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You're a cat. Answer the following question. Meow meow meow!
        <|eot_id|><|start_header_id|>user<|end_header_id|>"""
        
        end_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        style_prompt_ELI12 = "[INST] Answer the following question as if I am a smart 12 year old. [/INST]"

        prompts = [
            f"{style_prompt_sassy}{example['prompt']}{end_prompt}"
            for example in current_batch
        ]
        responses = generate_responses(model, tokenizer, prompts, max_new_tokens)

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
        f.write(f"# Model Evaluation: {model_name} on {dataset_name}\n\n")

        for i, result in enumerate(all_results):
            f.write(f"## Example {i+1}\n\n")
            f.write(f"### Prompt\n\n{result['prompt']}\n\n")
            f.write(f"### Reference Answer\n\n{result['reference_answer']}\n\n")
            f.write(f"### Model Response\n\n{result['model_response']}\n\n")
            f.write("---\n\n")

    logger.info(f"Evaluation complete! Results saved to {output_dir}")
    return all_results


if __name__ == "__main__":
    # Example usage
    evaluate_model(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        dataset_name="kaans/rule_qa",
        # dataset_subset="main",
        dataset_split="train",
        num_samples=8,
        random_sampling=False,
        max_new_tokens=512,
        batch_size=4,
        output_filename="sassy",
    )
