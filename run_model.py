import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import amp


def load_model(model_name_or_path: str):
    """Load model and tokenizer."""
    # Simple check if it's a local path that might be relative
    # to a common results directory, otherwise treat as HF path or absolute.
    # This is a simplified version from batch_comparison.py
    possible_local_path = os.path.join("results", model_name_or_path)
    if os.path.isdir(model_name_or_path):
        actual_model_path = model_name_or_path
    elif os.path.isdir(possible_local_path):
        actual_model_path = possible_local_path
    else:
        actual_model_path = model_name_or_path
        
    print(f"Loading model from: {actual_model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        actual_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.getenv("HF_TOKEN"), # Assumes HF_TOKEN might be needed
    )
    tokenizer = AutoTokenizer.from_pretrained(actual_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_responses(
    model, tokenizer, prompts: list[str], max_new_tokens: int = 512, skip_special_tokens: bool = False
) -> list[str]:
    """Generate responses for a list of prompts."""
    batch_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(
        model.device
    )

    with amp.autocast("cuda"): # Assuming CUDA is available as in original script
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, # Using conservative generation parameters
            num_beams=1,
            top_p=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

    torch.cuda.synchronize() # Ensure completion on CUDA device

    responses = []
    for i, output in enumerate(outputs):
        prompt_tokens = len(batch_inputs["input_ids"][i])
        response = tokenizer.decode(
            output[prompt_tokens:], skip_special_tokens=skip_special_tokens
        )
        responses.append(response)

    return responses


def main():
    """Main function to load model and generate response based on CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a response from a model given a system and user prompt."
    )
    parser.add_argument(
        "model_path", type=str, help="Path to the model or model name on Hugging Face."
    )
    parser.add_argument(
        "system_prompt", type=str, help="The system prompt for the model."
    )
    parser.add_argument("user_prompt", type=str, help="The user prompt for the model.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--skip_special_tokens", type=bool, default=False, help="Whether to skip special tokens in the output."
    )

    args = parser.parse_args()

    try:
        model, tokenizer = load_model(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Llama 3.2 style prompt format
    # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # {system_prompt}
    # <|eot_id|><|start_header_id|>user<|end_header_id|>
    # {user_prompt}
    # <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{args.system_prompt}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{args.user_prompt}\n"
        f"<|eot_id|>"
    )
    
    print("\n--- Prompt Start ---")
    print(formatted_prompt)
    print("--- Prompt End ---\n")


    try:
        print("Generating response...")
        responses = generate_responses(
            model, 
            tokenizer, 
            [formatted_prompt], 
            max_new_tokens=args.max_new_tokens,
            skip_special_tokens=args.skip_special_tokens
        )
        if responses:
            print("\n------------- Model response ------------------------------\n")
            print(responses[0])

        else:
            print("No response generated.")
            
    except Exception as e:
        print(f"Error during response generation: {e}")


if __name__ == "__main__":
    main() 