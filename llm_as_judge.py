import os
import logging
from pprint import pprint
import random
import numpy as np
from pydantic import BaseModel
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
import time
from typing import Literal

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up deterministic behavior
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load models
logger.info("Loading models...")

# Model A: Logits model
model_path_a = "./results/checkpoint-504-logits"
config_a = AutoConfig.from_pretrained(model_path_a)
model_a = AutoModelForCausalLM.from_pretrained(model_path_a)
tokenizer_a = AutoTokenizer.from_pretrained(model_path_a)
logger.info(f"Model A loaded: {model_path_a}")

# Model B: Hard targets model
model_path_b = "./results/checkpoint-504-hard-targets"
config_b = AutoConfig.from_pretrained(model_path_b)
model_b = AutoModelForCausalLM.from_pretrained(model_path_b, device_map="auto")
tokenizer_b = AutoTokenizer.from_pretrained(model_path_b)
logger.info(f"Model B loaded: {model_path_b}")

# Load dataset
try:
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    logger.info("Dataset loaded successfully")
except Exception as e:
    logger.error("Error loading dataset: %s", e)
    raise

# Initialize OpenAI client
client = OpenAI()


def generate_response(model, tokenizer, prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

        # skip the prompt tokens
        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_response = tokenizer.decode(
            outputs[0][prompt_tokens:], skip_special_tokens=True
        )
        return generated_response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error: {str(e)}"


def get_gpt4_judgment(question, reference_answer, answer_a, answer_b):
    class Judgment(BaseModel):
        better_answer: Literal["A", "B"]
        explanation: str

    try:
        system_prompt = """You are an impartial judge evaluating answers to biomedical questions.
        You will be given a question, a reference answer, and two candidate answers (A and B).
        Evaluate which answer (A or B) better addresses the question based on:
        1. Accuracy compared to the reference answer
        2. Completeness of information
        3. Clarity and conciseness
        Do not consider formatting, only content.
        You must not know which model produced which answer - this is a blind evaluation.
        """

        user_prompt = f"""Question: {question}

        Reference Answer: {reference_answer}

        Answer A: {answer_a}

        Answer B: {answer_b}

        Which answer is better? Return your judgment in the required JSON format."""

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format=Judgment,
        )

        judgment = response.choices[0].message.parsed
        return judgment
    except Exception as e:
        logger.error(f"Error getting GPT-4o judgment: {e}")
        return {"better_answer": "Error", "explanation": str(e)}



# TEST - DELETE LATER 
sample = dataset["train"][120010]
question = sample["question"]
reference = sample["long_answer"]
prompt = (
    f"You are a biomedical expert. Provide a concise answer to the following biomedical question.\n\n"
    f"Question: {question}"
)

# Randomize the order of models for each question to ensure fairness
if random.random() > 0.5:
    response_a = generate_response(model_a, tokenizer_a, prompt)
    response_b = generate_response(model_b, tokenizer_b, prompt)
    model_a_name, model_b_name = (
        "checkpoint-504-logits",
        "checkpoint-504-hard-targets",
    )
else:
    response_b = generate_response(model_a, tokenizer_a, prompt)
    response_a = generate_response(model_b, tokenizer_b, prompt)
    model_b_name, model_a_name = (
        "checkpoint-504-logits",
        "checkpoint-504-hard-targets",
    )

# Get judgment
judgment = get_gpt4_judgment(question, reference, response_a, response_b)
pprint(judgment)
exit (0)




# Main evaluation loop
results = []
max_samples = 100
all_indices = list(range(len(dataset["train"])))
shuffled_indices = [idx for idx in all_indices if idx > 3000]
random.shuffle(shuffled_indices)

logger.info(f"Starting evaluation of {max_samples} samples...")

for i, idx in enumerate(shuffled_indices[:max_samples]):
    sample = dataset["train"][idx]
    question = sample["question"]
    reference = sample["long_answer"]

    logger.info(f"Sample {i+1}/{max_samples} (dataset idx: {idx})")

    # Generate prompts
    prompt = (
        f"You are a biomedical expert. Provide a concise answer to the following biomedical question.\n\n"
        f"Question: {question}"
    )

    # Randomize the order of models for each question to ensure fairness
    if random.random() > 0.5:
        response_a = generate_response(model_a, tokenizer_a, prompt)
        response_b = generate_response(model_b, tokenizer_b, prompt)
        model_a_name, model_b_name = (
            "checkpoint-504-logits",
            "checkpoint-504-hard-targets",
        )
    else:
        response_b = generate_response(model_a, tokenizer_a, prompt)
        response_a = generate_response(model_b, tokenizer_b, prompt)
        model_b_name, model_a_name = (
            "checkpoint-504-logits",
            "checkpoint-504-hard-targets",
        )

    # Get judgment
    judgment = get_gpt4_judgment(question, reference, response_a, response_b)

    # Translate judgment back to actual model names
    if judgment.better_answer == "A":
        winner = model_a_name
    elif judgment.better_answer == "B":
        winner = model_b_name
    else:
        winner = "Error"

    # Store results
    result = {
        "question_idx": idx,
        "question": question,
        "reference": reference,
        "logits_response": (
            response_a if model_a_name == "checkpoint-504-logits" else response_b
        ),
        "hard_targets_response": (
            response_b if model_b_name == "checkpoint-504-hard-targets" else response_a
        ),
        "winner": winner,
        "explanation": judgment.explanation,
    }
    results.append(result)

    logger.info(f"Winner for question {i+1}: {winner}")
    logger.info(f"Explanation: {judgment.explanation}")
    logger.info("-" * 50)

    # Save progress after every 10 samples
    if (i + 1) % 10 == 0:
        df = pd.DataFrame(results)
        df.to_csv(f"model_comparison/model_comparison_results_{i+1}.csv", index=False)

    # Add a small delay to avoid rate limits
    time.sleep(1)

# Calculate final statistics
logits_wins = sum(1 for r in results if r["winner"] == "checkpoint-504-logits")
hard_targets_wins = sum(
    1 for r in results if r["winner"] == "checkpoint-504-hard-targets"
)
errors = sum(1 for r in results if r["winner"] == "Error")

logger.info(f"\n\nFinal Results:")
logger.info(
    f"checkpoint-504-logits wins: {logits_wins} ({logits_wins/len(results)*100:.2f}%)"
)
logger.info(
    f"checkpoint-504-hard-targets wins: {hard_targets_wins} ({hard_targets_wins/len(results)*100:.2f}%)"
)
logger.info(f"Errors: {errors} ({errors/len(results)*100:.2f}%)")

# Save final results
df = pd.DataFrame(results)
df.to_csv("model_comparison/model_comparison_final_results.csv", index=False)

# Create a summary file
summary = {
    "total_samples": len(results),
    "logits_wins": logits_wins,
    "logits_win_percentage": logits_wins / len(results) * 100,
    "hard_targets_wins": hard_targets_wins,
    "hard_targets_win_percentage": hard_targets_wins / len(results) * 100,
    "errors": errors,
    "error_percentage": errors / len(results) * 100,
}

with open("model_comparison/model_comparison_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

logger.info("Evaluation complete. Results saved to CSV and JSON files.")
