from datetime import datetime
import os
import logging
import random
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from typing import Literal, List, Dict, Tuple
from distillation_utils import load_config
from torch import amp


def setup():
    """Set up environment, seeds, and logging"""
    load_dotenv()
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_model(model_name_or_path):
    """Load model with optimizations for speed"""
    # Initialize logger
    logger = logging.getLogger(__name__)

    try:
        if os.path.isdir(f"./results/{model_name_or_path}"):
            model_name_or_path = f"./results/{model_name_or_path}"
            logger.info(f"Loading model from local path: {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            token=os.getenv("HF_TOKEN"),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return model, tokenizer
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise


def format_examples(examples, dataset_name):
    if dataset_name == "dmis-lab/MedLFQA":
        formatted_examples = []
        for example in examples:
            must_have = "\n- ".join(example["Must_have"])
            nice_to_have = "\n- ".join(example["Nice_to_have"])

            reference_answer = f"""
{example["Free_form_answer"]}

Must-have reference statements:

- {must_have}

Nice-to-have reference statements:

- {nice_to_have}
            """
            formatted_examples.append(
                {
                    "prompt": example["Question"],
                    "reference_answer": reference_answer,
                }
            )

        evaluation_prompt = """
            You are an impartial judge evaluating answers to long-form biomedical questions.
            You will receive a question, must-have (MH) and nice-to-have (NH) reference statements, and two candidate answers (A and B).
            Choose the answer that aligns best with the reference. Explain your choice in <30 words with a clear, concrete reason.
            If neither answer is acceptable, mention that in the explanation. This is a blind evaluation.
            """
        return formatted_examples, evaluation_prompt

    elif dataset_name == "kaans/medqa-swe-with-responses":
        examples = [
            {
                "prompt": f"{x['question']}\n\n{x['options']}",
                "reference_answer": x["model_response"],
            }
            for x in examples
        ]

        evaluation_prompt = """You are an impartial judge evaluating answers to Swedish biomedical questions.
            You will be given a question, a reference answer, and two candidate answers (A and B).
            Evaluate which answer (A or B) aligns better with the reference answer, and explain concisely (<30 words) and concretely the reason for your choice.
            If neither answer is correct, mention it in the explanation.
            This is a blind evaluation.
            """
        return examples, evaluation_prompt

    elif dataset_name == "kaans/rule_qa":
        start_tokens = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        end_tokens = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        examples = [
            {
                "prompt": f"{start_tokens}{x['text']}{end_tokens}",
                "reference_answer": x["answer"],
            }
            for x in examples
        ]

        evaluation_prompt = """You are an impartial judge evaluating answers to rule-based legal questions.
        You will receive a question, a reference answer, and two candidate answers (A and B).
        Your primary task is to determine which answer (A or B) better aligns with the reference, even if neither is perfect.
        This is a comparative evaluation - focus on relative quality rather than absolute correctness.
        Choose the answer that captures more key points from the reference or has fewer inaccuracies.
        Only choose "neither" if both answers contain serious factual errors that would mislead a reader.
        Explain your choice in <30 words with a clear, concrete reason focusing on the better alignment.
        This is a blind evaluation.
        """

        return examples, evaluation_prompt

    elif dataset_name == "ucinlp/drop":
        system_prompt = "You are a highly intelligent reasoning assistant, expert in solving complex and discrete reasoning problems. For every question, think step-by-step, explicitly state your assumptions, and use clear, logical deductions. Always show your full chain of thought before answering. Even if the answer seems obvious, explain how you arrived at it. Break down the problem thoroughly as an expert would. Answer the following question based on the provided passage."
        examples = [
            {
                "prompt": f"{system_prompt}\n\nPassage: {x['passage']}\n\nQuestion: {x['question']}",
                "reference_answer": ", ".join(x["answers_spans"]["spans"]),
            }
            for x in examples
        ]

        evaluation_prompt = """You are an impartial judge evaluating answers to reading comprehension questions that require discrete reasoning.
            You will receive a passage, a question, a reference answer, and two candidate answers (A and B).
            Evaluate which answer (A or B) better aligns with the reference answer through accurate reasoning.
            Focus on accuracy and logical reasoning rather than narrative style.
            Choose the answer that provides more precise reasoning that matches the reference answer.
            Explain your choice in <50 words with a clear, concrete reason.
            This is a blind evaluation.
            """

        return examples, evaluation_prompt

    elif dataset_name == "openai/gsm8k":
        examples = [
            {
                "prompt": x["question"],
                "reference_answer": x["answer"],
            }
            for x in examples
        ]

        evaluation_prompt = """You are an impartial judge evaluating answers to math problems.
            You will receive a problem, a reference answer, and two candidate answers (A and B).
            Evaluate which answer (A or B) better aligns with the reference answer through accurate reasoning.
            Focus on accuracy and logical reasoning rather than narrative style.
            Choose the answer that provides more precise reasoning that matches the reference answer.
            Explain your choice in <50 words with a clear, concrete reason.
            This is a blind evaluation.
            """

        return examples, evaluation_prompt

    elif dataset_name == "kaans/swefaq":
        examples = [
            {
                "prompt": x["question"],
                "reference_answer": x["candidate_answers"][x["label"]],
            }
            for x in examples
        ]

        evaluation_prompt = """You are an impartial judge evaluating answers to questions collected from FAQs on the websites of Swedish authorities.
            You will be given a question, a reference answer, and two candidate answers (A and B).
            Evaluate which answer (A or B) better aligns with the reference answer in style and accuracy, and explain concisely (<30 words) and concretely the reason for your choice.
            If neither answer is correct, mention it in the explanation.
            This is a blind evaluation.
            """
        return examples, evaluation_prompt

    elif dataset_name == "tatsu-lab/alpaca":
        examples = [
            {
                "prompt": x["instruction"],
                "reference_answer": x["output"],
            }
            for x in examples
        ]

        evaluation_prompt = """
You are an impartial judge evaluating two AI-generated answers to a given prompt.
Your task is to determine which answer better satisfies two criteria:
1. It answers the question accurately and reasonably.
2. It maintains a consistent tone and style of a sassy teenager — informal, bold, a little sarcastic, but still engaging and intelligible.

You will receive:
- An instruction or question.
- Two candidate answers: A and B.

Evaluate both answers blindly. Focus on:
- Correctness or reasonableness of the response.
- How well the answer maintains the 'sassy teenager' style throughout.

Choose the answer that best balances correctness and tone. Explain your choice in under 50 words with a clear and specific justification.
Do not mention the style directly in your explanation — simply justify your pick based on tone and correctness without referring to the criteria.
"""

        return examples, evaluation_prompt

    elif dataset_name == "qiaojin/PubMedQA":

        formatted_examples = []
        for ex in examples:
            final_decision = ex["final_decision"]
            formatted_final_decision = final_decision[0].upper() + final_decision[1:]

            formatted_examples.append(
                {
                    "prompt": ex["question"],
                    "reference_answer": f"{formatted_final_decision} - {ex['long_answer']}",
                }
            )
        evaluation_prompt = """You are an impartial judge comparing two candidate answers (A and B) to a biomedical question against a given reference answer.
For each set—question, reference, A, and B—select which candidate best matches the reference.
Provide a single choice (A or B) and a concise (<30 words) explanation.
A non-answer (e.g., a mere literature-search plan) does not count as correct.
If neither aligns, state that instead.
This is a blind evaluation."""

        return formatted_examples, evaluation_prompt

    elif dataset_name == "Vezora/Tested-143k-Python-Alpaca":
        examples = [
            {
                "prompt": x["instruction"],
                "reference_answer": x["output"],
            }
            for x in examples
        ]

        evaluation_prompt = """
        You are an impartial judge evaluating code answers to a programming problem.
You will receive a problem description, a reference solution, and two candidate solutions (A and B).
Your task is to decide which candidate solution (A or B) better aligns with the reference solution in terms of correctness, logic, and problem-solving approach.

Focus strictly on functional correctness and logical alignment with the reference.
Ignore coding style, formatting, or performance unless it directly affects correctness.
Choose the answer that provides more accurate and logically sound implementation matching the reference.

Explain your choice in under 50 words with a concrete reason.
This is a blind evaluation.
       """
        return examples, evaluation_prompt

    else:
        raise NotImplementedError(f"{dataset_name} is not implemented.")


def generate_responses_batch(model_dict, prompts, max_new_tokens=512):
    """Generate responses for a batch of prompts"""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    # Process all prompts in one batch
    batch_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(
        model.device
    )

    with amp.autocast("cuda"):  # faster inference
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
    batch_responses = []
    for i, output in enumerate(outputs):
        prompt_tokens = len(batch_inputs["input_ids"][i])
        response = tokenizer.decode(output[prompt_tokens:], skip_special_tokens=True)
        batch_responses.append(response)

    return batch_responses


def get_gpt4o_judgment(
    evaluation_prompt, example, answer_a, answer_b, include_reference_answer=False
):
    class Judgment(BaseModel):
        better_answer: Literal["A", "B", "Neither"]
        explanation: str

    client = OpenAI()

    try:

        user_prompt_parts = [f"**Question:**\n\n{example['prompt']}\n"]

        if include_reference_answer:
            user_prompt_parts.append(
                f"**Reference answer:**\n\n{example['reference_answer']}\n"
            )

        user_prompt_parts.append(f"**Answer A:** {answer_a}\n")
        user_prompt_parts.append(f"**Answer B:** {answer_b}\n")
        user_prompt_parts.append(
            "Which answer aligns better with the reference answer? Respond with your judgment and a concise explanation (<50 words) in the required JSON format."
        )

        user_prompt = "\n\n".join(user_prompt_parts)

        response = client.beta.chat.completions.parse(
            model="o3-mini",
            messages=[
                {"role": "system", "content": evaluation_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=Judgment,
        )

        judgment = response.choices[0].message.parsed.model_dump()
        return judgment
    except Exception as e:
        logger.error(f"Error getting model judgment: {e}")
        return {
            "better_answer": "Error",
            "explanation": str(e),
        }


def process_example_batch(
    examples: List[Dict],
    model_1_dict: Dict,
    model_2_dict: Dict,
    randomize_order: bool = True,
    include_reference_answer: bool = False,
) -> List[Tuple[str, Dict]]:
    """Process a batch of examples for comparison"""

    examples, evaluation_prompt = format_examples(
        examples, comparison_config["dataset"]["name"]
    )

    model_1_responses = generate_responses_batch(
        model_1_dict,
        [x["prompt"] for x in examples],
        max_new_tokens=comparison_config["comparison"]["max_new_tokens"],
    )
    model_2_responses = generate_responses_batch(
        model_2_dict,
        [x["prompt"] for x in examples],
        max_new_tokens=comparison_config["comparison"]["max_new_tokens"],
    )

    results = []
    judgment_inputs = []

    for i, example in enumerate(examples):
        response_1 = model_1_responses[i]
        response_2 = model_2_responses[i]

        if randomize_order and random.random() > 0.5:
            response_a = response_1
            response_b = response_2
            model_a_name, model_b_name = model_1_dict["name"], model_2_dict["name"]
            model_order = "normal"
        else:
            response_a = response_2
            response_b = response_1
            model_a_name, model_b_name = model_2_dict["name"], model_1_dict["name"]
            model_order = "reversed"

        judgment_inputs.append(
            {
                "example": example,
                "response_1": response_1,
                "response_2": response_2,
                "response_a": response_a,
                "response_b": response_b,
                "model_a_name": model_a_name,
                "model_b_name": model_b_name,
                "model_order": model_order,
            }
        )

    # Get judgments in parallel
    with ThreadPoolExecutor(max_workers=min(10, len(examples))) as executor:
        judgment_futures = []

        for input_data in judgment_inputs:
            future = executor.submit(
                get_gpt4o_judgment,
                evaluation_prompt,
                input_data["example"],
                input_data["response_a"],
                input_data["response_b"],
                include_reference_answer=include_reference_answer,
            )
            judgment_futures.append((future, input_data))

        for future, input_data in judgment_futures:
            judgment = future.result()
            example = input_data["example"]
            response_1 = input_data["response_1"]
            response_2 = input_data["response_2"]
            model_order = input_data["model_order"]
            model_a_name = input_data["model_a_name"]
            model_b_name = input_data["model_b_name"]

            # Determine the winner based on judgment and order
            if judgment["better_answer"] == "A":
                winner = model_a_name
            elif judgment["better_answer"] == "B":
                winner = model_b_name
            else:
                winner = "Neither"

            # Create markdown content
            markdown_content = f"""
### Prompt

{example['prompt']}

### Reference Answer

{example['reference_answer']}

### Model Responses

<details>
<summary><b> Model 1: ({model_1_dict['name']}) </b></summary>

{response_1}
</details>

<details>
<summary><b> Model 2: ({model_2_dict['name']}) </b></summary>

{response_2}
</details>

### LLM Judgment

**Winner:** {winner}

<details>
<summary><b>Explanation</b></summary>

{judgment["explanation"]}
</details>
"""
            judgment_details = {
                "winner": winner,
                "explanation": judgment["explanation"],
                "model_order": model_order,
            }

            results.append((markdown_content, judgment_details))

    return results


def compare_models_batch(
    model_1_dict,
    model_2_dict,
    examples,
    output_file="comparisons/model_comparison.md",
    batch_size=8,
    generate_markdown=True,
    include_reference_answer=False,
):
    """Compare models using batch processing for efficiency"""
    # Create directory for output if it doesn't exist
    if generate_markdown:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Initialize output file
        with open(output_file, "w") as f:
            f.write(f"# Model Comparison on {comparison_config['dataset']['name']}\n\n")
            f.write("Models being compared:\n\n")
            f.write(f"* **{model_1_dict['name']}**\n")
            f.write(f"* **{model_2_dict['name']}**\n\n")

    accumulated_judgments = {
        model_1_dict["name"]: {"wins": 0, "correct_answers": 0},
        model_2_dict["name"]: {"wins": 0, "correct_answers": 0},
        "ties": 0,
        "details": [],
    }

    # Process batches
    for batch_start in tqdm(
        range(0, len(examples), batch_size), desc="Processing batches"
    ):
        batch_end = min(batch_start + batch_size, len(examples))
        current_batch = examples[batch_start:batch_end]

        # Process the current batch
        batch_results = process_example_batch(
            current_batch,
            model_1_dict,
            model_2_dict,
            include_reference_answer=include_reference_answer,
        )

        # Update accumulated judgments and write to file
        for i, (markdown_content, judgment_details) in enumerate(batch_results):
            # Get the overall index for this example
            example_index = batch_start + i

            # Accumulate judgment results
            accumulated_judgments["details"].append(judgment_details)

            if judgment_details["winner"] == model_1_dict["name"]:
                accumulated_judgments[model_1_dict["name"]]["wins"] += 1
            elif judgment_details["winner"] == model_2_dict["name"]:
                accumulated_judgments[model_2_dict["name"]]["wins"] += 1
            elif judgment_details["winner"] == "Neither":
                accumulated_judgments["ties"] += 1

            # Write to markdown file if requested
            if generate_markdown:
                with open(output_file, "a") as f:
                    f.write(markdown_content)
                    if example_index < len(examples) - 1:
                        f.write("\n\n---\n\n")

    # Add summary to markdown
    if generate_markdown:
        with open(output_file, "a") as f:
            f.write("\n\n## Summary\n\n")
            f.write(
                f"**{model_1_dict['name']}**: {accumulated_judgments[model_1_dict['name']]['wins']} wins\n\n"
            )
            f.write(
                f"**{model_2_dict['name']}**: {accumulated_judgments[model_2_dict['name']]['wins']} wins\n\n"
            )
            f.write(f"**Ties**: {accumulated_judgments['ties']}\n\n")

            # Calculate overall winner
            if (
                accumulated_judgments[model_1_dict["name"]]["wins"]
                > accumulated_judgments[model_2_dict["name"]]["wins"]
            ):
                overall_winner = model_1_dict["name"]
            elif (
                accumulated_judgments[model_2_dict["name"]]["wins"]
                > accumulated_judgments[model_1_dict["name"]]["wins"]
            ):
                overall_winner = model_2_dict["name"]
            else:
                overall_winner = "Tie"

            f.write(f"**Overall winner**: {overall_winner}\n")

    return accumulated_judgments


if __name__ == "__main__":
    logger = setup()
    logger.info("Starting batch evaluation")
    comparison_config = load_config("comparison_config.yaml")
    logger.info("Loaded configuration")

    try:
        logger.info("Loading dataset")
        dataset = (
            load_dataset(
                comparison_config["dataset"]["name"],
                comparison_config["dataset"].get("subset"),
                split=comparison_config["dataset"]["split"],
            )
            if comparison_config["dataset"].get("subset")
            else load_dataset(
                comparison_config["dataset"]["name"],
                split=comparison_config["dataset"]["split"],
            )
        )
        if comparison_config["dataset"]["name"] == "tatsu-lab/alpaca":
            dataset = dataset.filter(lambda x: x["input"] == "")

        logger.info(f"Loaded {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    try:
        logger.info("Loading models")
        model_1, tokenizer = load_model(
            comparison_config["comparison"]["models"]["model_1"]["name"]
        )
        model_2, _ = load_model(
            comparison_config["comparison"]["models"]["model_2"]["name"]
        )
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_1_dict = {
        "name": comparison_config["comparison"]["models"]["model_1"]["name"],
        "model": model_1,
        "tokenizer": tokenizer,
    }

    model_2_dict = {
        "name": comparison_config["comparison"]["models"]["model_2"]["name"],
        "model": model_2,
        "tokenizer": tokenizer,
    }

    # Run comparison with batching
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_file = f"comparisons/{comparison_config['comparison']['name']}/batch_comparison_{comparison_config['dataset']['num_samples']}_{timestamp}.md"

    logger.info(f"Starting comparison, output will be saved to {output_file}")
    examples = [
        dataset[i]
        for i in range(comparison_config["dataset"]["start_index"], dataset.num_rows)
    ]
    if comparison_config["dataset"]["random_sample"]:
        logger.info(
            f"Randomly sampling {comparison_config['dataset']['num_samples']} examples, starting from index {comparison_config['dataset']['start_index']}"
        )
        selected_examples = random.sample(
            examples, comparison_config["dataset"]["num_samples"]
        )
    else:
        logger.info(
            f"Using first {comparison_config['dataset']['num_samples']} examples after index {comparison_config['dataset']['start_index']}"
        )
        selected_examples = examples[: comparison_config["dataset"]["num_samples"]]

    accumulated_judgments = compare_models_batch(
        model_1_dict,
        model_2_dict,
        selected_examples,
        output_file=output_file,
        batch_size=16,
        generate_markdown=True,
        include_reference_answer=comparison_config["dataset"]["name"]
        != "tatsu-lab/alpaca",
    )

    logger.info("Comparison complete!")
    logger.info(
        f"Summary: {model_1_dict['name']} wins: {accumulated_judgments[model_1_dict['name']]['wins']}"
    )
    logger.info(
        f"Summary: {model_2_dict['name']} wins: {accumulated_judgments[model_2_dict['name']]['wins']}"
    )
    logger.info(f"Summary: Ties: {accumulated_judgments['ties']}")

    # Save the accumulated judgments to a JSON file
    with open(
        f"comparisons/{comparison_config['comparison']['name']}/batch_comparison_{timestamp}.json",
        "w",
    ) as f:
        json.dump(accumulated_judgments, f)

    logger.info(f"Saved accumulated judgments to {output_file}")
