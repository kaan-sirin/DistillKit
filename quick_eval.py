from datetime import datetime
import os
import logging
import random
import numpy as np
from openai import OpenAI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dotenv import load_dotenv
from datasets import load_dataset
from distillation_utils import load_config
from evaluation_utils import generate_response
from typing import Literal


def setup():
    load_dotenv()
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name_or_path):
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return model, tokenizer, config
    except Exception as e:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map="auto", token=os.getenv("HF_TOKEN")
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            return model, tokenizer, None
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise


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

        # Skip the prompt tokens
        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_response = tokenizer.decode(
            outputs[0][prompt_tokens:], skip_special_tokens=True
        )
        return generated_response
    except Exception as e:
        raise Exception(f"Error generating response: {e}")



def get_gpt4o_judgment(question, reference_answer, answer_a, answer_b):
    client = OpenAI()

    class Judgment(BaseModel):
        better_answer: Literal["A", "B", "Neither"]
        explanation: str

    try:
        system_prompt = """You are an impartial judge evaluating answers to Swedish biomedical questions.
        You will be given a question, a reference answer, and two candidate answers (A and B).
        Evaluate which answer (A or B) is better at answering the question, and explain concisely and concretely the reason for your choice. 
        If neither answer is correct, mention it in the explanation.
        This is a blind evaluation.
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
        logger.error(f"Error getting gpt-4o judgment: {e}")
        return {
            "better_answer": "Error",
            "one_sentence_explanation": str(e),
        }


# TODO: It should return the markdown content and the other details
def compare_models_on_one_example(
    model_dict1,
    model_dict2,
    example,
    llm_judgment=False,
):
    prompt = f"{example['question']}\n\n" f"{example['options']}\n\n"

    response1 = generate_response(
        model_dict1["model"], model_dict1["tokenizer"], prompt
    )
    response2 = generate_response(
        model_dict2["model"], model_dict2["tokenizer"], prompt
    )

    if llm_judgment:
        if random.random() > 0.5:
            response_a = response1
            response_b = response2
            model_a_name, model_b_name = (
                model_dict1["name"],
                model_dict2["name"],
            )
        else:
            response_a = response2
            response_b = response1
            model_a_name, model_b_name = (
                model_dict2["name"],
                model_dict1["name"],
            )

        judgment = get_gpt4o_judgment(prompt, example["model_response"], response_a, response_b)
        if judgment.better_answer == "A":
            winner = model_a_name
        elif judgment.better_answer == "B":
            winner = model_b_name
        else:
            winner = "Neither"

    markdown_content = f"""
### Example {example['Unnamed: 0']}

{example['question']}

{example['options']}

### Expected Answer
{example['answer']}

### Model Responses

<details>
<summary><b> Model 1: ({model_dict1['name']}) </b></summary>

{response1}
</details>

<details>
<summary><b> Model 2: ({model_dict2['name']}) </b></summary>

{response2}
</details>
"""

    if llm_judgment:
        markdown_content += f"""
### LLM Judgment

**Winner:** {winner}

**Explanation:** {judgment.explanation}
"""
    judgment_details = {
        "winner": winner,
        "explanation": judgment.explanation,
    }

    return markdown_content, judgment_details


def compare_models(
    model_dict1,
    model_dict2,
    examples,
    output_file="comparisons/model_comparison.md",
    generate_markdown=True,
):
    accumulated_judgments = {
        model_dict1["name"]: {"wins": 0, "correct_answers": 0},
        model_dict2["name"]: {"wins": 0, "correct_answers": 0},
        "ties": 0,
        "details": [],  # Store detailed results for each example
    }

    for index, example in enumerate(examples):
        markdown_content, judgment_details = compare_models_on_one_example(
            model_dict1, model_dict2, example, llm_judgment=True
        )

        # Accumulate judgment results
        accumulated_judgments["details"].append(judgment_details)

        if judgment_details["winner"] == model_dict1["name"]:
            accumulated_judgments[model_dict1["name"]]["wins"] += 1
        elif judgment_details["winner"] == model_dict2["name"]:
            accumulated_judgments[model_dict2["name"]]["wins"] += 1
        elif judgment_details["winner"] == "Neither":
            accumulated_judgments["ties"] += 1

        if generate_markdown:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "a") as f:
                if index == 0:
                    f.write("# Model Comparison\n\n")
                    f.write(
                        f"Comparing **{model_dict1['name']}** and **{model_dict2['name']}**\n\n"
                    )
                f.write(markdown_content)
                if index < len(examples) - 1:
                    f.write("\n\n---\n\n")

    # Add summary statistics at the end of the markdown file
    if generate_markdown:
        with open(output_file, "a") as f:
            f.write("\n\n## Summary\n\n")
            f.write(
                f"**{model_dict1['name']}**: {accumulated_judgments[model_dict1['name']]['wins']} wins, {accumulated_judgments[model_dict1['name']]['correct_answers']} correct answers\n\n"
            )
            f.write(
                f"**{model_dict2['name']}**: {accumulated_judgments[model_dict2['name']]['wins']} wins, {accumulated_judgments[model_dict2['name']]['correct_answers']} correct answers\n\n"
            )
            f.write(f"**Ties**: {accumulated_judgments['ties']}\n\n")

            # Calculate overall winner
            if (
                accumulated_judgments[model_dict1["name"]]["wins"]
                > accumulated_judgments[model_dict2["name"]]["wins"]
            ):
                overall_winner = model_dict1["name"]
            elif (
                accumulated_judgments[model_dict2["name"]]["wins"]
                > accumulated_judgments[model_dict1["name"]]["wins"]
            ):
                overall_winner = model_dict2["name"]
            else:
                overall_winner = "Tie"

            f.write(f"**Overall winner**: {overall_winner}\n")

    return accumulated_judgments


if __name__ == "__main__":
    setup()
    SAMPLE_INDEX = 3100

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    general_config = load_config()

    original_model, original_tokenizer, original_config = load_model(
        "meta-llama/Llama-3.2-1B-Instruct"
    )

    try:
        dataset = (
            load_dataset(
                general_config["dataset"]["name"],
                general_config["dataset"]["subset"],
                split=general_config["dataset"]["split"],
            )
            if general_config["dataset"].get("subset")
            else load_dataset(
                general_config["dataset"]["name"],
                split=general_config["dataset"]["split"],
            )
        )

    except Exception as e:
        logger.error("Error loading dataset: %s", e)
        raise

    distilled_models = [
        "./results/distill_medqa_swe_with_responses_300samples_3epochs_03_28_14_00",
        "./results/distill_medqa_swe_with_responses_3000samples_3epochs_03_28_17_14",
        "./results/sft_medqa_swe_with_responses_3000samples_3epochs_03_28_22_04",
    ]
    sample_indices = [100, 500, 1000, 2000, 3100]

    # ------- Code below is for running a single comparison to debug -------

    distilled_model_path = distilled_models[1]
    distilled_model, distilled_tokenizer, distilled_config = load_model(
        distilled_model_path
    )
    distilled_model_dict = {
        "name": distilled_model_path.split("/")[-1],
        "model": distilled_model,
        "tokenizer": distilled_tokenizer,
    }

    sft_model_path = distilled_models[2]
    sft_model, sft_tokenizer, sft_config = load_model(sft_model_path)
    sft_model_dict = {
        "name": sft_model_path.split("/")[-1],
        "model": sft_model,
        "tokenizer": sft_tokenizer,
    }


    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    compare_models(
        sft_model_dict,
        distilled_model_dict,
        [dataset[sample_indices[-1]]],
        output_file=f"comparisons/debug_sft_vs_distilled/comparison_{timestamp}.md",
        generate_markdown=True,
    )

    # ------- Code below is for looping through all the models and comparing them to the original student model -------
    # for model_path in distilled_models:
    #     distilled_model, distilled_tokenizer, distilled_config = load_model(model_path)
    #     try:
    #         # Load the original model
    #         model_dict1 = {
    #             "name": "meta-llama/Llama-3.2-1B-Instruct",
    #             "model": original_model,
    #             "tokenizer": original_tokenizer,
    #         }

    #         model_dict2 = {
    #             "name": f"{model_path.split('/')[-1]}",
    #             "model": distilled_model,
    #             "tokenizer": distilled_tokenizer,
    #         }

    #         for sample_index in samples:
    #             sample = dataset[sample_index]

    #             # Compare models and save results
    #             compare_models(
    #                 model_dict1,
    #                 model_dict2,
    #                 sample,
    #                 f"comparisons/{model_dict2['name']}/example_{sample_index}.md",
    #             )

    #     except Exception as e:
    #         logger.error("Error during model comparison: %s", e)
    #         raise
