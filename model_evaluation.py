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
    enable_logging: bool = False,
):
    """Evaluate a model on a dataset"""
    logger = setup() if enable_logging else None
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    if logger:
        logger.info(f"Evaluating model {model_name} on dataset {dataset_name}")

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

    # Load model
    try:
        model, tokenizer = load_model(model_name)
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

        start_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        end_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

        prompts = [
            f"{start_prompt}{example['prompt']}{end_prompt}"
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
    top_k_forward = (
        "results/top_k_forward_alpaca_320_samples_3_epochs_3_top_96_04_24_13-14"
    )
    top_k_reverse = (
        "results/top_k_reverse_alpaca_320_samples_3_epochs_3_top_96_04_24_12-30"
    )
    random_sample_small = "results/sparse_kd_student_20250421_230313_final_1000"
    random_sample = "results/sparse_kd_student_20250422_114315_final_10000"
    sft = "results/sft_distill_alpaca_320samples_3epochs_04_08_16_08"
    teacher = "meta-llama/Llama-3.2-3B-Instruct"

    code_9600 = (
        "distilled_models/CodeAlpaca_20k_forward_9600_samples_3_epochs_05_01_02_27"
    )

    # texts = [
    #     "After weeks of relentless rain, the sun finally broke through the clouds this afternoon.",
    #     "She whispered the secret confidently, knowing it would change their plans forever.",
    #     "The sun was shining brightly, and the birds were singing sweetly.",
    # ]
    # for text in texts:
    #     print(f"Text: {text}")
    #     print("-" * 50)
    #     print(
    #         ask_model_question(
    #             "results/sparse_kd_student_translation_20250502_120451_320_samples_10000",
    #             text,
    #             skip_special_tokens=True,
    #         )
    #     )
    #     print("=" * 50)
    # exit()

    # Example usage
    # evaluate_model(
    #     model_name=distilled_large,
    #     dataset_name="kaans/rule_qa",
    #     # dataset_subset="main",
    #     dataset_split="train",
    #     num_samples=16,
    #     random_sampling=False,
    #     max_new_tokens=512,
    #     batch_size=4,
    #     output_filename="sassy_16_random_sample_logit_distilled_large",
    #     # enable_logging=True # Uncomment to enable logging
    # )

    # Edit the model name and question below for quick testing
    translation_prompt = "Translate the following English sentence into idiomatic Swedish. Focus on natural, fluent phrasing that a native speaker would use in everyday conversation, not just literal translations. Maintain the meaning and tone of the original. Answer with only the Swedish translation, no other text."
    questions = [
        "After weeks of relentless rain, the sun finally broke through the clouds this afternoon.",
        "She whispered the secret confidently, knowing it would change their plans forever.",
        "The sun was shining brightly, and the birds were singing sweetly.",
        "Which power block won the second world war?",
        "Explain what magnetism is as if I were a 10-year-old, using a real-world analogy.",
        "List step‑by‑step instructions for cooking a perfect omelette, including tips to avoid common mistakes.",
        "Summarize the process of photosynthesis in two sentences, preserving the key facts and tone.",
        "Write a 6‑line poem in the style of Shakespeare about the changing seasons.",
        "Given this Python function stub, complete it to compute the nth Fibonacci number efficiently:\n```python\ndef fib(n):\n    # your code here\n```",
    ]

    math_prompt = "Think in enumerated steps internally, then present only the final answer externally."
    math_questions = [
        "A school sold 128 tickets to a play on Monday and twice as many on Tuesday. If each ticket costs $7, how much money did the school make from ticket sales on those two days?",
        "Emma read 18 pages of a book on each weekday and 45 pages on each weekend day. After two full weeks, how many pages has she finished?",
        "A bakery uses 250 g of flour for one loaf of bread and 120 g for one batch of cookies. If the bakery has 5 kg of flour and makes 8 batches of cookies, how many full loaves can it still bake?",
        "A rectangular garden is 6 m wide. Its perimeter is 40 m. What is the garden’s length?",
        "A jar contains red, blue, and green marbles in the ratio 3 : 4 : 5. If there are 48 blue marbles, how many marbles are in the jar altogether?",
    ]

    code_questions = [
        #         """Implement the function f that takes n as a parameter,
        # and returns a list of size n, such that the value of the element at index i is the factorial of i if i is even
        # or the sum of numbers from 1 to i otherwise.
        # i starts from 1.
        # the factorial of i is the multiplication of the numbers from 1 to i (1 * 2 * ... * i).
        # Example:
        # f(5) == [1, 2, 6, 24, 15]
        # """,
        "Write a function in HTML for creating a table of n rows and m columns.",
    ]

    # for question in code_questions:
    #     print("Question:")
    #     print(question)
    #     print("-" * 50)
    #     print("Teacher:")
    #     print(ask_model_question(teacher, question, skip_special_tokens=True))
    #     print("-" * 50)
    #     print("Distilled model:")
    #     print(ask_model_question(code_9600, question, skip_special_tokens=True))
    #     print("-" * 50)

    conversation = [
        "Say , Jim , how about going for a few beers after dinner ?",
        "You know that is tempting but is really not good for our fitness .",
        "What do you mean ? It will help us to relax .",
        "Do you really think so ? I don't . It will just make us fat and act silly . Remember last time ?",
        "I guess you are right.But what shall we do ? I don't feel like sitting at home .",
        "I suggest a walk over to the gym where we can play singsong and meet some of our friends .",
        "That's a good idea . I hear Mary and Sally often go there to play pingpong.Perhaps we can make a foursome with them .",
        "Sounds great to me ! If they are willing , we could ask them to go dancing with us.That is excellent exercise and fun , too .",
        "Good.Let ' s go now .",
        "All right .",
    ]
    convo = ""
    for i, turn in enumerate(conversation):
        # Remove space before punctuation
        turn = re.sub(r"\s+([^\w\s])", r"\1", turn)
        # Remove space after punctuation
        turn = re.sub(r"([^\w\s])\s+", r"\1", turn)
        # Add space after punctuation (but not apostrophe) if followed by a letter/digit
        turn = re.sub(r"([^\w\s'])(?=[a-zA-Z0-9])", r"\1 ", turn)
        if i % 2 == 0:
            convo += "A: " + turn + "\n"
        else:
            convo += "B: " + turn + "\n"

    text = f"""Översätt den här vardagliga konversationen från engelska till naturlig svenska. Använd uttryck och formuleringar som en person med svenska som modersmål faktiskt skulle säga i samma situation. Fånga tonen och känslan i samtalet – inte bara orden. Svara enbart med översättningen.    
    
{convo}
    """
    print(text)
