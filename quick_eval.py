import os
import logging
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dotenv import load_dotenv
from datasets import load_dataset
import evaluate


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, device_map="auto", token=os.getenv("HF_TOKEN")
# )

model_path = "./results/checkpoint-504"
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

seed_value = 42  # You can choose any number
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
except Exception as e:
    logger.error("Error loading dataset: %s", e)
    raise

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

predictions = []
references = []

try:
    for i in range(2):
        if i == 0:
            continue
        sample = dataset["train"][i]
        prompt = (
            f"You are a biomedical expert. Answer the following biomedical question. Provide a concise, evidence-backed response.\n\n"
            f"Question: {sample['question']}"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  
            num_beams=1,  
            temperature=1.0,
            top_p=1.0,  
        )

        # skip the prompt tokens
        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_response = tokenizer.decode(
            outputs[0][prompt_tokens:], skip_special_tokens=True
        )

        logger.info(
            f"\n\nPROMPT: {prompt}\n\n"
            f"MODEL'S ANSWER: {generated_response}\n\n"
            f"EXPECTED ANSWER: {sample['long_answer']}\n\n"
            f"{'-' * 50}"
        )

        predictions.append(generated_response)
        references.append(sample["long_answer"])

    # rouge_results = rouge.compute(predictions=predictions, references=references)
    # bertscore_results = bertscore.compute(
    #     predictions=predictions, references=references, lang="en"
    # )

    # logger.info("Evaluation results:\n\n")
    # logger.info(f"\n\nROUGE evaluation results: {rouge_results}")
    # logger.info(f"\n\nBERTScore evaluation results: {bertscore_results}")

    #     logger.info(f"Model answered: {generated_response}, expected: {sample['answer']}\n")
    #     if len(generated_response.strip()) == 1:
    #         tot += 1
    #         if generated_response.strip() == sample['answer']:
    #             correct += 1
    # logger.info(f"Accuracy: {correct/tot} ({correct}/{tot})")

except Exception as e:
    logger.error("Error during generation: %s", e)
    raise
