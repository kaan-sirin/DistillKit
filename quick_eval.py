import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from datasets import load_dataset


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_dir = f"./models/{model_name.replace('/', '_')}"

try:
    if not os.path.exists(model_dir):
        logger.info("Model not found locally. Downloading from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=os.getenv("HF_TOKEN")
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", token=os.getenv("HF_TOKEN")
        )
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    else:
        logger.info("Loading model from local directory.")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
except Exception as e:
    logger.error("Error loading or downloading the model: %s", e)
    raise

try:
    dataset = load_dataset("nicher92/medqa-swe")
except Exception as e:
    logger.error("Error loading dataset: %s", e)
    raise

try:
    tot = 0 # only count one-letter responses
    correct = 0
    for i in range(20):
        sample = dataset["train"][i]
        prompt = (
            "You are a knowledgeable medical expert. Answer the following multiple-choice question by replying with only the letter corresponding to the correct option and nothing else. Your answer must consist of only one letter.\n\n"
            f"Question: {sample['question']}\n\n"
            f"Options: \n{sample['options']}\n\n"
            f"Answer: "
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        
        # skip the prompt tokens
        prompt_tokens = inputs['input_ids'].shape[-1]
        generated_response = tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)
        
        logger.info(f"Model answered: {generated_response}, expected: {sample['answer']}\n")
        if len(generated_response.strip()) == 1:
            tot += 1
            if generated_response.strip() == sample['answer']:
                correct += 1
    logger.info(f"Accuracy: {correct/tot} ({correct}/{tot})")

except Exception as e:
    logger.error("Error during generation: %s", e)
    raise
