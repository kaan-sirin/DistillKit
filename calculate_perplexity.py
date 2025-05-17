import argparse
import json
import random
from datasets import load_dataset
from evaluate import load
from pathlib import Path
import logging

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define constants
PERPLEXITY_FILE = Path("perplexity.json")
RANDOM_SEED = 42

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, help="model path")
    ap.add_argument("--model_identifier", type=str, help="model identifier")
    ap.add_argument("--num_samples", type=int, default=100, help="number of samples")
    args = ap.parse_args()

    perplexity = load("perplexity", module_type="metric")

    # Initialize perplexity.json as an empty json array if it doesn't exist
    if not PERPLEXITY_FILE.exists():
        with open(PERPLEXITY_FILE, "w") as f:
            json.dump([], f)

    # Load dataset
    dataset = load_dataset(
        "nicher92/magpie_llama70b_260k_filtered_swedish", split="train"
    )
    logging.info(f"Loaded {len(dataset)} examples")
    logging.info("Filtering: task_category != Math")
    dataset = dataset.filter(lambda x: x["task_category"] != "Math")
    logging.info(f"{len(dataset)} examples remaining")
    logging.info("Filtering: Math not in other_task_category")
    dataset = dataset.filter(lambda x: "Math" not in x["other_task_category"])
    logging.info(f"{len(dataset)} examples remaining")

    num_samples_for_perplexity = args.num_samples
    random.seed(RANDOM_SEED)
    random_indices = random.sample(range(len(dataset)), num_samples_for_perplexity)
    sample_texts = [dataset[i]["response"] for i in random_indices]
    logging.info(
        f"\nSelected {len(sample_texts)} sample texts for perplexity calculation."
    )

    try:
        valid_sample_texts = [text for text in sample_texts if text and text.strip()]
        if not valid_sample_texts:
            logging.warning(
                "No valid (non-empty) sample texts found. Skipping perplexity calculation for %s.",
                args.model_identifier,
            )
        else:
            result = perplexity.compute(
                predictions=valid_sample_texts,
                model_id=args.model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            logging.info(
                "Perplexity for %s: %.4f",
                args.model_identifier,
                result["mean_perplexity"],
            )

            # Add the perplexity result to the json file
            try:
                with open(PERPLEXITY_FILE, "r") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logging.warning(
                    "Perplexity file %s not found or corrupted. Initializing with new data.",
                    PERPLEXITY_FILE,
                )
                data = []

            data.append(
                {
                    "model_identifier": args.model_identifier,
                    "model_path": args.model_path,
                    "num_samples": num_samples_for_perplexity,
                    "perplexity": result["mean_perplexity"],
                }
            )

            data.sort(key=lambda item: item["perplexity"])

            with open(PERPLEXITY_FILE, "w") as f:
                json.dump(data, f, indent=4)
    except ValueError as ve:
        logging.error(
            "ValueError during perplexity calculation for %s: %s",
            args.model_identifier,
            ve,
        )
    except RuntimeError as rte:
        logging.error(
            "RuntimeError during perplexity calculation for %s: %s",
            args.model_identifier,
            rte,
        )
    except Exception as e:
        logging.error(
            "Unexpected error calculating perplexity for %s: %s",
            args.model_identifier,
            e,
            exc_info=True,
        )
