import torch
from pathlib import Path
import logging
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def combine_logit_files(file_paths, output_dir=None):
    """
    Combine multiple logit files into a single file.

    Args:
        file_paths: List of paths to logit files in the order they should be combined
        output_dir: Directory to save the combined file (defaults to same directory as first file)
    """
    if not file_paths:
        logger.error("No files provided")
        return None

    all_logits = []
    total_length = 0

    # Load and combine logits
    for path in file_paths:
        logger.info(f"Loading {path}")
        try:
            logits = torch.load(path)
            all_logits.extend(logits)
            total_length += len(logits)
            logger.info(f"Loaded {len(logits)} examples")
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    # Determine output path
    if output_dir is None:
        output_dir = Path(file_paths[0]).parent
    else:
        output_dir = Path(output_dir)

    output_path = output_dir / f"combined_sparse_logits_{total_length}.pt"

    # Save combined logits
    torch.save(all_logits, output_path)
    logger.info(f"Saved {total_length} combined examples to {output_path}")

    return output_path


if __name__ == "__main__":
    # logits = torch.load("generated_tokens/pubmedqa/combined_sparse_logits_16.pt")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    # print(len(logits))
    # for seq in logits:
    #     for step in seq:
    #         step.sort(key=lambda x: x[1], reverse=True)
    #         print(tokenizer.decode(step[0][0]), end="")
    #     exit()

    paths = [
        "teacher_random_logits_0-599_R50000_tau1.0.pt",
        "teacher_random_logits_600-799_R50000_tau1.0.pt",
        "teacher_random_logits_800-999_R50000_tau1.0.pt",
        "teacher_random_logits_1000-1199_R50000_tau1.0.pt",
        "teacher_random_logits_1200-1799_R50000_tau1.0.pt",
        "teacher_random_logits_1800-2399_R50000_tau1.0.pt",
        "teacher_random_logits_2400-2999_R50000_tau1.0.pt",
    ]

    # paths = [
    #     "teacher_random_logits_0-399_R50000_tau0.8.pt",
    #     "teacher_random_logits_400-799_R50000_tau0.8.pt",
    #     "teacher_random_logits_800-1199_R50000_tau0.8.pt",
    #     "teacher_random_logits_1200-1599_R50000_tau0.8.pt",
    # ]

    paths = [
        f"/leonardo_work/EUHPC_D17_084/DistillKit/generated_tokens/medqa_swe/{path}"
        for path in paths
    ]

    combine_logit_files(paths)
