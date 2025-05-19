import re
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from transformers import TrainerCallback
import numpy as np
import os
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F


USER_PROMPT_START = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
USER_PROMPT_END = f"<|eot_id|>"
ASSISTANT_PROMPT_START = f"<|start_header_id|>assistant<|end_header_id|>"
ASSISTANT_PROMPT_END = f"<|eot_id|>"


def medqa_swe_format(example):
    try:
        text = (
            "Du är en medicinsk expert. Svara på följande flervalsfråga. \n\n"
            f"{example['question']}\n\n"
            f"{example['options']}\n\n"
            f"{example['model_response']}"
        )

        return {"text": text}

    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def medlfqa_format(example):
    try:
        text = (
            f"Question: {example['Question']}\n\n"
            f"Answer: {example['Free_form_answer']}"
        )
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def decode_sparse_sequence(sequence, tokenizer):
    decoded_output = ""
    for token in sequence:
        token.sort(key=lambda pair: pair[1], reverse=True)
        most_probable_id = int(token[0][0])
        decoded_output += tokenizer.decode(most_probable_id)

    return decoded_output


def alpaca_format(example, tokenizer):
    try:
        user_prompt_start = f"<|start_header_id|>user<|end_header_id|>"
        user_prompt_end = f"<|eot_id|>"
        decoded_output = decode_sparse_sequence(example["sparse_logits"], tokenizer)
        text = f"{user_prompt_start}{example['instruction']}{user_prompt_end}{decoded_output}"
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def dailydialog_format(example, tokenizer):
    try:
        decoded_output = decode_sparse_sequence(example["sparse_logits"], tokenizer)
        text = f"{USER_PROMPT_START}\n\n{format_dialog(example['utterances']) + USER_PROMPT_END}{decoded_output}"
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def pubmedqa_format(example, tokenizer):
    try:
        decoded_output = decode_sparse_sequence(example["sparse_logits"], tokenizer)
        text = f"{USER_PROMPT_START}\n\n{example['question']}{USER_PROMPT_END}{decoded_output}"
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def code_alpaca_format(example):
    try:
        text = f"{USER_PROMPT_START}\n\n{example['instruction']}"
        if example.get("input"):
            text += f"\n\n{example['input']}"
        text += f"{USER_PROMPT_END}{ASSISTANT_PROMPT_START}\n\n{example['output']}{ASSISTANT_PROMPT_END}"
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def python_alpaca_format(example):
    try:
        text = f"{USER_PROMPT_START}\n\n{example['instruction']}{USER_PROMPT_END}{ASSISTANT_PROMPT_START}\n\n{example['output']}{ASSISTANT_PROMPT_END}"
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def gsm8k_format(example):
    try:
        text = f"{USER_PROMPT_START}\n\n{example['question']}{USER_PROMPT_END}{ASSISTANT_PROMPT_START}\n\n{example['answer']}{ASSISTANT_PROMPT_END}"
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def random_sampled_gsm8k_format(example, tokenizer):
    try:
        decoded_output = decode_sparse_sequence(example["sparse_logits"], tokenizer)
        text = f"{USER_PROMPT_START}\n\n{example['question']}{USER_PROMPT_END}{decoded_output}"
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def random_sampled_medqa_swe_format(example, tokenizer):
    try:
        decoded_output = decode_sparse_sequence(example["sparse_logits"], tokenizer)
        text = f"{USER_PROMPT_START}\n\n{example['question']}\n\n{example['options']}{USER_PROMPT_END}{decoded_output}"
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def magpie_format(example):
    try:
        text = f"{USER_PROMPT_START}\n\n{example['instruction']}{USER_PROMPT_END}{ASSISTANT_PROMPT_START}\n\n{example['response']}{ASSISTANT_PROMPT_END}"
        return {"text": text}
    except Exception as e:
        print(f"Sample keys: {list(example.keys())}")
        print(f"Error formatting example: {e}")
        raise


def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            if config.get("training", {}).get("learning_rate"):
                config["training"]["learning_rate"] = float(
                    config["training"]["learning_rate"]
                )
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: {config_path} not found. Program will exit.")
        raise  # Still crash, but with a clearer message
    except yaml.YAMLError as e:
        print(f"Error parsing {config_path}: {e}")
        raise


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def reverse_kld(student_logits, teacher_logits, reduction="none"):
    log_ps = F.log_softmax(student_logits, dim=-1)
    ps = log_ps.exp()  # p_s

    with torch.no_grad():
        log_pt = F.log_softmax(teacher_logits, dim=-1)  # log p_t

    # Return per-element KLD without summing over vocabulary dimension
    per_element_kld = ps * (log_ps - log_pt)

    if reduction == "none":
        return per_element_kld
    elif reduction == "mean":
        # average over all elements in the batch
        return per_element_kld.mean()
    elif reduction == "sum":
        # sum over all elements in the batch
        return per_element_kld.sum()
    elif reduction == "batchmean":
        # sum over all elements and divide by batch size
        batch_size = student_logits.shape[0]
        return per_element_kld.sum() / batch_size
    else:
        raise ValueError(
            f"Invalid reduction type: {reduction}. Choose from 'none', 'mean', 'sum', 'batchmean'."
        )


def get_max_token_length(
    dataset, tokenizer, generate_plot=True, plot_path="token_length_stats.png"
):
    max_length = 0
    lengths = []

    max_index = 1000
    for i, example in enumerate(dataset):
        try:

            # Tokenize without padding or truncation to get true length
            tokens = tokenizer(example["text"], truncation=False, padding=False)

            # Get token count
            length = len(tokens.input_ids)
            lengths.append(length)

            # Update max length
            if length > max_length:
                max_length = length
                max_index = i
            # Print progress occasionally
            if i % 1000 == 0:
                print(f"Processed {i} examples. Current max length: {max_length}")

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            print(f"Example keys: {list(example.keys())}")

    # Calculate statistics
    stats = {
        "max_length": max_length,
        "max_index": max_index,
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "min": np.min(lengths),
        "percentiles": {
            "25": np.percentile(lengths, 25),
            "50": np.percentile(lengths, 50),
            "75": np.percentile(lengths, 75),
            "90": np.percentile(lengths, 90),
            "95": np.percentile(lengths, 95),
            "99": np.percentile(lengths, 99),
        },
        "total_examples": len(lengths),
    }

    # Generate histogram plot
    if generate_plot:
        plt.figure(figsize=(12, 8))

        # Histogram
        plt.subplot(2, 1, 1)
        plt.hist(lengths, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(
            stats["mean"],
            color="red",
            linestyle="dashed",
            linewidth=1,
            label=f'Mean: {stats["mean"]:.1f}',
        )
        plt.axvline(
            stats["median"],
            color="green",
            linestyle="dashed",
            linewidth=1,
            label=f'Median: {stats["median"]:.1f}',
        )
        plt.axvline(
            stats["max_length"],
            color="purple",
            linestyle="dashed",
            linewidth=1,
            label=f'Max: {stats["max_length"]}',
        )
        plt.title("Token Length Distribution")
        plt.xlabel("Token Length")
        plt.ylabel("Number of Examples")
        plt.legend()
        plt.grid(alpha=0.3)

        # Box plot
        plt.subplot(2, 1, 2)
        plt.boxplot(lengths, vert=False, patch_artist=True)
        plt.title("Token Length Box Plot")
        plt.xlabel("Token Length")
        plt.grid(alpha=0.3)

        # Save figure
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        stats["plot_path"] = plot_path

    print(f"Analysis complete. Maximum token length: {max_length}")
    print(f"Mean token length: {stats['mean']:.2f}")
    print(f"Median token length: {stats['median']:.2f}")
    print(f"95th percentile: {stats['percentiles']['95']:.2f}")

    return max_length, max_index, stats


def format_dialog(utterances):
    dialog = ""
    for i, turn in enumerate(utterances):
        # Remove space before punctuation
        turn = re.sub(r"\s+([^\w\s])", r"\1", turn)
        # Remove space after punctuation
        turn = re.sub(r"([^\w\s])\s+", r"\1", turn)
        # Add space after punctuation (but not apostrophe) if followed by a letter/digit
        turn = re.sub(r"([^\w\s'])(?=[a-zA-Z0-9])", r"\1 ", turn)
        if i % 2 == 0:
            dialog += "Person A: " + turn + "\n"
        else:
            dialog += "Person B: " + turn + "\n"
    return dialog


if __name__ == "__main__":
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    config = load_config()

    # ---- Load dataset --------------------------------------------------------
    dataset_name = config["dataset"]
    dataset_config = load_config("datasets.yaml")[dataset_name]
    dataset = (
        load_dataset(
            dataset_name,
            dataset_config["subset"],
            split=dataset_config["split"],
        )
        if dataset_config.get("subset")
        else load_dataset(dataset_name, split=dataset_config["split"])
    )

    dataset = dataset.map(code_alpaca_format)
    print("\n\n", dataset[0]["text"])
    # --------------------------------------------------------------------------

    # ---- Get max token length ------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], token=HF_TOKEN
    )
    max_tokens, max_index, token_stats = get_max_token_length(dataset, tokenizer)
    print(f"Max token length: {max_tokens}, Max index: {max_index}")
    print(f"\n\nToken stats: {token_stats}")

    # Calculate coverage using percentiles
    max_length = config["tokenizer"]["max_length"]
    percentiles = token_stats["percentiles"]

    coverage = 0
    for percentile in token_stats["percentiles"]:
        if int(percentiles[percentile]) <= max_length:
            coverage = percentile
        else:
            break

    print(f"\n\nCoverage (at least): {coverage}%")
