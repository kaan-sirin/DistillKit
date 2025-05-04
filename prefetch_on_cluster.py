import os
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def prefetch_dataset(config_path: str, dataset_config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    with open(dataset_config_path) as f:
        datasets = yaml.safe_load(f)
        dataset_config = datasets[config["dataset"]]

    # if a subset is given, pass it; else drop the arg
    if dataset_config.get("subset"):
        load_dataset(config["dataset"], dataset_config["subset"], split=dataset_config["split"])
    else:
        load_dataset(config["dataset"], split=dataset_config["split"])


def prefetch_models(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for model_name in (config["models"]["teacher"], config["models"]["student"]):
        AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)


if __name__ == "__main__":
    # ensure network access
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ["HF_HOME"] = f"$WORK/hf-cache"
    os.environ["TORCH_HOME"] = f"$WORK/torch-cache"
    

    config_file = "config.yaml"
    dataset_config_file = "datasets.yaml"
    prefetch_dataset(config_file, dataset_config_file)
    prefetch_models(config_file)