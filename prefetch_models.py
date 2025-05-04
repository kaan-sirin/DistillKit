from huggingface_hub import snapshot_download
from transformers import AutoTokenizer           
import os, yaml
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = "/leonardo_work/EUHPC_D17_084/hf-cache"

def prefetch_models(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for repo in (cfg["models"]["teacher"], cfg["models"]["student"]):
        snapshot_download(repo_id=repo,
                          cache_dir=CACHE_DIR,
                          token=HF_TOKEN,
                          resume_download=True)

        AutoTokenizer.from_pretrained(repo,
                                      cache_dir=CACHE_DIR,
                                      token=HF_TOKEN)

if __name__ == "__main__":
    prefetch_models("config.yaml")