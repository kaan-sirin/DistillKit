#!/bin/bash
#SBATCH --account=EUHPC_D17_084 # Update this with your from saldo -b
#SBATCH --partition=boost_usr_prod # This can stay
#SBATCH --qos=normal
#SBATCH --gres=gpu:4 # GPUs PER NODE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL # Sends you an email when it starts/stop
#SBATCH --mail-user=kaansirin@yahoo.com

module load profile/deeplrn
module load python cuda nccl cudnn 

source .venv/bin/activate

# Some environment variables to set up cache directories
export TORCH_HOME=$WORK/torch-cache
export HF_HOME=$WORK/hf-cache
export GPUS_PER_NODE=4

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

# Disable internal parallelism of huggingface's tokenizer since we
# want to retain direct control of parallelism options.
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 1 distill_logits_final.py