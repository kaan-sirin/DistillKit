#!/bin/bash
#SBATCH --account=EUHPC_D16_003 # Update this with your from saldo -b
#SBATCH --partition=boost_usr_prod # This can stay
#SBATCH --qos=boost_qos_dbg # For production write "normal" here
#SBATCH --gres=gpu:1 # GPUs PER NODE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:15:00
#SBATCH --mail-type=ALL # Sends you an email when it starts/stop
#SBATCH --mail-user=<put_email@here.com>

module load profile/deeplrn
module load cineca-ai/4.3.0

# Some environment variables to set up cache directories
export TORCH_HOME=$WORK/torch-cache
export HF_HOME=$WORK/hf-cache
export GPUS_PER_NODE=1
mkdir -p $TORCH_HOME $HF_HOME

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_OFFLINE=1 # Check if this actually works

# Disable internal parallelism of huggingface's tokenizer since we
# want to retain direct control of parallelism options.
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes $GPUS_PER_NODE distill_logits_final.py
