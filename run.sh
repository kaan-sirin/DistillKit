#!/bin/bash
#SBATCH --account=EUHPC_D17_084 # Update this with your from saldo -b
#SBATCH --partition=boost_usr_prod # This can stay
#SBATCH --qos=boost_qos_dbg # For production write "normal" here
#SBATCH --gres=gpu:2 # GPUs PER NODE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL # Sends you an email when it starts/stop
#SBATCH --mail-user=kaansirin@yahoo.com

module load profile/deeplrn
module load python cuda nccl cudnn 

source .venv/bin/activate

# Some environment variables to set up cache directories
export TORCH_HOME=$WORK/torch-cache
export HF_HOME=$WORK/hf-cache
export GPUS_PER_NODE=2

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

# Disable internal parallelism of huggingface's tokenizer since we
# want to retain direct control of parallelism options.
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes $GPUS_PER_NODE distill_logits_final.py