#!/bin/bash
#SBATCH --account=EUHPC_D17_084
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kaansirin@yahoo.com

module load profile/deeplrn
module load python cuda nccl cudnn 
source .venv/bin/activate

export TORCH_HOME=$WORK/torch-cache
export HF_HOME=$WORK/hf-cache
export GPUS_PER_NODE=4
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --num_processes $GPUS_PER_NODE distillation_new_attempt.py