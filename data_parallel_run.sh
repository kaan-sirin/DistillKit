#!/bin/bash
#SBATCH --account=EUHPC_D17_084
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:4       
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4               
#SBATCH --cpus-per-task=32
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kaansirin@yahoo.com

module load profile/deeplrn
module load python cuda nccl cudnn
source .venv/bin/activate

export TORCH_HOME=$WORK/torch-cache
export HF_HOME=$WORK/hf-cache
export NUM_NODES=4
export GPUS_PER_NODE=4

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

# Rendezvous for multi-node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

srun \
  accelerate launch \
    --num_processes $GPUS_PER_NODE \
    --num_machines  $NUM_NODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --mixed_precision bf16 \
    distillation_new_attempt.py