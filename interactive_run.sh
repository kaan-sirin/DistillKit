#!/usr/bin/env bash
srun -n 1 \
    -t 00:30:00 \
    -A EUHPC_D17_084 \
    -p boost_usr_prod \
    --qos boost_qos_dbg \
    --gres=gpu:2 \
    --cpus-per-task=16 \
    bash -c '
       module load profile/deeplrn
       module load python cuda nccl cudnn
       source .venv/bin/activate

       export TORCH_HOME=$WORK/torch-cache
       export HF_HOME=$WORK/hf-cache
       export GPUS_PER_NODE=2
       export HF_HUB_OFFLINE=1
       export HF_DATASETS_OFFLINE=1
       export WANDB_MODE=offline
       export TOKENIZERS_PARALLELISM=false

       accelerate launch --num_processes $GPUS_PER_NODE distill_logits_final.py
     '
