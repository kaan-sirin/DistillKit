#!/usr/bin/env bash
# SLURM: 2 nodes, 4 GPUs each, 1 task per node
srun -N 2 -n 2 \
    --gpus-per-node=4 \
    --gres=gpu:4 \
    --cpus-per-task=32 \
    -t 00:30:00 \
    -A EUHPC_D17_084 \
    -p boost_usr_prod \
    --qos boost_qos_dbg \
    bash -c '
        module load profile/deeplrn
        module load python cuda nccl cudnn
        source .venv/bin/activate

        export TORCH_HOME=$WORK/torch-cache
        export HF_HOME=$WORK/hf-cache
        export TOKENIZERS_PARALLELISM=false

        export HF_HUB_OFFLINE=1
        export HF_DATASETS_OFFLINE=1
        export WANDB_MODE=offline
        

        # ---- distributed init ----
        export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
        export MASTER_PORT=29500
        
        accelerate launch \
            --num_processes 1 \
            --num_machines 2 \
            --machine_rank $SLURM_NODEID \
            --main_process_ip $MASTER_ADDR \
            --main_process_port $MASTER_PORT \
            distill_logits_final.py
        '
