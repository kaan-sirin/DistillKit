#!/usr/bin/env bash
# SLURM: 2 nodes, 4 GPUs each, 1 task per node
srun -N 2 -n 2 \
     --gpus-per-node=4 \
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
    
    
    # ---- distributed init ----
    MASTER_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
    export MASTER_ADDR=$MASTER_NODE
    export MASTER_PORT=29500
    export WORLD_SIZE=$SLURM_NTASKS
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=0           # one process per node

    accelerate launch \
        --num_processes 1 \
        --num_machines 2 \
        --machine_rank $RANK \
        --cpu_threads_per_process 8 \
        distill_logits_final.py
'