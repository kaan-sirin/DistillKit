#!/bin/bash
#SBATCH --account=EUHPC_D17_084 
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --gres=gpu:4 # GPUs PER NODE
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
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


# python model_evaluation.py --model_path "/leonardo_work/EUHPC_D17_084/hf-cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/" --start_index 30000 --num_samples 100 --batch_size 8 --enable_logging True
# python model_evaluation.py --model_path "/leonardo_work/EUHPC_D17_084/DistillKit/distilled_models/random_sampling_sparse_logits/PubMedQA_forward_10000_samples_3_epochs_05_12_22_06" --start_index 30000 --num_samples 100 --batch_size 8 --enable_logging True