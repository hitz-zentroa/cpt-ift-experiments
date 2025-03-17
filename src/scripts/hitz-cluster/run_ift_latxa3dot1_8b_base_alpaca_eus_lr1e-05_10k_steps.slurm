#!/bin/bash
#SBATCH --job-name=ift-alpaca-eus
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/sft_latxa3.1_alpaca_eus_lr1e-05_10ksteps.out
#SBATCH --error=.slurm/sft_latxa3.1_alpaca_eus_lr1e-05_10ksteps.err


source /gscratch5/users/asalem/environments/axoltl-again/bin/activate

export HF_DATASETS_CACHE="/gscratch5/users/asalem/cache/hf_datasets_cache"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="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"
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"

export PYTHONPATH="$PYTHONPATH:$PWD"

cd /gscratch5/users/asalem/cpt-ift


accelerate launch --use_fsdp \
        --fsdp_sharding_strategy HYBRID_SHARD \
        --fsdp_state_dict_type SHARDED_STATE_DICT \
        --fsdp_backward_prefetch BACKWARD_PRE \
        --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
        --fsdp_cpu_ram_efficient_loading true \
        -m axolotl.cli.train src/configs/sft_alpaca_latxa3.1_base_lr1e-05_10ksteps.yaml