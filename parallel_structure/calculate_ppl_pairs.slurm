#!/bin/bash
#SBATCH --job-name=ppl_calc
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/ppl_calc.out
#SBATCH --error=.slurm/ppl_calc.err


source /gscratch5/users/asalem/environments/latest-lmeval/bin/activate #lmeval-env2/bin/activate


export HF_DATASETS_CACHE="/gscratch5/users/asalem/hf_datasets_cache"
export HF_TOKEN="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"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false


python3  /gscratch5/users/asalem/parallel_structure/PPl_calculator.py
