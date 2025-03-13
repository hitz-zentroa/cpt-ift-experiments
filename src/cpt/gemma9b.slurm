#!/bin/bash
#SBATCH --job-name=cpt-gm9b
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:3
#SBATCH --output=.slurm/cpt-gemma2-9b.out
#SBATCH --error=.slurm/cpt-gemma2-9b.err


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

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --main_process_port=29501 -m axolotl.cli.train src/configs/cpt_euscrawl_gemma2-9b.yaml 