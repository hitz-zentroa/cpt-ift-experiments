#!/bin/bash
#SBATCH -A EUHPC_E04_042       # account name
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00
#SBATCH --job-name=gemma2-9b-cpt
#SBATCH --cpus-per-task=16
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=.slurm/gemma2-9b-cpt.out
#SBATCH --error=.slurm/gemma2-9b-cpt.err
#SBATCH --mem=168GB


#source  $WORK/environments/axolotl/bin/activate
#source /leonardo_work/EUHPC_E04_042/latxa_accelerate/env_axolotl_5/bin/activate

module load profile/deeplrn
module load python/3.10.8--gcc--11.3.0
module load cuda/11.8
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
module load zlib/1.2.13--gcc--11.3.0
module load git-lfs

source $SCRATCH/environments/axolotl-env/bin/activate

export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export TOKENIZERS_PARALLELISM=true

export HF_HUB_OFFLINE=1 # Offline mode for Hugging Face
export WANDB_MODE=offline

#export NCCL_TIMEOUT=7200 # 2 hours, prevent crashing when loading datasets


export HF_DATASETS_CACHE=$SCRATCH"/hf_datasets_cache"
export TRANSFORMERS_CACHE=$SCRATCH"/transformers_cache/"
#export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="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"
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16

# Maximum time (in seconds) to wait for NCCL operations before timing out
#export NCCL_TIMEOUT=7200 # 2 hours, prevent crashing when loading datasets

# Enable asynchronous error handling for NCCL operations
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_TIMEOUT=7200 # 2 hours, prevent crashing when loading datasets

nvidia-smi topo -m # See the topology of the nodes

#MASTER_PORT=9327
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)



echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"

export PYTHONPATH="$PYTHONPATH:$PWD"



srun accelerate launch --mixed_precision bf16 \
                       --dynamo_backend "no" \
                       --rdzv_backend c10d \
                       --main_process_ip $MAIN_PROCESS_IP \
                       --machine_rank $SLURM_NODEID \
                       --use_fsdp \
                       --fsdp_sharding_strategy HYBRID_SHARD \
                       --fsdp_state_dict_type SHARDED_STATE_DICT \
                       --fsdp_backward_prefetch BACKWARD_PRE \
                       --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
                       --fsdp_cpu_ram_efficient_loading true \
                       -m axolotl.cli.train /leonardo/home/userexternal/amohamed/cpt-ift-experiments/src/configs/cpt_euscrawl_gemma2-9b-leonardo.yaml 
