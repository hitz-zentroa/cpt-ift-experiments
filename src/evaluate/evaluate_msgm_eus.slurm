#!/bin/bash
#SBATCH --job-name=msgm
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/icl-eus-bench-lmeval.out
#SBATCH --error=.slurm/icl-eus-bench-lmeval.err


source /gscratch5/users/asalem/environments/lmevalharness/bin/activate 


export HF_TOKEN="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"
export HF_DATASETS_CACHE="/gscratch5/users/asalem/cache/hf_datasets_cache"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false


models=(

    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "HiTZ/latxa-13b-v1.1"
    #"ahmedselhady/llama2-7B-eus-v1"
    #"ahmedselhady/Llama-2-7B-eu-ema"
    #"HiTZ/latxa-7b-v1.1"
)
benchmarks=(
    #"mgsm_direct_eu"
    "mgsm_native_cot_eu"
)

results_path=/gscratch5/users/asalem/cpt-ift/results/msgm

mkdir -p ${results_path}

for model in "${models[@]}"; do

    if [[ $model == "ahmedselhady/"* || $model == "google/"* ]]; then
        model_name=$model
    else
        model_name=$model
        #'/gscratch5/users/asalem/taskvectors/'$model
    fi

    for benchmark_name in "${benchmarks[@]}"; do


        if [[ $benchmark_name == "xnli_eu" || $benchmark_name == "xstorycloze_eu" ]]; then
            num_fewshot=0
        else
            num_fewshot=8
        fi

        python3 -m lm_eval \
            --model hf \
            --model_args pretrained=$model_name,parallelize=True \
            --tasks $benchmark_name \
            --device cuda \
            --output_path ${results_path}/${model}/${benchmark_name}/${num_fewshot}-shot.json \
            --num_fewshot ${num_fewshot} \
            --log_samples
    done
done
