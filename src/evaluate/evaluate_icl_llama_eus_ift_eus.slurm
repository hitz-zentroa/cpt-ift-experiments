#!/bin/bash
#SBATCH --job-name=icl-eus-bench-lmeval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/icl-eus-bench-lmeval.out
#SBATCH --error=.slurm/icl-eus-bench-lmeval.err


source /gscratch5/users/asalem/environments/latest-lmeval/bin/activate 


export HF_TOKEN="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"
export HF_DATASETS_CACHE="/gscratch5/users/asalem/cache/hf_datasets_cache"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false


models=(
    #"ift-llama3dot1-baseline-alpaca_eus-lr1e05-10ksteps/checkpoint-15"
    #"meta-llama/Llama-3.1-8B"
    #"ift-llama-eus-alpaca-lr1e05-10ksteps/checkpoint-26"
    #"ahmedselhady/llama2-7B-eus-v1"
    #"ift-llama-eus-alpaca-lr1e04"
    #"ift-llama-eus-alpaca-lr1e05"
    #"google/gemma-2-9b"
    "ift-gemma2-9b_alpaca_eus_lr1e-05-10ksteps/checkpoint-14"
)
benchmarks=(
    "paws_eu"
    "piqa_eu"
    "belebele_eus_Latn"
    "eus_trivia"
    "eus_reading"
    "eus_proficiency"
    "eus_exams_eu"
    "xstorycloze_eu"
    "xnli_eu"
    #"copain"
)

results_path=/gscratch5/users/asalem/cpt-ift/results/icl_benchmarks



for model in "${models[@]}"; do

    if [[ $model == "ahmedselhady/"* || $model == "google/"* ]]; then
        model_name=$model
    else
        model_name='/gscratch5/users/asalem/cpt-ift/model/'$model
    fi

    for benchmark_name in "${benchmarks[@]}"; do


        if [[ $benchmark_name == "xnli_eu" || $benchmark_name == "xstorycloze_eu" ]]; then
            num_fewshot=0
        else
            num_fewshot=5
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
