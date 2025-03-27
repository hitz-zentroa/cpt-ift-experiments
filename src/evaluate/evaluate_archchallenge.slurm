#!/bin/bash
#SBATCH --job-name=arc_eu_challenge
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:2
#SBATCH --output=.slurm/arc_eu_challenge.out
#SBATCH --error=.slurm/arc_eu_challenge.err


source /gscratch5/users/asalem/environments/latest-lmeval/bin/activate 


export HF_TOKEN="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"
export HF_DATASETS_CACHE="/gscratch5/users/asalem/cache/hf_datasets_cache"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false


models=(
    #"ift-llama3dot1-baseline-alpaca_eus-lr1e05-10ksteps/checkpoint-15"
    #"meta-llama/Llama-3.1-8B"
    #"ift-llama-eus-alpaca-lr1e05-10ksteps/checkpoint-26"
    #"ift-llama-eus-alpaca-lr1e04"
    #"ift-llama-eus-alpaca-lr1e05"
    #"meta-llama/Llama-3.1-8B"
    #"google/gemma-2-9b"
    "Latxa3.1_8b_lr1e-5"
    #"ift-gemma2-9b_alpaca_eus_lr1e-05-10ksteps/checkpoint-14"
    #"ahmedselhady/llama2-7B-eus-v1"
    #"meta-llama/Llama-2-7b-hf"
    #"ift-latxa12-alpaca_eus-lr1e05-10ksteps/checkpoint-26"
    #"ift-llama-eus-alpaca-lr1e05-10ksteps/checkpoint-26"
)
benchmarks=(
#"arc_challenge"
    "arc_eu_challenge"

)

results_path=/gscratch5/users/asalem/cpt-ift/results/icl_benchmarks



for model in "${models[@]}"; do
    
    #model_name=$model
    model_name='/proiektuak/ikergaitu-data/azabala106/model_evaluation/trained_models/'$model
    #if [[ $model == "ahmedselhady/"* || $model == "google/"* || $model == "meta/"* ]]; then
    #    model_name=$model
    #else
    #    model_name='/proiektuak/ikergaitu-data/azabala106/model_evaluation/trained_models/'$model
        #model_name='/gscratch5/users/asalem/cpt-ift/model/'$model
    #fi

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
