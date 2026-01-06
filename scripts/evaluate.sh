#!/usr/bin/env bash

# can train MAQ+DSAC/RLPD/IQL
# with parameter 
# - sequence length 
# - codebook size
# - dataset_source (training and testing must both included) 
# - environment
# - seed

#!/bin/bash
set -e

usage() {
    echo "Usage: $0 [OPTION]..."
    echo ""
    echo "Optional arguments:"
    echo "  -h, --help                  Give this help list"
    echo "  --method                    Evaluating method (MAQ+DSAC|MAQ+RLPD|MAQ+IQL|SAC|RLPD|IQL)"
    echo "  -env, --environment         Environment (full name, e.g., pen-human-v1)"
    echo "  -seqlen, --sequence_length  Macro action length"
    echo "  -cbsz, --codebook_size      VQVAE codebook size"
    echo "  -s, --seed                  Seed for this experiment"
    echo " For custmized dataset, you must neither fill both training_source and testing_source nor leave them blank, otherwise the script will generate the default dataset automatically, make sure that both dataset are same as the training (train.sh) you are using."
    echo "  -trs, --training_source     Training dataset, must put your dataset in workspace/offline_data/, if you are using D4RL Adroit, you can leave this blank, default as <env>-human-v1 dataset"
    echo "  -tes, --testing_source      Testing dataset, must put your dataset in workspace/offline_data/, if you are using D4RL Adroit, you can leave this blank, default as <env>-human-v1 dataset"
    echo "  -t, --tag                   Tag for this experiment (e.g., dates)"
    echo " You can also directly choose one method path with correct 'method', environment, and dataset without setting seqlen, cbsz; this is for SAC, RLPD, and IQL methods."
    echo "  -path, --model_path          The path of the model to be evaluated"
    exit 1
}


############################################################
# 1. Function: Get the least used GPU
############################################################
get_free_gpu() {
  local best_gpu=0
  local best_usage=999999
  for i in 0 1 2 3; do
    local util_mem
    util_mem=$(nvidia-smi \
      --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits -i "$i" 2>/dev/null)
    if [ -z "$util_mem" ]; then
        continue
    fi
    local util=$(echo "$util_mem" | awk -F, '{print $1}' | xargs)
    local mem=$(echo "$util_mem"  | awk -F, '{print $2}' | xargs)
    local usage=$((util + mem))
    if [ "$usage" -lt "$best_usage" ]; then
      best_gpu=$i
      best_usage=$usage
    fi
  done
  echo "$best_gpu"
}

method="MAQ+RLPD"
environement_name="door-human-v1"
seed="1"
sequence_length="9"
codebook_size="16"
training_source=""
testing_source=""
tag=""
model_path=""


while :; do
    case $1 in
        -h|--help) shift; usage ;;
        --method) shift; method="${1}" ;;
        -s|--seed) shift; seed="${1}" ;;
        -env|--environment) shift; environement_name="${1}" ;;
        -seqlen|--sequence_length) shift; sequence_length="${1}" ;;
        -cbsz|--codebook_size) shift; codebook_size="${1}" ;;
        -trs|--training_source) shift; training_source="${1}" ;;
        -tes|--testing_source) shift; testing_source="${1}" ;;
        -t|--tag) shift; tag="${1}" ;;
        -path|--model_path) shift; model_path="${1}" ;;
        "") break ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done


default_generate=false

if [ -f "offline_data/$training_source" ] && [ -f "offline_data/$testing_source" ]; then
    echo "Using the custmized dataset: $training_source and $testing_source"
    training_and_testing_source_exist=false
else
    echo ""
    training_source="$environement_name""_train_seed$seed""_ratio0.9.pkl"
    testing_source="$environement_name""_test_seed$seed""_ratio0.9.pkl"
    echo "Training source: $training_source"
    echo "Testing source: $testing_source"  
    default_generate=true
fi

if [ ! -f "offline_data/$training_source" ] || [ ! -f "offline_data/$testing_source" ]; then
  echo "Dataset does not exist in offline_data/"
  exit 1
fi

cd offline_data || exit

rm -f error.txt

echo "Checking dataset: $training_source and $testing_source"
python gen_offline_data.py --env $environement_name --seed $seed --training_dataset_name $training_source --testing_dataset_name $testing_source --train_ratio 0.9 

if [ -f "error.txt" ]; then
    echo "Error: $(cat error.txt)"
    exit 1
fi

cd ..

suffix="seed$seed"_"sq$sequence_length"_"k$codebook_size"


if [ "$model_path" == "" ]; then
    python3 human_similarity_calculator.py \
    --env $environement_name \
    --eval_agent $method \
    --suffix $suffix \
    --seed $seed \
    --training_dataset $training_source \
    --testing_dataset $testing_source 
else
    python3 human_similarity_calculator.py \
    --env $environement_name \
    --eval_agent $method \
    --model_path $model_path \
    --seed $seed \
    --training_dataset $training_source \
    --testing_dataset $testing_source
fi

