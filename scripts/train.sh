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
    echo "  --method                    Training method (MAQ+DSAC|MAQ+RLPD|MAQ+IQL|SAC|RLPD|IQL)"
    echo "  -env, --environment         Environment (full name, e.g., pen-human-v1)"
    echo "  -seqlen, --sequence_length  Macro action length"
    echo "  -cbsz, --codebook_size      VQVAE codebook size"
    echo "  -s, --seed                  Seed for this experiment"
    echo " For custmized dataset, you must neither fill both training_source and testing_source nor leave them blank, otherwise the script will generate the default dataset automatically"
    echo "  -trs, --training_source     Training dataset, must put your dataset in workspace/offline_data/, if you are using D4RL Adroit, you can leave this blank, default as <env>-human-v1 dataset"
    echo "  -tes, --testing_source      Testing dataset, must put your dataset in workspace/offline_data/, if you are using D4RL Adroit, you can leave this blank, default as <env>-human-v1 dataset"
    echo "  -t, --tag                   Tag for this experiment (e.g., dates)"
    echo "  --auto_evaluate             Whether to automatically evaluate the trained model after training"
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


MAQ_VQVAE_TRAINING() {
    local method="$1"
    local environment_name="$2"
    local sequence_length="$3"
    local codebook_size="$4"
    local training_source="$5"
    local testing_source="$6"
    local suffix="$7"
    local seed="$8"
    echo "Starting training with method: $method"
    echo "Environment: $environment_name"
    echo "Sequence Length: $sequence_length"
    echo "Codebook Size: $codebook_size"
    echo "Training Source: $training_source"
    echo "Testing Source: $testing_source"
    echo "Log Suffix: $suffix"
    echo "Seed: $seed"

    gpuid=$(get_free_gpu)
    echo "Use GPU=$gpuid on machine=$(hostname)"

    cd VQVAE || exit

    CUDA_VISIBLE_DEVICES="$gpuid" python3 vqvae_train.py  --env "$environment_name" \
                        --suffix "$suffix" \
                        --seed "$seed" \
                        --seqlen "$sequence_length" \
                        --k "$codebook_size" \
                        --training_dataset "$training_source" \
                        --testing_dataset "$testing_source" 

    cd ..

}

MAQ_RL_TRAINING() {
    local method="$1"
    local environment_name="$2"
    local sequence_length="$3"
    local codebook_size="$4"
    local training_source="$5"
    local testing_source="$6"
    local suffix="$7"
    local seed="$8"
    echo "Starting training with method: $method"
    echo "Environment: $environment_name"
    echo "Sequence Length: $sequence_length"
    echo "Codebook Size: $codebook_size"
    echo "Training Source: $training_source"
    echo "Testing Source: $testing_source"
    echo "Log Suffix: $suffix"
    echo "Seed: $seed"


    gpuid=$(get_free_gpu)
    echo "Use GPU=$gpuid on machine=$(hostname)"
    if [ "$method" == "MAQ+RLPD" ]; then
        
        cd RLPD_MAQ || exit
        CUDA_VISIBLE_DEVICES="$gpuid" python3 main.py --env "$environment_name" \
                        --suffix "$suffix" \
                        --seed "$seed" \
                        --seqlen "$sequence_length" \
                        --training_dataset "$training_source" \
                        --testing_dataset "$testing_source" \
                        --k "$codebook_size" \
                        --type "RLPD"

        cd ..

    elif [ "$method" == "MAQ+DSAC" ]; then

        cd RLPD_MAQ || exit

        CUDA_VISIBLE_DEVICES="$gpuid" python3 main.py --env "$environment_name" \
                        --suffix "$suffix" \
                        --seed "$seed" \
                        --seqlen "$sequence_length" \
                        --training_dataset "$training_source" \
                        --testing_dataset "$testing_source" \
                        --k "$codebook_size" \
                        --type "DSAC"

        cd ..

    elif [ "$method" == "MAQ+IQL" ]; then
        # train prior first 

        cd VQVAE || exit

        CUDA_VISIBLE_DEVICES="$gpuid" python3 prior_train.py  --env "$environment_name" \
                        --suffix "$suffix" \
                        --seed "$seed" \
                        --seqlen "$sequence_length" \
                        --k "$codebook_size" \
                        --training_dataset "$training_source" \
                        --testing_dataset "$testing_source" 

        cd ..

        cd IQL || exit
            python3 iql_MAQ_trainer.py --env "$environment_name" \
                        --seed "$seed" \
                        --seqlen "$sequence_length" \
                        --k "$codebook_size" \
                        --gpuid "$gpuid" \
                        --training_dataset "$training_source" \
                        --testing_dataset "$testing_source" \
                        --suffix "$suffix"
        cd ..
    else
        echo "Unsupported method: $method"
        exit 1
    fi

    echo "=================================================="
    echo "Done for env=$environment_name, seed=$seed, seqlen=$sequence_length, k=$codebook_size, method=$method"
    echo "=================================================="
}




training_method="MAQ+RLPD"
environement_name="door-human-v1"
seed="1"
sequence_length="9"
codebook_size="16"
training_source=""
testing_source=""
tag=""
auto_evaluate=false

while :; do
    case $1 in
        -h|--help) shift; usage ;;
        --method) shift; training_method="${1}" ;;
        -s|--seed) shift; seed="${1}" ;;
        -env|--environment) shift; environement_name="${1}" ;;
        -seqlen|--sequence_length) shift; sequence_length="${1}" ;;
        -cbsz|--codebook_size) shift; codebook_size="${1}" ;;
        -trs|--training_source) shift; training_source="${1}" ;;
        -tes|--testing_source) shift; testing_source="${1}" ;;
        -t|--tag) shift; tag="${1}" ;;
        --auto_evaluate) auto_evaluate=true ;;
        "") break ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
    shift
done

if [ "$training_method" == "MAQ+DSAC" ] || [ "$training_method" == "MAQ+RLPD" ] || [ "$training_method" == "MAQ+IQL" ]; then

    suffix="seed$seed"_"sq$sequence_length"_"k$codebook_size"
    if [ "$tag" != "" ]; then
        suffix="${suffix}_$tag"
    fi

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

    cd offline_data || exit
    
    rm -f error.txt
    
    echo "Checking dataset: $training_source and $testing_source"
    if [ "$default_generate" = true ]; then
        python gen_offline_data.py --env $environement_name --seed $seed --training_dataset_name $training_source --testing_dataset_name $testing_source --train_ratio 0.9 --default_generate 
    else
        python gen_offline_data.py --env $environement_name --seed $seed --training_dataset_name $training_source --testing_dataset_name $testing_source --train_ratio 0.9 
    fi
    

    if [ -f "error.txt" ]; then
        echo "Error: $(cat error.txt)"
        exit 1
    fi
    cd .. || exit

    MAQ_VQVAE_TRAINING "$training_method" "$environement_name" "$sequence_length" "$codebook_size" "$training_source" "$testing_source" "$suffix" "$seed"
    MAQ_RL_TRAINING "$training_method" "$environement_name" "$sequence_length" "$codebook_size" "$training_source" "$testing_source" "$suffix" "$seed"

elif [ "$training_method" == "SAC" ]; then
    
    cd SAC || exit
    python3 sac_train.py --env $environement_name \
                            --seed $seed
    cd ..

elif [ "$training_method" == "RLPD" ]; then
    echo "Please follow the instruction in rlpd/MAQ_version_README.md to train RLPD without MAQ"
    echo "Go to rlpd folder and run "
    echo """
    XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py --env_name=<your env> \\
    --utd_ratio=20 \\
    --start_training 10000 \\
    --max_steps 1000000 \\
    --config=configs/rlpd_config.py \\
    --project_name=rlpd_locomotion \\
    --seed=<your see> \\
    --checkpoint_model=true \\
    --log_dir=/workspace/rlpd/<your log> \\
    --training_dataset=<your training dataset stored in the ./offline_data>
    """

elif [ "$training_method" == "IQL" ]; then

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

    cd offline_data || exit
    
    rm -f error.txt
    
    echo "Checking dataset: $training_source and $testing_source"
    if [ "$default_generate" = true ]; then
        python gen_offline_data.py --env $environement_name --seed $seed --training_dataset_name $training_source --testing_dataset_name $testing_source --train_ratio 0.9 --default_generate 
    else
        python gen_offline_data.py --env $environement_name --seed $seed --training_dataset_name $training_source --testing_dataset_name $testing_source --train_ratio 0.9 
    fi

    cd ..


    cd IQL || exit
    python3 iql.py --env $environement_name \
                        --seed $seed \
                        --training_dataset "$training_source" \
                        --testing_dataset "$testing_source" 
    cd ..

else
    echo "Training method: $training_method"
    exit 1
fi

if [ "$auto_evaluate" = true ]; then
    echo "Starting automatic evaluation..."
    ./scripts/evaluate.sh \
        --method "$training_method" \
        -env "$environement_name" \
        -seqlen "$sequence_length" \
        -cbsz "$codebook_size" \
        -s "$seed" \
        --tag "$tag" \
        -trs "$training_source" \
        -tes "$testing_source"
fi