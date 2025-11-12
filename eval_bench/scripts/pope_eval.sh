#!/bin/bash

seeds=(4 6 8 10) # Llava 4, 6, 8, # Minigpt 4, 8, 10 # mplu 4, 10, 13

dataset_name="coco" # coco | aokvqa | gqa
type="random" # random | popular | adversarial

# llava
model="llava"
model_path="/home/data/vgilab/jeongeun/checkpoints/llava-v1.5-7b"

# instructblip
# model="instructblip"
# model_path=None

# qwen-vl
# model="qwen-vl"
# model_path="/data/zifu/model/Qwen-VL-Chat"

# minigpt
# model="minigpt"
# model_path="/home/data/vgilab/jeongeun/checkpoints/pretrained_minigpt4_llama2_7b.pth"

# mplu 
# model="mplu"
# model_path="/home/data/vgilab/jeongeun/checkpoints/mplug-owl2-llama2-7b"

pope_path="/home/data/vgilab/jeongeun/datasets/POPE/${dataset_name}/${dataset_name}_pope_${type}.json"
data_path="/home/data/vgilab/jeongeun/datasets/coco/val2014"

# data_path="/data/ce/data/gqa/images"

log_path="./logs"

use_ritual=False
use_vcd=False
use_m3id=False
use_only=True

ritual_alpha_pos=3.0
ritual_alpha_neg=1.0
ritual_beta=0.1
js_gamma=0.2
enhance_layer_index=0

temperature=1.0  # do_sample=False

#####################################
# Run single experiment
#####################################
export CUDA_VISIBLE_DEVICES=1

# Calculate experiment directory path (same as in pope_eval_mplug.py)
model_string_name=$(basename ${model_path})
if [ "${use_only}" = "True" ]; then
    method_name="ONLY"
elif [ "${use_ritual}" = "True" ]; then
    method_name="RITUAL"
elif [ "${use_vcd}" = "True" ]; then
    method_name="VCD"
elif [ "${use_m3id}" = "True" ]; then
    method_name="M3ID"
else
    method_name="Regular"
fi

# Array to store results
declare -a results

for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo "Running evaluation with seed=${seed}"
    echo "=========================================="
    
    # Run evaluation
    python eval_bench/pope_eval_${model}.py \
        --seed ${seed} \
        --model_path ${model_path} \
        --model_base ${model} \
        --pope_path ${pope_path} \
        --data_path ${data_path} \
        --log_path ${log_path} \
        --use_ritual ${use_ritual} \
        --use_vcd ${use_vcd} \
        --use_m3id ${use_m3id} \
        --use_only ${use_only} \
        --ritual_alpha_pos ${ritual_alpha_pos} \
        --ritual_alpha_neg ${ritual_alpha_neg} \
        --ritual_beta ${ritual_beta} \
        --js_gamma ${js_gamma} \
        --type ${type} \
        --dataset_name ${dataset_name} \
        --enhance_layer_index ${enhance_layer_index} \
        --temperature ${temperature}

    # Construct log file path (same as in pope_eval_mplug.py)
    log_file="${log_path}/pope/${model_string_name}/${method_name}_${dataset_name}_${type}_${ritual_alpha_pos}_${ritual_alpha_neg}_${ritual_beta}_${js_gamma}_layer_${enhance_layer_index}_seed_${seed}/log.txt"
    
    # Extract the final result line from the log file
    if [ -f "${log_file}" ]; then
        result_line=$(grep -E "acc: [0-9]+\.[0-9]+, precision: [0-9]+\.[0-9]+, recall: [0-9]+\.[0-9]+, f1: [0-9]+\.[0-9]+, yes_ratio: [0-9]+\.[0-9]+" "${log_file}" | tail -1)
        
        if [ -n "$result_line" ]; then
            results+=("Seed ${seed}: ${result_line}")
            echo "Seed ${seed} completed: ${result_line}"
        else
            results+=("Seed ${seed}: Result not found in log file")
            echo "Warning: Could not extract result for seed ${seed} from ${log_file}"
        fi
    else
        results+=("Seed ${seed}: Log file not found at ${log_file}")
        echo "Warning: Log file not found at ${log_file}"
    fi
    
    echo ""
done

#####################################
# Print summary results
#####################################
echo "=========================================="
echo "SUMMARY RESULTS"
echo "=========================================="
for result in "${results[@]}"; do
    echo "$result"
done
echo "=========================================="

