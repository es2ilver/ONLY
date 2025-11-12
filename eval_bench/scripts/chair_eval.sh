#!/bin/bash

seeds=(4 6 8 10)

# llava
model="llava"
model_path="/home/data/vgilab/jeongeun/checkpoints/llava-v1.5-7b"



coco_path="/home/data/vgilab/jeongeun/datasets/coco"
img_path="${coco_path}/val2014/"
anno_path="${coco_path}/annotations/instances_val2014.json"
log_path="./logs/chair"
out_path="./chair_results/${model}"

use_ritual=False
use_vcd=False
use_m3id=False
use_only=True
method_name="only"


ritual_alpha_pos=3.0
ritual_alpha_neg=1.0
ritual_beta=0.1
js_gamma=0.25
enhance_layer_index=0

num_eval_samples=500
max_new_tokens=64

#####################################
# Run experiment
#####################################
export CUDA_VISIBLE_DEVICES=0

for seed in "${seeds[@]}"; do
    echo "=========================================="
    echo "Running CHAIR evaluation with seed=${seed}, ONLY=${use_only}"
    echo "=========================================="
    
    # Run evaluation
    python eval_bench/chair_eval_${model}.py \
        --seed ${seed} \
        --model_path ${model_path} \
        --model_base ${model} \
        --llama_model_path ${llama_model_path} \
        --data_path ${img_path} \
        --anno_path ${anno_path} \
        --log_path ${log_path} \
        --out_path ${out_path} \
        --use_ritual ${use_ritual} \
        --use_vcd ${use_vcd} \
        --use_m3id ${use_m3id} \
        --use_only ${use_only} \
        --method_name ${method_name} \
        --ritual_alpha_pos ${ritual_alpha_pos} \
        --ritual_alpha_neg ${ritual_alpha_neg} \
        --ritual_beta ${ritual_beta} \
        --js_gamma ${js_gamma} \
        --num_eval_samples ${num_eval_samples} \
        --max_new_tokens ${max_new_tokens} \
        --enhance_layer_index ${enhance_layer_index}
    
    # Construct result file path
    cap_json_path="${out_path}/${ritual_alpha_pos}_${ritual_alpha_neg}_${ritual_beta}_${js_gamma}_${max_new_tokens}_${method_name}.jsonl"
    result_json_path="${out_path}/${ritual_alpha_pos}_${ritual_alpha_neg}_${ritual_beta}_${js_gamma}_${max_new_tokens}_${method_name}_result.jsonl"
    
    # Run CHAIR evaluation
    if [ -f "${cap_json_path}" ]; then
        echo "Running CHAIR metrics calculation..."
        python eval_bench/chair.py \
            --cap_file ${cap_json_path} \
            --coco_path ${coco_path}/annotations \
            --save_path ${result_json_path} \
            --image_id_key image_id \
            --caption_key caption > /tmp/chair_output_${seed}.txt 2>&1
        
        # Extract metrics from output
        if [ -f "/tmp/chair_output_${seed}.txt" ]; then
            chair_s=$(grep -E "^CHAIRs" /tmp/chair_output_${seed}.txt | awk -F': ' '{print $2}')
            chair_i=$(grep -E "^CHAIRi" /tmp/chair_output_${seed}.txt | awk -F': ' '{print $2}')
            recall=$(grep -E "^Recall" /tmp/chair_output_${seed}.txt | awk -F': ' '{print $2}')
            len=$(grep -E "^Len" /tmp/chair_output_${seed}.txt | awk -F': ' '{print $2}')
            
            if [ -n "$chair_s" ] && [ -n "$chair_i" ]; then
                result_line="CHAIRs: ${chair_s}, CHAIRi: ${chair_i}, Recall: ${recall}, Len: ${len}"
                results+=("Seed ${seed} (ONLY=${use_only}): ${result_line}")
                echo "Seed ${seed} completed: ${result_line}"
            else
                results+=("Seed ${seed} (ONLY=${use_only}): Could not extract metrics")
                echo "Warning: Could not extract metrics for seed ${seed}"
            fi
        else
            results+=("Seed ${seed} (ONLY=${use_only}): CHAIR evaluation output not found")
            echo "Warning: CHAIR evaluation output not found for seed ${seed}"
        fi
    else
        results+=("Seed ${seed} (ONLY=${use_only}): Caption file not found at ${cap_json_path}")
        echo "Warning: Caption file not found at ${cap_json_path}"
    fi
    
    echo ""
done

#####################################
# Print summary results
#####################################
echo "=========================================="
echo "SUMMARY RESULTS - MiniGPT4"
echo "=========================================="
echo "Method: ${method_name} (ONLY=${use_only})"
echo "Parameters: alpha_pos=${ritual_alpha_pos}, alpha_neg=${ritual_alpha_neg}, beta=${ritual_beta}, gamma=${js_gamma}, layer=${enhance_layer_index}"
echo "=========================================="
for result in "${results[@]}"; do
    echo "$result"
done
echo "=========================================="