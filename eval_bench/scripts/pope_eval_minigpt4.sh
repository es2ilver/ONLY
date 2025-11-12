#!/bin/bash

seed=4 # Llava 4, 6, 8, # Minigpt 4, 8, 10 # mplu 4, 10, 13

dataset_name="coco" # coco | aokvqa | gqa
type="adversarial" # random | popular | adversarial

# minigpt
model="minigpt"
model_path="/home/data/vgilab/jeongeun/checkpoints/pretrained_minigpt4_llama2_7b.pth"
llama_model_path="/home/data/vgilab/jeongeun/checkpoints/llama-2-7b-chat-hf"

pope_path="/home/data/vgilab/jeongeun/datasets/POPE/${dataset_name}/${dataset_name}_pope_${type}.json"
data_path="/home/data/vgilab/jeongeun/datasets/coco/val2014"

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


#####################################
# Run single experiment
#####################################
export CUDA_VISIBLE_DEVICES=1
python eval_bench/pope_eval_${model}.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
--llama_model_path ${llama_model_path} \
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

