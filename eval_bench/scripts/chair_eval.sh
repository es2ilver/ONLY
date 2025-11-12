#!/bin/bash

seed=4

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
max_new_tokens=128

#####################################
# Run experiment
#####################################
export CUDA_VISIBLE_DEVICES=0
python eval_bench/chair_eval_${model}.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
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
--enhance_layer_index ${enhance_layer_index} \

#####################################
# Run evaluation
#####################################
cap_json_path="${out_path}/${ritual_alpha_pos}_${ritual_alpha_neg}_${ritual_beta}_${js_gamma}_${max_new_tokens}_${method_name}.jsonl"
# cap_json_path="${out_path}/exp_003.jsonl"
echo ${cap_json_path}
python eval_bench/chair.py \
--cap_file ${cap_json_path} \
--coco_path ${coco_path}/annotations \
--save_path ${out_path}/${ritual_alpha_pos}_${ritual_alpha_neg}_${ritual_beta}_${js_gamma}_${max_new_tokens}_${method_name}_result.jsonl \
--image_id_key image_id \
--caption_key caption