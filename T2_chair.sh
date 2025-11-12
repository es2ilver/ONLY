#!/bin/bash

set -e

bash eval_bench/scripts/chair_eval.sh

echo "chair_eval.sh (chair version ONLY) completed"
echo "=========================================="

# bash eval_bench/scripts/chair_eval_minigpt4.sh

# echo "chair_eval_minigpt4.sh (chair version MINIGPT4) completed"
# echo "=========================================="

bash eval_bench/scripts/chair_eval_mplug.sh

echo "chair_eval_mplug.sh (chair version MPLUG) completed"
echo "=========================================="