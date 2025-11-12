#!/bin/bash

set -e

bash eval_bench/scripts/pope_eval.sh

echo "pope_eval.sh (vanilla llava version ONLY) completed"
echo "=========================================="

# bash eval_bench/scripts/pope_eval_minigpt4.sh

# echo "pope_eval_minigpt4.sh (minigpt4 version ONLY) completed"
# echo "=========================================="

bash eval_bench/scripts/pope_eval_mplug.sh

echo "pope_eval_mplug.sh (mplug version ONLY) completed"
echo "=========================================="