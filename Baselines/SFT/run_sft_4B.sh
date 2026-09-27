#!/bin/bash

MODEL_NAME="Qwen/Qwen3-4B"
DATA_PATH="$(dirname "$0")/my_sft.json"
OUTPUT_DIR="$(dirname "$0")/results_sft_4B"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_seq_length 4096 \
    --num_train_epochs 3.0 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.03
