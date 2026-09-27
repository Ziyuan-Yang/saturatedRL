#!/bin/bash
set -xru
# export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0

MODEL_PATH=Qwen/Qwen3-4B # replace it with your local file path
STORAGE_PATH="/path/to/your/storage"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    data.train_files=data/MATH-TTT/train_4303.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.cuts_enabled=True \
    trainer.experiment_name=1-baseline-CUTS-4B \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.actor.fsdp.enable_cpu_offload=false \
    trainer.save_checkpoint_path=${STORAGE_PATH}/checkpoints_4B_CUTS \
    worker.reward.reward_function=examples/reward_function/math.py:compute_score \
    trainer.find_last_checkpoint=false \
    trainer.total_epochs=2 \

