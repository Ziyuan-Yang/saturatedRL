#!/bin/bash
set -x
# export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-1.7B # replace it with your local file path
STORAGE_PATH="/path/to/your/storage"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    data.train_files=data/MATH-TTT/train_4303.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=prefixwrong-1.7B \
    trainer.save_checkpoint_path=${STORAGE_PATH}/checkpoints_1.7B_prefixwrong \
    worker.reward.reward_function=examples/reward_function/math.py:compute_score \
    trainer.find_last_checkpoint=false \
    trainer.total_epochs=2 \
    algorithm.continue_wrong_for_all_correct=true \
    algorithm.continue_wrong_n=2 \

