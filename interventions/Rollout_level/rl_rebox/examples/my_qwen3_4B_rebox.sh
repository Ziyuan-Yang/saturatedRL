#!/bin/bash
set -x
# export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-4B # replace it with your local file path
STORAGE_PATH="/path/to/your/storage"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    data.train_files=data/MATH-TTT/train_4303.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=rewrite-boxanswer-4B \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.fsdp.enable_cpu_offload=false \
    trainer.save_checkpoint_path=${STORAGE_PATH}/checkpoints_4B_rewrite_boxanswer \
    worker.reward.reward_function=examples/reward_function/math.py:compute_score \
    trainer.find_last_checkpoint=false \
    trainer.total_epochs=2 \
    algorithm.perturb_all_correct_answers=true \
    algorithm.perturb_n=1 \

