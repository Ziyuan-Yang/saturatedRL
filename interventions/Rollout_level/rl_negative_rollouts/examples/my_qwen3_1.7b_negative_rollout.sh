#!/bin/bash
set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-1.7B
STORAGE_PATH="/path/to/your/storage"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    data.train_files=data/MATH-TTT/train_4303.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=negative_rollout_1.7B \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.actor.fsdp.enable_cpu_offload=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.fsdp.enable_cpu_offload=false \
    trainer.save_checkpoint_path=${STORAGE_PATH}/checkpoints_1.7B_negative_rollout \
    worker.reward.reward_function=examples/reward_function/math.py:compute_score \
    algorithm.generate_mistakes_for_all_correct=true \
    algorithm.mistake_metric_key=accuracy \
    algorithm.mistake_correct_threshold=0.999 \
    algorithm.mistake_n=2 \
    algorithm.mistake_temperature=1.2 \
    algorithm.mistake_top_p=0.95 \
    trainer.find_last_checkpoint=false \
    trainer.total_epochs=2
