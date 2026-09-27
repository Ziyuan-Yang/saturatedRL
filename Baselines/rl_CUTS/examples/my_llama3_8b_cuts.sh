#!/bin/bash
set -x
# export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

MODEL_PATH=meta-llama/Meta-Llama-3-8B # replace it with your local file path
STORAGE_PATH="/path/to/your/storage"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    data.train_files=data/MATH-TTT/train_4303.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=2 \
    worker.actor.fsdp.enable_cpu_offload=false \
    trainer.n_gpus_per_node=2 \
    trainer.experiment_name=1-baseline-CUTS-Llama3-8B \
    trainer.save_checkpoint_path=${STORAGE_PATH}/checkpoints_Llama3_8B_CUTS \
    worker.reward.reward_function=examples/reward_function/math.py:compute_score \
    trainer.find_last_checkpoint=false \
    trainer.total_epochs=2 \
    data.override_chat_template=examples/chat_template/llama3.jinja \
