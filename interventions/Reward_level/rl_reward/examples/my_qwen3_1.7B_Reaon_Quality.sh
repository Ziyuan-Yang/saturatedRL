#!/bin/bash
#SBATCH --job-name=grpo6
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=sft_%j.out
#SBATCH --error=sft_%j.err

# ===== 环境准备 =====
module load conda                    
conda activate rl
cd "/path/to/your/storage"
set -x
export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen3-1.7B # replace it with your local file path
STORAGE_PATH="/path/to/your/storage"
export VLLM_CACHE_ROOT="/path/to/your/vllm_cache"
VLLM_PORT=7781

# start vllm judge in GPU 0
CUDA_VISIBLE_DEVICES=0 python3 utils/vllm_service4.py \
    --model Qwen/Qwen3-8B \
    --port ${VLLM_PORT} \
    --tensor_parallel_size 1 &
VLLM_PID=$!

# wait vllm ready
echo "Waiting for vllm service on port ${VLLM_PORT}..."
until curl -sf http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; do
    sleep 5
done
echo "vllm service is ready."

# kill vllm when exit
trap "kill ${VLLM_PID}" EXIT

# start training on GPU 1
CUDA_VISIBLE_DEVICES=1 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    data.train_files=data/MATH-TTT/train_4303.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.actor.fsdp.enable_cpu_offload=false \
    worker.rollout.temperature=1.5 \
    trainer.experiment_name=1-GPRO-1.7B-reason-LLM \
    trainer.save_checkpoint_path=${STORAGE_PATH}/checkpoints_1.7B_GRPO_reason_LLM \
    worker.reward.reward_function=examples/reward_function/math_reason.py:compute_score \
    trainer.find_last_checkpoint=false \
    trainer.total_epochs=2
