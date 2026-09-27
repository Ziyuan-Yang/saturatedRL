"""
Reward model service backed by vLLM (runner="pooling", vLLM 0.9+).
Exposes POST /v1/score — accepts pre-formatted texts, returns raw scores.

Usage:
    python vllm_service6.py \
        --model Skywork/Skywork-Reward-V2-Qwen3-8B \
        --port 7780 \
        --tensor-parallel-size 1
"""

import argparse
import asyncio
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM

app = FastAPI(title="Reward Model Service (vLLM backend)")

llm_engine: Optional[LLM] = None


class ScoreRequest(BaseModel):
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"
    input: List[str]


class ScoreItem(BaseModel):
    index: int
    score: float
    object: str = "score"


class ScoreResponse(BaseModel):
    data: List[ScoreItem]


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "vllm-reward"}


@app.post("/v1/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input:
        raise HTTPException(status_code=400, detail="input list is empty")

    # run_in_executor: vLLM encode() is synchronous; keep event loop unblocked
    loop = asyncio.get_event_loop()
    encode_fn = lambda: llm_engine.encode(req.input, pooling_task="classify")
    outputs = await loop.run_in_executor(None, encode_fn)

    # outputs.data: [scalar] per vLLM 0.9+ pooling API
    scores = [float(out.outputs.data[0]) for out in outputs]
    return ScoreResponse(data=[ScoreItem(index=i, score=s) for i, s in enumerate(scores)])


def main():
    global llm_engine

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Skywork/Skywork-Reward-V2-Qwen3-8B")
    parser.add_argument("--port", type=int, default=7780)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    print("=" * 50)
    print(f"Starting Reward Model Service on port {args.port}")
    print(f"Loading model: {args.model}  tp={args.tensor_parallel_size}")

    llm_engine = LLM(
        model=args.model,
        runner="pooling",
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
