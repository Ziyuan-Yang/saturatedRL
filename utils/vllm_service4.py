"""
vLLM service for LLM-as-judge (text generation, returns text output).
Compatible with math_reason.py which expects {"results": [{"text": "..."}]}.

Usage:
    python utils/vllm_service4.py \
        --model Qwen/Qwen3-8B \
        --port 7781 \
        --tensor_parallel_size 1
"""

import argparse

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from vllm import LLM, SamplingParams

app = FastAPI(title="vLLM Judge Service")

llm_engine: LLM = None


class GenerationRequest(BaseModel):
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = []
    n: Optional[int] = 1


class BatchGenerationRequest(BaseModel):
    requests: List[GenerationRequest]


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "vllm-judge"}


@app.post("/generate")
async def generate(req: BatchGenerationRequest):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    vllm_inputs = [
        {"prompt_token_ids": r.prompt_token_ids} if r.prompt_token_ids is not None else r.prompt
        for r in req.requests
    ]

    params_list = [
        SamplingParams(
            max_tokens=r.max_tokens,
            temperature=r.temperature,
            top_p=r.top_p,
            stop=r.stop or [],
            n=r.n or 1,
        )
        for r in req.requests
    ]

    outputs = llm_engine.generate(vllm_inputs, params_list)

    results = []
    for out in outputs:
        for o in out.outputs:
            results.append({
                "text": o.text,
                "finish_reason": o.finish_reason,
            })

    return {"results": results}


def main():
    global llm_engine

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--port", type=int, default=7781)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    args = parser.parse_args()

    print("=" * 50)
    print(f"Starting vLLM Judge Service on port {args.port}")
    print(f"Loading model: {args.model}")

    llm_engine = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        # max_model_len=args.max_model_len,
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
