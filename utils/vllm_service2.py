import uvicorn
import argparse
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from typing import List, Optional

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    stop: Optional[List[str]] = []
    n: Optional[int] = 1

class BatchGenerationRequest(BaseModel):
    requests: List[GenerationRequest]

class GenerationResponse(BaseModel):
    text: str
    finish_reason: str

llm_engine: LLM = None
executor = None

app = FastAPI(title="Batch-capable VLLM Service")

@app.get("/health") 
async def health(): 
    return {"status": "ok", "engine": "vllm"}

# @app.post("/generate", response_model=GenerationResponse)
# async def generate(req: GenerationRequest): 
#     if llm_engine is None: 
#         raise HTTPException(status_code=503, detail="Model not loaded") 
    
#     params = SamplingParams(
#         max_tokens=req.max_tokens, 
#         temperature=req.temperature, 
#         top_p=req.top_p, 
#         stop=req.stop or [], 
#         )
    
#     outputs = llm_engine.generate([req.prompt], params) 
#     out = outputs[0].outputs[0] 
#     return GenerationResponse( 
#         text=out.text, 
#         finish_reason=out.finish_reason 
#         )

@app.post("/generate")#, response_model=GenerationResponse)
async def generate(req: BatchGenerationRequest): 
    if llm_engine is None: 
        raise HTTPException(status_code=503, detail="Model not loaded") 
    
    prompts = [r.prompt for r in req.requests] 

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

    outputs = llm_engine.generate(prompts, params_list)

    results = []
    for out in outputs:
        for o in out.outputs:  # return all n samples per prompt
            results.append({
                "text": o.text,
                "finish_reason": o.finish_reason,
            })

    return {"results": results}

# --- Main Program Entry ---

def main():
    global llm_engine

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen3-8B", help="Path to the VLLM model")
    parser.add_argument('--port', type=int, default=7780, help="Port to run the service on")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help="Fraction of GPU memory to be used.")
    args = parser.parse_args()

    print("="*50)
    print(f"Starting Batch VLLM FastAPI Server on port {args.port}")
    print(f"Loading model: {args.model}")

    llm_engine = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_num_batched_tokens=2048,
    )

    uvicorn.run(
        app,
        host='127.0.0.1',
        port=args.port,
        workers=1,
        log_level="info"
    )

if __name__ == '__main__':
    
    main()
