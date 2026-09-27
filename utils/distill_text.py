import os
import json
import torch
import argparse
from distributed_sft import distributed_sft
import distributed_generation
import eval
from torch import _dynamo


def distill(
    teacher_model_name: str,
    dataset_path: str,
    student_model_path: str,
    output_path: str,
    gpu_ids: list,
    epoch: int = 1,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 64,
):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # ── 1. Generate teacher outputs ───────────────────────────────────────────
    print(f"[1/2] Generating  teacher={teacher_model_name}  dataset={dataset_path}")
    distributed_generation.update_generation_hyperparameters(
        max_response_length=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        big_model_mode=True,
    )
    inputs = eval.prepare_inputs(dataset_path, "exact_match", "dev")
    print(f"      {len(inputs)} examples")
    outputs = distributed_generation.distributed_generation([teacher_model_name], [inputs], gpu_ids)[0]

    # ── 2. Save as JSONL ──────────────────────────────────────────────────────
    sft_data_path = output_path + "_sft_data.jsonl"
    with open(sft_data_path, "w") as f:
        for inp, out in zip(inputs, outputs):
            f.write(json.dumps({"prompt": inp, "completion": out}) + "\n")
    print(f"      Saved {len(inputs)} pairs → {sft_data_path}")

    torch.cuda.empty_cache()
    _dynamo.reset_code_caches()

    # ── 3. SFT: student learns from teacher outputs ───────────────────────────
    print(f"[2/2] Distilling  student={student_model_path}  →  {output_path}")
    distributed_sft(
        [student_model_path],
        [sft_data_path],
        [0],
        [output_path],
        epoch=epoch,
    )
    print(f"      Done. Model saved to {output_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(description="Generate teacher outputs then distill to student.")
    parser.add_argument("--teacher_model_name", type=str, required=True)
    parser.add_argument("--student_model_path",  type=str, required=True)
    parser.add_argument("--dataset_path",         type=str, required=True)
    parser.add_argument("--output_path",          type=str, required=True)
    parser.add_argument("--gpu_ids",   type=str, required=True, help="e.g. '0,1,2,3'")
    parser.add_argument("--epoch",          type=int,   default=1)
    parser.add_argument("--max_new_tokens", type=int,   default=1024)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--top_p",          type=float, default=0.9)
    parser.add_argument("--batch_size",     type=int,   default=64)
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]

    distill(
        teacher_model_name=args.teacher_model_name,
        dataset_path=args.dataset_path,
        student_model_path=args.student_model_path,
        output_path=args.output_path,
        gpu_ids=gpu_ids,
        epoch=args.epoch,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )
