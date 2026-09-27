import json
import re
import requests
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

REWARD_NAME = "math_reason_judge"
REWARD_TYPE = "batch"

_QWEN3_PREFIX = "<|im_start|>system\n/no_think\n<|im_end|>\n<|im_start|>user\n"
_QWEN3_SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n"

_JUDGE_TEMPLATE = """\
You are an expert math evaluator. Score the student's REASONING PROCESS (not just the final answer).

Problem: {problem}
Ground Truth Answer: {ground_truth}
Student's Solution: {response}

Scoring (integer only):
- 2: correct approach and valid steps, even if there are minor computational mistakes
- 1: right direction but has significant errors (e.g. wrong substitution, unjustified jumps)
- 0: fundamentally wrong reasoning or no meaningful attempt

Output ONLY JSON: {{"score": <0|1|2>, "reason": "<brief>"}}"""


def _accuracy_score(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def _build_prompt(problem: str, response: str, ground_truth: str) -> str:
    body = _JUDGE_TEMPLATE.format(problem=problem, response=response, ground_truth=ground_truth)
    return _QWEN3_PREFIX + body + _QWEN3_SUFFIX


def _parse_score(text: str) -> float:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return float(max(0.0, min(2.0, json.loads(m.group())["score"])))
        except (KeyError, ValueError, TypeError):
            pass
    m = re.search(r'"score"\s*:\s*([\d.]+)', text)
    if m:
        try:
            return float(max(0.0, min(2.0, float(m.group(1)))))
        except ValueError:
            pass
    return 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    judge_url: str = "http://127.0.0.1:7781/generate",
    max_tokens: int = 256,
    timeout: int = 600,
    accuracy_weight: float = 0.5,
    reason_weight: float = 0.5,
) -> list[dict[str, float]]:
    # Rule-based accuracy (no API call needed)
    acc_scores = [
        _accuracy_score(inp["response"], inp["ground_truth"])
        for inp in reward_inputs
    ]

    # LLM-as-judge for reasoning quality
    prompts = [
        _build_prompt(
            problem=inp.get("problem") or inp.get("prompt") or "",
            response=inp["response"],
            ground_truth=inp["ground_truth"],
        )
        for inp in reward_inputs
    ]
    try:
        payload = {
            "requests": [
                {"prompt": p, "max_tokens": max_tokens, "temperature": 0.0, "n": 1}
                for p in prompts
            ]
        }
        resp = requests.post(judge_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        results = resp.json()["results"]
        reason_scores = [_parse_score(r.get("text", "")) / 2.0 for r in results]
    except Exception as e:
        print(f"[math_reason] judge API failed: {e}")
        reason_scores = [0.0] * len(reward_inputs)

    return [
        {
            "overall": accuracy_weight * acc + reason_weight * reason,
            "accuracy": acc,
            "reason_score": reason,
        }
        for acc, reason in zip(acc_scores, reason_scores)
    ]
