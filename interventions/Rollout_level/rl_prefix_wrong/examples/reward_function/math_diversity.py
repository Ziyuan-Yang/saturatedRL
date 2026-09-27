import json
import re
import requests
from collections import defaultdict
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

REWARD_NAME = "math_diversity_judge"
REWARD_TYPE = "batch"

_QWEN3_PREFIX = "<|im_start|>system\n/no_think\n<|im_end|>\n<|im_start|>user\n"
_QWEN3_SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n"

_DIVERSITY_JUDGE_TEMPLATE = """\
You are an expert math evaluator assessing solution diversity.

Problem: {problem}

Below are {n} student solutions (numbered 1 to {n}). Evaluate how UNIQUE each solution's \
reasoning approach is compared to the others. Consider: different mathematical methods, \
different algebraic manipulations, different proof strategies, or different key insights.

{solutions}

For each solution, assign a uniqueness score:
- 2: clearly uses a distinct approach not seen in any other solution
- 1: partially different — shares some steps but has a notable unique element
- 0: essentially the same approach as another solution

Output ONLY JSON (array of length {n}):
{{"scores": [<0|1|2>, ...], "reasons": ["<brief>", ...]}}"""


def _accuracy_score(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def _build_diversity_prompt(
    problem: str, responses: list[str], max_chars_per_response: int = 1000
) -> str:
    def _truncate(r: str) -> str:
        return r[:max_chars_per_response] + " [truncated]" if len(r) > max_chars_per_response else r

    solutions_text = "\n\n".join(
        f"Solution {i+1}:\n{_truncate(r)}" for i, r in enumerate(responses)
    )
    body = _DIVERSITY_JUDGE_TEMPLATE.format(
        problem=problem, n=len(responses), solutions=solutions_text
    )
    return _QWEN3_PREFIX + body + _QWEN3_SUFFIX


def _parse_diversity_scores(text: str, n: int) -> list[float]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r'\{[^{}]*"scores"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            scores = json.loads(m.group())["scores"]
            if isinstance(scores, list) and len(scores) == n:
                return [float(max(0, min(2, round(s)))) for s in scores]
        except (KeyError, ValueError, TypeError):
            pass
    # fallback: extract bare array
    m = re.search(r'"scores"\s*:\s*\[([^\]]+)\]', text)
    if m:
        try:
            scores = [float(x.strip()) for x in m.group(1).split(",")]
            if len(scores) == n:
                return [float(max(0, min(2, round(s)))) for s in scores]
        except ValueError:
            pass
    return [0.0] * n


def _call_judge(
    prompts: list[str],
    judge_url: str,
    max_tokens: int,
    timeout: int,
) -> list[str]:
    payload = {
        "requests": [
            {"prompt": p, "max_tokens": max_tokens, "temperature": 0.0, "n": 1}
            for p in prompts
        ]
    }
    resp = requests.post(judge_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return [r.get("text", "") for r in resp.json()["results"]]


def compute_score(
    reward_inputs: list[dict[str, Any]],
    judge_url: str = "http://127.0.0.1:7781/generate",
    max_tokens: int = 512,
    timeout: int = 600,
    accuracy_weight: float = 0.5,
    diversity_weight: float = 0.5,
) -> list[dict[str, float]]:
    n = len(reward_inputs)

    # --- 1. Rule-based accuracy (no API) ---
    acc_scores = [
        _accuracy_score(inp["response"], inp["ground_truth"])
        for inp in reward_inputs
    ]

    # --- 2. LLM judge: diversity (one request per unique problem) ---
    # Group by uid (same as math3.py); fall back to prompt text if uid absent.
    uid_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, inp in enumerate(reward_inputs):
        key = inp.get("uid") or inp.get("id") or inp.get("problem") or inp.get("prompt") or ""
        uid_to_indices[key].append(i)

    diversity_scores = [0.0] * n
    diversity_prompts = []
    prompt_meta = []  # (uid_key, [indices])

    for uid_key, indices in uid_to_indices.items():
        # Only score correct responses (same logic as math3.py)
        correct_indices = [i for i in indices if acc_scores[i] == 1.0]
        if len(correct_indices) < 2:
            for i in correct_indices:
                diversity_scores[i] = 2.0
            continue
        problem_text = reward_inputs[correct_indices[0]].get("problem") or reward_inputs[correct_indices[0]].get("prompt") or uid_key
        responses = [reward_inputs[i]["response"] for i in correct_indices]
        diversity_prompts.append(_build_diversity_prompt(problem_text, responses))
        prompt_meta.append((uid_key, correct_indices))

    if diversity_prompts:
        try:
            texts = _call_judge(
                diversity_prompts, judge_url, max_tokens=max_tokens, timeout=timeout
            )
            for text, (_, indices) in zip(texts, prompt_meta):
                scores = _parse_diversity_scores(text, len(indices))
                for idx, s in zip(indices, scores):
                    diversity_scores[idx] = s
        except Exception as e:
            print(f"[math_diversity] diversity judge API failed: {e}")

    # --- 3. Combine ---
    return [
        {
            "overall": accuracy_weight * acc + diversity_weight * div,
            "accuracy": acc,
            "diversity_score": div,
        }
        for acc, div in zip(acc_scores, diversity_scores)
    ]
