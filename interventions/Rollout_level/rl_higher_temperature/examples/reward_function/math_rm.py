import requests
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

REWARD_NAME = "math_reward_model"
REWARD_TYPE = "batch"

# Qwen3 chat template — no system prompt per Skywork-Reward-V2 official docs
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def _format_conversation(problem: str, response: str) -> str:
    return (
        f"{_IM_START}user\n{problem}{_IM_END}\n"
        f"{_IM_START}assistant\n{response}{_IM_END}"
    )


def _accuracy_score(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    rm_url: str = "http://127.0.0.1:7780/v1/score",
    model: str = "Skywork/Skywork-Reward-V2-Qwen3-8B",
    timeout: int = 600,
) -> list[dict[str, float]]:
    acc_scores = [
        _accuracy_score(inp["response"], inp["ground_truth"])
        for inp in reward_inputs
    ]

    texts = [
        _format_conversation(
            problem=inp.get("problem") or inp.get("prompt") or "",
            response=inp["response"],
        )
        for inp in reward_inputs
    ]

    raw_scores = [0.0] * len(reward_inputs)
    try:
        payload = {
            "model": model,
            "encoding_format": "float",
            "input": texts,
        }
        resp = requests.post(rm_url, json=payload, timeout=timeout,
                             proxies={"http": None, "https": None})
        resp.raise_for_status()
        data = resp.json()["data"]
        raw_scores = [item["score"] for item in sorted(data, key=lambda x: x["index"])]
    except Exception as e:
        print(f"[math_rm] reward model API failed: {e}")

    return [
        {
            "overall": 0.5 * raw + 0.5 * acc,
            "accuracy": acc,
            "rm_score_raw": raw,
        }
        for acc, raw in zip(acc_scores, raw_scores)
    ]
