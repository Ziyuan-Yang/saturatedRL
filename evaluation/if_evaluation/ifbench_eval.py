import json
import os
import sys


EVALUATION_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_IFBENCH_DATA_PATH = os.path.join(EVALUATION_DIR, "ifbench.json")
DEFAULT_IFEVAL_DATA_PATH = os.path.join(EVALUATION_DIR, "ifeval.json")

if EVALUATION_DIR not in sys.path:
    sys.path.append(EVALUATION_DIR)


def _load_json_or_jsonl(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        if data_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def load_ifbench_examples(data_path=None, split=None, default_data_path=DEFAULT_IFBENCH_DATA_PATH):
    path = data_path or default_data_path
    if path in ("dev", "test") and not os.path.exists(path):
        split = path
        path = default_data_path
    payload = _load_json_or_jsonl(path)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if split and split in payload:
            return payload[split]
        if "test" in payload:
            return payload["test"]
        if "dev" in payload:
            return payload["dev"]

    raise ValueError(f"Unsupported IFBench data format in {path}")


def score_ifbench(examples, responses):
    from ifbench.run_eval import if_score

    scores = if_score(examples, responses)
    average_score = sum(scores) / len(scores) if scores else 0.0
    return scores, average_score
