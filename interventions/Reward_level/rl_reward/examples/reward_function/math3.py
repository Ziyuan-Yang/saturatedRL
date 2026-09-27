# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    # pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    # Match responses that contain a LaTeX-style boxed answer anywhere.
    pattern = re.compile(r".*\\boxed\{.*?\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def _get_ngrams(text: str, n: int) -> set:
    tokens = text.split()
    return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _jaccard_distance(a: set, b: set) -> float:
    union = len(a | b)
    return 1.0 - len(a & b) / union if union > 0 else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    diversity_weight: float = 0.05,
    ngram_n: int = 2,
) -> list[dict[str, float]]:
    # Pre-compute per-sample scores
    responses, format_scores, accuracy_scores = [], [], []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        responses.append(response)
        format_scores.append(format_reward(response))
        accuracy_scores.append(accuracy_reward(response, reward_input["ground_truth"]))

    # Group by uid to compute inter-response diversity within same question
    from collections import defaultdict
    uid_to_indices: dict = defaultdict(list)
    for i, reward_input in enumerate(reward_inputs):
        uid_to_indices[reward_input["uid"]].append(i)

    diversity_scores = [0.0] * len(reward_inputs)
    for indices in uid_to_indices.values():
        if len(indices) < 2:
            continue
        ngrams = [_get_ngrams(responses[i], ngram_n) for i in indices]
        for k, i in enumerate(indices):
            if accuracy_scores[i] != 1.0:
                continue
            dists = [_jaccard_distance(ngrams[k], ngrams[j]) for j in range(len(indices)) if j != k]
            diversity_scores[i] = sum(dists) / len(dists)

    scores = []
    for i in range(len(reward_inputs)):
        scores.append(
            {
                "overall": (1 - format_weight - diversity_weight) * accuracy_scores[i]
                    + format_weight * format_scores[i]
                    + diversity_weight * diversity_scores[i],
                "format": format_scores[i],
                "accuracy": accuracy_scores[i],
                "diversity": diversity_scores[i],
            }
        )
    return scores
