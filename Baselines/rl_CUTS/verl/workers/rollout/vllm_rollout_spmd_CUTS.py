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
#
# Mixed-CUTS implementation based on:
#   "Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data"
#   arXiv:2604.18493

import copy
from typing import List, Optional

import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from ...protocol import DataProto
from ...utils import torch_functional as VF
from .config import RolloutConfig
from .vllm_rollout_spmd import (
    _get_logit_bias,
    _process_multi_modal_data,
    _repeat_interleave,
    vLLMRollout,
)


class CUTSLogitsProcessor:
    """
    Constrained Uniform Top-K Sampling (CUTS) from arXiv:2604.18493.

    At each decoding step t:
      - For t < warmup_steps: standard sampling (no modification)
      - For t >= warmup_steps:
          1. Identify top-K tokens by probability
          2. Keep only tokens with prob >= prob_threshold (delta)
             - if all fail threshold: fall back to full top-K set
             - if exactly one passes: deterministic (uniform over 1 token)
          3. Set equal logits (0.0) for valid candidates, -inf elsewhere
             → after softmax this yields a uniform distribution over the set

    Args:
        top_k: candidate set size K (default 5)
        prob_threshold: minimum probability delta (default 0.03)
        warmup_steps: number of initial tokens using standard sampling (default 5)
    """

    def __init__(self, top_k: int = 5, prob_threshold: float = 0.03, warmup_steps: int = 5):
        self.top_k = top_k
        self.prob_threshold = prob_threshold
        self.warmup_steps = warmup_steps

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        if len(token_ids) < self.warmup_steps:
            return logits

        vocab_size = logits.size(-1)
        k = min(self.top_k, vocab_size)

        probs = torch.softmax(logits.float(), dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=k)

        valid_mask = top_k_probs >= self.prob_threshold
        if valid_mask.any():
            valid_indices = top_k_indices[valid_mask]
        else:
            # Fallback: use full top-K when all candidates fail the threshold
            valid_indices = top_k_indices

        # Equal logits → uniform distribution after softmax at temperature 1.0
        new_logits = torch.full_like(logits, float("-inf"))
        new_logits[valid_indices] = 0.0
        return new_logits


class vLLMRolloutCUTS(vLLMRollout):
    """
    Mixed-CUTS rollout (arXiv:2604.18493).

    For each prompt, generates n total trajectories split as:
      - n_exploit = round(n * exploit_ratio)  via standard sampling (exploitation)
      - n_explore = n - n_exploit             via CUTS decoding    (exploration)

    The mixed group preserves non-zero advantage variance even when the base model
    already achieves high accuracy on the training data (saturation regime).

    New config fields (all optional, with defaults matching the paper):
      cuts_enabled        bool   True    enable Mixed-CUTS (False = original behavior)
      cuts_top_k          int    5       K in CUTS candidate set
      cuts_prob_threshold float  0.03    probability threshold delta
      cuts_warmup_steps   int    5       warm-up steps before CUTS activates
      cuts_exploit_ratio  float  0.5     fraction of rollouts using standard sampling
    """

    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        **kwargs,
    ):
        super().__init__(model_path, config, tokenizer, processor, **kwargs)

        self.cuts_enabled = getattr(config, "cuts_enabled", True)
        self.cuts_top_k = getattr(config, "cuts_top_k", 5)
        self.cuts_prob_threshold = getattr(config, "cuts_prob_threshold", 0.03)
        self.cuts_warmup_steps = getattr(config, "cuts_warmup_steps", 5)
        self.cuts_exploit_ratio = getattr(config, "cuts_exploit_ratio", 0.5)

        n_total = self.sampling_params.n
        self.n_exploit = max(1, round(n_total * self.cuts_exploit_ratio))
        self.n_explore = n_total - self.n_exploit

        if self.cuts_enabled and self.n_explore > 0:
            print(
                f"[Mixed-CUTS] n_exploit={self.n_exploit}, n_explore={self.n_explore}, "
                f"top_k={self.cuts_top_k}, delta={self.cuts_prob_threshold}, "
                f"warmup={self.cuts_warmup_steps}"
            )

    def _build_sampling_params(self, meta_info: dict, n: int, use_cuts: bool = False) -> SamplingParams:
        """
        Build a SamplingParams instance from self.sampling_params + meta_info overrides.
        The n value is set explicitly (meta_info 'n' is ignored) so that exploit/explore
        split is preserved regardless of what the caller passes via meta_info.
        """
        params = copy.deepcopy(self.sampling_params)
        for key, value in meta_info.items():
            if key == "n":
                continue  # always use the computed exploit/explore n
            if hasattr(params, key):
                setattr(params, key, value)
        params.n = n
        if use_cuts:
            params.logits_processors = [
                CUTSLogitsProcessor(
                    top_k=self.cuts_top_k,
                    prob_threshold=self.cuts_prob_threshold,
                    warmup_steps=self.cuts_warmup_steps,
                )
            ]
        return params

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # Fall back to base implementation when CUTS is disabled or no explore budget
        if not self.cuts_enabled or self.n_explore == 0:
            return super().generate_sequences(prompts)

        input_ids = prompts.batch["input_ids"]          # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not working properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append({
                    "prompt_token_ids": list(raw_prompt_ids),
                    "multi_modal_data": _process_multi_modal_data(
                        multi_modal_data,
                        prompts.meta_info["min_pixels"],
                        prompts.meta_info["max_pixels"],
                        prompts.meta_info["video_fps"],
                        return_video_metadata=self.return_video_metadata,
                    ),
                })
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(
                        lora_name=f"{lora_int_id}",
                        lora_int_id=lora_int_id,
                        lora_path="/simon-stub-path",
                    )
                ] * batch_size

        meta_info = prompts.meta_info
        std_params = self._build_sampling_params(meta_info, self.n_exploit, use_cuts=False)
        cuts_params = self._build_sampling_params(meta_info, self.n_explore, use_cuts=True)

        # Exploitation pass: standard sampling
        completions_std: list[RequestOutput] = self.inference_engine.generate(
            prompts=vllm_inputs,
            sampling_params=std_params,
            lora_request=lora_requests,
            use_tqdm=self.use_tqdm,
        )

        # Exploration pass: CUTS sampling
        completions_cuts: list[RequestOutput] = self.inference_engine.generate(
            prompts=vllm_inputs,
            sampling_params=cuts_params,
            lora_request=lora_requests,
            use_tqdm=False,
        )

        # Merge outputs: for each prompt, std outputs first, then CUTS outputs
        # Result shape: [bs * n_total] in prompt-major order matching _repeat_interleave
        n_total = self.n_exploit + self.n_explore
        response_ids_list = []
        for c_std, c_cuts in zip(completions_std, completions_cuts):
            for output in c_std.outputs:
                response_ids_list.append(output.token_ids)
            for output in c_cuts.outputs:
                response_ids_list.append(output.token_ids)

        response_ids = VF.pad_2d_list_to_length(
            response_ids_list, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)

        # Expand prompt tensors to match n_total samples per prompt
        if n_total > 1:
            batch_size = batch_size * n_total
            input_ids = _repeat_interleave(input_ids, n_total)
            attention_mask = _repeat_interleave(attention_mask, n_total)
            position_ids = _repeat_interleave(position_ids, n_total)
            if batch_multi_modal_data is not None:
                batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, n_total)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (bs, 4, seq_len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, position_ids.size(1), -1
            )

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
