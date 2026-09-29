# Save Your Saturated Data: Learning Beyond Reward Saturation in Group-Based RL

> We study how to recover useful learning signals from reward-saturated reasoning data in group-relative RL by testing interventions across data, rollout, reward, and advantage levels.

Paper link: [https://arxiv.org/abs/2609.33126](https://arxiv.org/abs/2609.33126)

## Overview

![](/asserts/overview.png)

As language models become increasingly capable, existing training data can become **reward-saturated**: all sampled responses to the same problem may receive uniformly high rewards, causing group-relative learning signals to vanish and leaving previously useful data obsolete.

In this work, we investigate whether useful learning signals can be recovered from such saturated data. We study interventions at four levels of GRPO pipelines: **data, rollout, reward, and advantage**. Among these methods, nudging the policy to generate "high-quality" incorrect solutions is the most effective. Other interventions, such as increasing rollout temperature or adding auxiliary rewards, can also restore non-zero advantages, but they yield less consistent gains.

Further analyses show that effective negative rollouts require **informative** negative trajectories, that the method remains effective when combined with unsaturated data, and that it supports iterative recycling of newly saturated examples. While increasingly stronger LLMs will render more data saturated, our results show that saturated data should not be wasted: with the right strategies, it can be recycled into useful RL training signals in an increasingly data-scarce world.

## Installation

```bash
conda create -n saturated-rl python=3.10 -y
conda activate saturated-rl

pip install -r requirements.txt
```

## Running RL Experiments

We use the [verl](https://github.com/verl-project/verl) framework for RL training. Each experiment directory contains launch scripts under `examples/`. Before running an experiment, update `MODEL_PATH`, `STORAGE_PATH`, and any GPU-related settings in the corresponding script.

The core intervention implementations are provided under `interventions/`. You can replace the corresponding components in the training pipeline or run the provided examples directly.

## Baselines

### Supervised Fine-Tuning Baseline

The SFT baseline is located in:

```text
Baselines/SFT/
```

Run the 1.7B or 4B baseline:

```bash
cd Baselines/SFT
bash run_sft.sh
bash run_sft_4B.sh
```

The scripts use `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` by default. Edit `MODEL_NAME`, `DATA_PATH`, and `OUTPUT_DIR` as needed.

### RL Baselines

Standard RL baselines include GRPO and [Mixed-CUTS](https://aclanthology.org/2026.acl-short.19/). They are located in:

```text
Baselines/rl_baselines/
Baselines/rl_CUTS/
```

Example:

```bash
cd Baselines/rl_baselines
bash examples/my_qwen3_1.7B.sh
```

## Interventions

### Data-Level Intervention

This intervention provides modified training sets for data-level ablations, including irrelevant-context and rewritten-problem variants. The files are located in:

```text
interventions/Data_level/
```

Available files:

```text
train_4303.json              # Standard training data
train_4303_irrelevant.json   # Irrelevant information
train_4303_rewrite.json      # Prompt rephrasing
```

Use these files by overriding `data.train_files` in an experiment script or on the command line:

```bash
data.train_files=interventions/Data_level/train_4303_rewrite.json
```

### Rollout-Level Intervention

This intervention modifies rollout generation to recover learning signals from saturated groups. We study higher-temperature sampling and three types of wrong rollouts: full negative rollouts, continuations from wrong prefixes, and boxed-answer replacements. The files are located in:

```text
interventions/Rollout_level/
```

Available rollout-level variants:

```text
rl_higher_temperature/     # Higher rollout sampling temperature
rl_negative_rollouts/      # Additional negative rollouts for all-correct groups
rl_prefix_wrong/           # Continue from wrong prefixes for all-correct groups
rl_rebox/                  # Rewrite boxed answers for all-correct groups
```

Example:

```bash
cd interventions/Rollout_level/rl_higher_temperature
bash examples/my_qwen3_1.7B.sh
```

### Reward-Level Intervention

This intervention modifies reward computation by adding auxiliary rewards, including reward-model scores, LLM-as-judge response diversity scores, and LLM-as-judge reasoning-quality scores. The files are located in:

```text
interventions/Reward_level/rl_reward/
```

Available reward-level variants include:

```text
examples/my_qwen3_1.7B_RM.sh
examples/my_qwen3_1.7B_Reaon_Quality.sh
examples/my_qwen3_1.7B_Response_diversity.sh
examples/my_qwen3_4B_RM.sh
examples/my_qwen3_4B_Reaon_Quality.sh
examples/my_qwen3_4B_Response_diversity.sh
```

Some scripts start a local vLLM service for reward-model or LLM-judge scoring before launching RL training. Update `MODEL_PATH`, `STORAGE_PATH`, `VLLM_CACHE_ROOT`, and GPU assignments before running.

Example:

```bash
cd interventions/Reward_level/rl_reward
bash examples/my_qwen3_1.7B_Response_diversity.sh
```

### Advantage-Level Intervention

This intervention modifies advantage computation by appending zero-reward pseudo-samples to saturated reward groups before computing group statistics. The files are located in:

```text
interventions/Advantage_level/
```

`ADD_ZERO_COUNT` controls how many zeros are appended.

Example:

```bash
cd interventions/Advantage_level/rl_advantages
ADD_ZERO_COUNT=4 bash examples/my_qwen3_1.7B_add_k.sh
ADD_ZERO_COUNT=4 bash examples/my_qwen3_4B_add_k.sh
```

Use `ADD_ZERO_COUNT=-1` to randomly sample the number of appended zeros from 1 to 7.

## Evaluation

Evaluation scripts are provided in `evaluation/`. Set `STORAGE_PATH` before running the generation scripts; results will be saved under `$STORAGE_PATH/evaluation/<model_name>/`.

For math and reasoning benchmarks, run:

```bash
cd evaluation
python generate.py --model <MODEL_PATH> --dataset math
python generate.py --model <MODEL_PATH> --dataset gpqa
python generate.py --model <MODEL_PATH> --dataset minerva
python generate.py --model <MODEL_PATH> --dataset aime2024
python generate.py --model <MODEL_PATH> --dataset aime2025
```

Other supported dataset names include `gsm8k`, `amc`, `olympiad`, `aime2024`, `mmlu_pro`, `super_gpqa`, and `mydataset`.

For BBEH, run:

```bash
cd evaluation
python eval_bbeh.py --model_path <MODEL_PATH>
```

For instruction-following benchmarks, run:

```bash
cd evaluation/if_evaluation
python generate.py --model <MODEL_PATH> --dataset ifbench
python generate.py --model <MODEL_PATH> --dataset ifeval
```

## Acknowledgements

We build upon several open-source projects, including [**verl**](https://github.com/verl-project/verl). We sincerely thank the authors for their excellent work.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@misc{yang2026saturationrl,
      title={Save Your Saturated Data: Learning Beyond Reward Saturation in Group-Based RL},
      author={Ziyuan Yang and Yike Wang and Shangbin Feng and Yulia Tsvetkov},
      year={2026},
      eprint={2609.33126},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.33126}
}
```
