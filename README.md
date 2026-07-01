# HAPPO on Two-Player Sokoban

This repository implements and studies the **HAPPO** (Heterogeneous-Agent Proximal Policy Optimization) algorithm from the HARL framework on a **two-player Sokoban** environment. Two cooperative agents are trained to push all boxes onto target locations in randomly generated grid rooms.

The implementation covers:
- Small rooms: `5×5` with 1 and 2 boxes
- Larger rooms: `7×7` with 1 and 2 boxes
- Curriculum transfer: bootstrapping `7×7-2box` from a trained `7×7-1box` model
- Evaluation, rendering, and visualization of trained policies

All experiments were run on a single NVIDIA RTX 5060 (8 GB VRAM) with 4 parallel rollout threads.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Quick Start](#3-quick-start)
4. [Training](#4-training)
5. [Evaluation](#5-evaluation)
6. [Visualization](#6-visualization)
7. [Results](#7-results)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Project Overview

| Component | Description |
|-----------|-------------|
| Algorithm | HAPPO from HARL |
| Environment | Two-player Sokoban via `gym-sokoban` |
| State representation | Symbolic grid with 5 binary channels: walls, goals, boxes, player 1, player 2 |
| Action space | Each agent: `Discrete(9)` — NOOP, push up/down/left/right, move up/down/left/right |
| Reward | `-0.1` per step, `+1` for pushing a box on target, `+10` for solving the puzzle, optional timeout penalty |

One HARL step executes **two underlying environment steps** (player 1 acts, then player 2 acts). Therefore `max_steps` should be at least `2 × episode_length`.

---

## 2. Environment Setup

### 2.1 Prerequisites

- Python >= 3.8
- NVIDIA GPU with CUDA support (tested on RTX 5060 8 GB)

### 2.2 Create and Activate a Conda Environment

Open Anaconda Prompt (or any terminal with conda available) and run:

```powershell
conda create -n harl_sokoban python=3.10
conda activate harl_sokoban
```

After activation, your prompt should start with `(harl_sokoban)`.

### 2.3 Install Dependencies

```powershell
pip install -e .
pip install gym-sokoban imageio matplotlib
```

The base HARL package requires `torch`, `pyyaml`, `tensorboard`, `tensorboardX`, and `setproctitle`. `gym-sokoban` uses the older `gym` API; the wrapper in `harl/envs/sokoban/` handles compatibility.

### 2.4 Verify Installation

```powershell
python -c "from harl.envs.sokoban.sokoban_env import SokobanEnv; print('Sokoban env OK')"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 3. Quick Start

The main training script is `examples/train.py`. A typical command looks like:

```powershell
python examples/train.py --algo happo --env sokoban --exp_name <run_name> ^
  --dim_room [H,W] --num_boxes N --max_steps M ^
  --hidden_sizes [256,256] --lr 0.0005 --critic_lr 0.0005 ^
  --entropy_coef 0.1 --n_rollout_threads 4 --episode_length 100 ^
  --num_env_steps 200000
```

The trailing `^` is the Windows line-continuation character.

Results are saved under:

```text
results/sokoban/TwoPlayerSokoban-<H>x<W>-<N>boxes/happo/<exp_name>/<seed>/
```

Each run contains:
- `config.json` — full hyperparameter snapshot
- `progress.txt` — training/evaluation reward curve
- `models/` — saved checkpoints (`actor_agent0.pt`, `actor_agent1.pt`, `critic_agent.pt`, `value_normalizer.pt`)
- `logs/` — TensorBoard summaries

---

## 4. Training

### 4.1 Baseline: 5×5 with 1 box

Fast-converging baseline. Expected result: ~+11 reward.

```powershell
python examples/train.py --algo happo --env sokoban --exp_name sokoban_easy ^
  --num_env_steps 200000 ^
  --n_rollout_threads 4 ^
  --episode_length 100 ^
  --hidden_sizes [256,256] ^
  --entropy_coef 0.1 ^
  --dim_room [5,5] ^
  --num_boxes 1
```

### 4.2 5×5 with 2 boxes

Shows scaling from 1 box to 2 boxes. Expected result: ~+12 reward.

```powershell
python examples/train.py --algo happo --env sokoban --exp_name sokoban_diff ^
  --num_env_steps 1000000 ^
  --n_rollout_threads 4 ^
  --episode_length 100 ^
  --hidden_sizes [256,256] ^
  --entropy_coef 0.1 ^
  --dim_room [5,5] ^
  --num_boxes 2
```

### 4.3 7×7 with 1 box (pre-training)

This checkpoint is used as the starting point for the harder `7×7-2box` task.

```powershell
python examples/train.py --algo happo --env sokoban --exp_name sokoban_7x7_1box_final ^
  --dim_room [7,7] --num_boxes 1 --max_steps 200 ^
  --timeout_penalty 10.0 ^
  --hidden_sizes [256,256] --lr 0.0003 --critic_lr 0.0003 ^
  --entropy_coef 0.05 ^
  --n_rollout_threads 4 ^
  --episode_length 100 ^
  --num_env_steps 5000000
```

Expected result: evaluation reward reaches ~+10 to +11 after 3M–5M steps.

### 4.4 7×7 with 2 boxes (curriculum transfer)

Fine-tune the trained `7×7-1box` model on `7×7-2box`. Replace `<1box_model_dir>` with the actual `models` path from Section 4.3, for example:

```text
results/sokoban/TwoPlayerSokoban-7x7-1boxes/happo/sokoban_7x7_1box_final/seed-00001-YYYY-MM-DD-HH-MM-SS/models
```

**Stage 1 — high-entropy fine-tuning:**

```powershell
python examples/train.py --algo happo --env sokoban --exp_name sokoban_7x7_2box_ft_v2 ^
  --dim_room [7,7] --num_boxes 2 --max_steps 400 ^
  --timeout_penalty 10.0 ^
  --hidden_sizes [256,256] --lr 0.0001 --critic_lr 0.0001 ^
  --entropy_coef 0.2 ^
  --n_rollout_threads 4 ^
  --episode_length 100 ^
  --actor_num_mini_batch 4 ^
  --critic_num_mini_batch 4 ^
  --torch_threads 2 ^
  --num_env_steps 10000000 ^
  --model_dir "<1box_model_dir>"
```

**Stage 2 — lower entropy and learning rate:**

```powershell
python examples/train.py --algo happo --env sokoban --exp_name sokoban_7x7_2box_ft_v4 ^
  --dim_room [7,7] --num_boxes 2 --max_steps 400 ^
  --timeout_penalty 10.0 ^
  --hidden_sizes [256,256] --lr 0.00003 --critic_lr 0.00003 ^
  --entropy_coef 0.03 ^
  --n_rollout_threads 4 ^
  --episode_length 100 ^
  --actor_num_mini_batch 4 ^
  --critic_num_mini_batch 4 ^
  --torch_threads 2 ^
  --n_eval_rollout_threads 4 ^
  --num_env_steps 5000000 ^
  --model_dir "<v2_model_dir>"
```

**Stage 3 — final convergence:**

```powershell
python examples/train.py --algo happo --env sokoban --exp_name sokoban_7x7_2box_ft_v6 ^
  --dim_room [7,7] --num_boxes 2 --max_steps 400 ^
  --timeout_penalty 10.0 ^
  --hidden_sizes [256,256] --lr 0.00003 --critic_lr 0.00003 ^
  --entropy_coef 0.02 ^
  --n_rollout_threads 4 ^
  --episode_length 100 ^
  --actor_num_mini_batch 4 ^
  --critic_num_mini_batch 4 ^
  --torch_threads 2 ^
  --n_eval_rollout_threads 4 ^
  --num_env_steps 5000000 ^
  --model_dir "<v4_model_dir>"
```

Expected final result: best training episodes reach ~0 to +8 reward; evaluation average is around -8 to -12 with high variance due to random room difficulty.

---

## 5. Evaluation

To evaluate a saved checkpoint without further training, add `--eval True`:

```powershell
python examples/train.py --algo happo --env sokoban --exp_name sokoban_7x7_2box_eval ^
  --dim_room [7,7] --num_boxes 2 --max_steps 400 ^
  --timeout_penalty 10.0 ^
  --hidden_sizes [256,256] ^
  --n_rollout_threads 4 ^
  --episode_length 100 ^
  --model_dir "<model_dir>" ^
  --eval True
```

This loads the model, runs 20 evaluation episodes, and prints the average episode reward.

You can also inspect the raw reward curve:

```powershell
type results\sokoban\TwoPlayerSokoban-7x7-2boxes\happo\sokoban_7x7_2box_ft_v6\seed-00001-*\progress.txt
```

---

## 6. Visualization

### 6.1 Plot Learning Curves

A helper script is provided at `plot_curves.py`. Edit it to point to your result paths, then run:

```powershell
python plot_curves.py
```

PDF figures are saved to `icml2022/figures/`.

For a quick custom plot from a single `progress.txt`:

```python
import matplotlib.pyplot as plt

steps, rewards = [], []
with open("results/sokoban/.../progress.txt") as f:
    for line in f:
        s, r = line.strip().split(",")
        steps.append(int(s))
        rewards.append(float(r))

plt.plot(steps, rewards)
plt.xlabel("Environment steps")
plt.ylabel("Average episode return")
plt.title("7×7-2box HAPPO training")
plt.grid(True)
plt.savefig("learning_curve.png", dpi=150)
```

### 6.2 Render a Trained Policy to GIF

Use `examples/render_sokoban_to_gif.py` to generate a GIF of the trained policy playing:

```powershell
python examples/render_sokoban_to_gif.py ^
  --config "results/sokoban/TwoPlayerSokoban-7x7-2boxes/happo/sokoban_7x7_2box_ft_v6/seed-00001-*/config.json" ^
  --model_dir "results/sokoban/TwoPlayerSokoban-7x7-2boxes/happo/sokoban_7x7_2box_ft_v6/seed-00001-*/models" ^
  --output sokoban_7x7_2box_demo.gif ^
  --render_episodes 1 ^
  --fps 4
```

The output GIF shows the two agents solving a room frame by frame.

### 6.3 TensorBoard

Training metrics are logged to TensorBoard:

```powershell
tensorboard --logdir results/sokoban
```

Open `http://localhost:6006` in a browser to view reward curves, value loss, policy entropy, and gradient norms.

---

## 7. Results

| Task | Final train reward | Best single episode | Notes |
|------|-------------------|---------------------|-------|
| 5×5-1box | ~+11 | ~+12 | Converges quickly, easy baseline |
| 5×5-2box | ~+12 | ~+13 | Slightly harder, still converges fast |
| 7×7-1box | ~+10 to +11 (eval) | ~+11 | Requires ~5M steps |
| 7×7-2box | ~0 to +8 (best), ~-10 (eval avg) | ~+11 | Hard; curriculum from 7×7-1box essential |

Key findings:
- `timeout_penalty=10.0` is critical for hard levels; without it the agent learns to timeout.
- High entropy (`0.2`) is needed early in `7×7-2box` fine-tuning; lower it (`0.02–0.05`) for final convergence.
- `7×7-2box` requires roughly **20–25M total environment steps** from the `7×7-1box` checkpoint.
- On a single RTX 5060, keep `n_rollout_threads=4`, `hidden_sizes=[256,256]`, and use `actor_num_mini_batch=4` to stay within 8 GB VRAM.

---

## 8. Troubleshooting

### GPU out of memory
- Reduce `n_rollout_threads` to 4.
- Use `hidden_sizes [128,128]` instead of `[256,256]`.
- Set `torch_threads 2`.
- Reduce `n_eval_rollout_threads` to 4.

### Room generation warnings (`Generated Model with score == 0`)
These are normal retries from `gym-sokoban` when a randomly generated room is invalid. Training continues automatically.

### Train reward much better than eval reward
This indicates the policy has partially memorized training room layouts. Solutions:
- Continue training with lower entropy to improve generalization.
- Use the CNN backbone (`--use_cnn True`) for better spatial generalization.
- Increase parallel environments if hardware allows.

### 7×7-2box does not improve from 7×7-1box fine-tuning
If the 1box model is not converged enough, 2box transfer fails. Make sure 1box eval reward reaches ~+10 before fine-tuning.

---

## File Reference

| File | Purpose |
|------|---------|
| `examples/train.py` | Main training and evaluation script |
| `harl/envs/sokoban/sokoban_env.py` | Sokoban wrapper for HARL |
| `harl/configs/envs_cfgs/sokoban.yaml` | Default Sokoban configuration |
| `examples/render_sokoban_to_gif.py` | Render a trained policy to GIF |
| `plot_curves.py` | Plot learning curves for reports |
