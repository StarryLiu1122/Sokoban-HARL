# HAPPO on Two-Player Sokoban — Demo README

This repository demonstrates training the **HAPPO** algorithm from the HARL framework on a **two-player Sokoban** environment. The goal is to train two cooperative agents to push all boxes onto target locations in a grid room.

The demo covers:
- Easy levels: `5×5` with 1 and 2 boxes
- Hard levels: `7×7` with 1 and 2 boxes
- Curriculum transfer: using a trained `7×7-1box` model to bootstrap `7×7-2box`
- Evaluation and visualization of trained policies

Hardware used for this demo: a single NVIDIA RTX 5060 (8 GB VRAM), 4 parallel rollout threads.

---

## 1. Project Overview

| Component | Description |
|-----------|-------------|
| Algorithm | HAPPO (Heterogeneous-Agent Proximal Policy Optimization) |
| Environment | Two-player Sokoban via `gym-sokoban` |
| State | Symbolic grid: walls, goals, boxes, player 1, player 2 |
| Action | Each agent: `Discrete(9)` (NOOP / push up,down,left,right / move up,down,left,right) |
| Reward | `-0.1` per step, `+1` for box on target, `+10` for solving, optional timeout penalty |

One HARL step executes **two underlying environment steps** (player 1 moves, then player 2 moves). Therefore `max_steps` should be at least `2 × episode_length`.

---

## 2. Environment Setup

### 2.1 Prerequisites

- Python >= 3.8
- CUDA-capable GPU (tested on RTX 5060 8 GB)
- Windows / Linux

### 2.2 Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate

# Install base HARL package
pip install -e .

# Install Sokoban environment and visualization tools
pip install gym-sokoban imageio matplotlib
```

> Note: `gym-sokoban` uses the older `gym` API. The wrapper in `harl/envs/sokoban/` handles the compatibility layer.

### 2.3 Verify Installation

```bash
python -c "from harl.envs.sokoban.sokoban_env import SokobanEnv; print('Sokoban env OK')"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 3. Quick Start

The main training entry point is `examples/train.py`. A typical command looks like:

```bash
python examples/train.py --algo happo --env sokoban --exp_name <run_name> ^
  --dim_room [H,W] --num_boxes N --max_steps M ^
  --hidden_sizes [256,256] --lr 0.0005 --critic_lr 0.0005 ^
  --entropy_coef 0.1 --n_rollout_threads 4 --episode_length 100 ^
  --num_env_steps 200000
```

On Linux/Mac replace the trailing `^` with `\`.

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

## 4. Training Recipes

All commands below assume Windows CMD line continuation (`^`).

### 4.1 Baseline: 5×5 with 1 box (fast convergence)

```bash
python examples/train.py --algo happo --env sokoban --exp_name sokoban_easy ^
  --num_env_steps 200000 ^
  --n_rollout_threads 4 ^
  --episode_length 100 ^
  --hidden_sizes [256,256] ^
  --entropy_coef 0.1 ^
  --dim_room [5,5] ^
  --num_boxes 1
```

Expected result: converges to ~+11 reward in ~50k–100k steps.

### 4.2 5×5 with 2 boxes

```bash
python examples/train.py --algo happo --env sokoban --exp_name sokoban_diff ^
  --num_env_steps 1000000 ^
  --n_rollout_threads 4 ^
  --episode_length 100 ^
  --hidden_sizes [256,256] ^
  --entropy_coef 0.1 ^
  --dim_room [5,5] ^
  --num_boxes 2
```

Expected result: converges to ~+12 reward in ~300k–500k steps.

### 4.3 7×7 with 1 box (pre-training for hard task)

```bash
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

### 4.4 7×7 with 2 boxes via curriculum transfer

After training `7×7-1box`, fine-tune on `7×7-2box`. Replace `<1box_model_dir>` with the path from Section 4.3, e.g.:

```text
results/sokoban/TwoPlayerSokoban-7x7-1boxes/happo/sokoban_7x7_1box_final/seed-00001-YYYY-MM-DD-HH-MM-SS/models
```

**Stage 1 — fine-tune with high exploration:**

```bash
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

**Stage 2 — lower entropy and learning rate to stabilize:**

```bash
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

```bash
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

Expected final result for 7×7-2box: final training episodes reach ~0 to +8 reward; average eval reward is around -8 to -12 with high variance due to random room difficulty.

---

## 5. Evaluation

To run a deterministic evaluation of a saved checkpoint without further training, use the `--eval True` flag:

```bash
python examples/train.py --algo happo --env sokoban --exp_name sokoban_7x7_2box_eval ^
  --dim_room [7,7] --num_boxes 2 --max_steps 400 ^
  --timeout_penalty 10.0 ^
  --hidden_sizes [256,256] ^
  --n_rollout_threads 4 ^
  --episode_length 100 ^
  --model_dir "<model_dir>" ^
  --eval True
```

This loads the model, runs `eval_episodes` (default 20) episodes, and prints the average episode reward.

You can also check `progress.txt` directly:

```bash
type results\sokoban\TwoPlayerSokoban-7x7-2boxes\happo\sokoban_7x7_2box_ft_v6\seed-00001-*\progress.txt
```

---

## 6. Visualization

### 6.1 Plot Learning Curves

A helper script is provided at `plot_curves.py`. Edit the script to point to your result paths, then run:

```bash
python plot_curves.py
```

This produces PDF figures in `icml2022/figures/` comparing different room sizes and training lengths.

If you want a quick custom plot from a single `progress.txt`:

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

```bash
python examples/render_sokoban_to_gif.py ^
  --config "results/sokoban/TwoPlayerSokoban-7x7-2boxes/happo/sokoban_7x7_2box_ft_v6/seed-00001-*/config.json" ^
  --model_dir "results/sokoban/TwoPlayerSokoban-7x7-2boxes/happo/sokoban_7x7_2box_ft_v6/seed-00001-*/models" ^
  --output sokoban_7x7_2box_demo.gif ^
  --render_episodes 1 ^
  --fps 4
```

This creates `sokoban_7x7_2box_demo.gif` showing the two agents solving a room.

### 6.3 TensorBoard

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir results/sokoban
```

Open `http://localhost:6006` to view reward curves, value loss, policy entropy, etc.

---

## 7. Results Summary

| Task | Final train reward | Best single episode | Notes |
|------|-------------------|---------------------|-------|
| 5×5-1box | ~+11 | ~+12 | Converges quickly, easy baseline |
| 5×5-2box | ~+12 | ~+13 | Slightly harder, still converges fast |
| 7×7-1box | ~+10 to +11 (eval) | ~+11 | Requires ~5M steps |
| 7×7-2box | ~0 to +8 (best), ~-10 (eval avg) | ~+11 | Hard; curriculum from 7×7-1box essential |

Key findings:
- `timeout_penalty=10` is critical for hard levels; without it the agent learns to timeout.
- High entropy (`0.2`) is needed early in `7×7-2box` fine-tuning; lower it (`0.02–0.05`) for final convergence.
- `7×7-2box` requires roughly **20–25M total environment steps** from the `7×7-1box` checkpoint.
- On a single 5060, keep `n_rollout_threads=4`, `hidden_sizes=[256,256]`, and use `actor_num_mini_batch=4` to stay within 8 GB VRAM.

---

## 8. Troubleshooting

### GPU out of memory
- Reduce `n_rollout_threads` to 4
- Use `hidden_sizes [128,128]` instead of `[256,256]`
- Set `torch_threads 2`
- Reduce `n_eval_rollout_threads` to 4

### Room generation warnings (`Generated Model with score == 0`)
These are normal retries from `gym-sokoban` when a randomly generated room is invalid. Training continues automatically.

### Train reward much better than eval reward
This indicates the policy has partially memorized training room layouts. Solutions:
- Continue training with lower entropy to improve generalization
- Use CNN backbone (`--use_cnn True`) for better spatial generalization
- Increase parallel environments if hardware allows

### 7×7-2box does not improve from 7×7-1box fine-tuning
If the 1box model is not converged enough, 2box transfer fails. Make sure 1box eval reward reaches ~+10 before fine-tuning.

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `examples/train.py` | Main training/evaluation script |
| `harl/envs/sokoban/sokoban_env.py` | Sokoban wrapper for HARL |
| `harl/configs/envs_cfgs/sokoban.yaml` | Default Sokoban config |
| `examples/render_sokoban_to_gif.py` | Render trained policy to GIF |
| `plot_curves.py` | Plot learning curves for report |

---

## 10. Citation

This demo is built on top of the HARL framework:

```bibtex
@article{harl,
  title={Heterogeneous-Agent Reinforcement Learning},
  author={PKU-MARL},
  journal={GitHub repository},
  year={2023},
  url={https://github.com/PKU-MARL/HARL}
}
```
