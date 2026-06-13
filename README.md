# Spatiotemporally Prioritized Experience Replay for Off-Policy Reinforcement Learning

This repository implements **STP** (Spatiotemporally Prioritized Experience Replay), 

and integrates it with **TD3** for continuous-control off-policy reinforcement learning tasks.

## Summary

STP assigns two kinds of priority to each experience:

- **Temporal priority**: newer experiences receive higher priority, reducing over-sampling of early transitions.
- **Spatial priority**: experiencSes in sparse state-action regions receive higher priority, improving sample diversity.

By combining temporal and spatial priorities, STP encourages sampling experiences that are both recent and non-redundant, improving sample efficiency in off-policy learning.


## Quick start

Run the default experiment:

```bash
python main.py
```

You can also specify the environment and STP hyperparameters:

```bash
python main.py \
  --env HalfCheetah-v4 \
  --seed 10 \
  --alpha 0.2 \
  --k 10 \
  --lambda_ 0.5
```

Evaluation results are saved under `./results/`.

## Repository structure

- `main.py`: training entry point, which runs TD3 + STP by default.
- `STP.py`: STP replay buffer, SumTree, and KMeans-based spatial density estimation.
- `TD3.py`: TD3 actor-critic networks and training logic.

## Requirements

- Python >= 3.8
- Gym <= 0.25
- MuJoCo >= 2.0.2



## Key arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--env` | `HalfCheetah-v4` | Training environment |
| `--seed` | `10` | Random seed |
| `--start_timesteps` | `25000` | Number of initial random exploration steps |
| `--eval_freq` | `5000` | Evaluation frequency |
| `--max_timesteps` | `1000000` | Maximum number of environment steps |
| `--batch_size` | `256` | Training batch size |
| `--alpha` | `0.2` | Spatial-priority clipping parameter |
| `--k` | `10` | Initial number of clusters |
| `--lambda_` | `0.5` | Trade-off factor λ  |
| `--save_plot` | `False` | Whether to save the reward curve |


