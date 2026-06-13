# STP: Spatiotemporally Prioritized Experience Replay

本仓库实现了论文 **Spatiotemporally prioritized experience replay for off-policy reinforcement learning** 中提出的 **STP**（Spatiotemporally Prioritized Experience Replay），并将其接入 **TD3**，用于连续动作空间的离策略强化学习任务。

## 论文要点

STP 的核心是为每条经验同时分配两类优先级：

- **时间优先级**：越新的经验优先级越高，用于减少早期经验被过度采样。
- **空间优先级**：位于稀疏状态-动作区域的经验优先级更高，用于提升样本多样性。

通过融合这两种优先级，STP 让回放池更倾向于采样“新颖且不冗余”的经验，从而提升离策略算法的样本利用效率。

## 代码结构

- `main.py`：训练入口，默认在 `HalfCheetah-v4` 上运行 TD3 + STP。
- `STP.py`：STP 回放池、SumTree，以及基于 KMeans 的空间密度估计。
- `TD3.py`：TD3 的 Actor / Critic 网络与训练逻辑。

## 环境依赖

- Python 3
- PyTorch
- NumPy
- Matplotlib
- Gym
- MuJoCo 环境

> 说明：代码使用的是旧版 Gym 接口风格，运行时需要与对应版本的 Gym / MuJoCo 配套。

## 运行方式

安装依赖后直接运行：

```bash
python main.py
```

也可以指定环境和 STP 参数：

```bash
python main.py \
  --env HalfCheetah-v4 \
  --seed 10 \
  --ip 0.2 \
  --k 10 \
  --save_plot
```

训练过程中的评估结果会保存到 `./results/`。

## 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--env` | `HalfCheetah-v4` | 训练环境 |
| `--seed` | `10` | 随机种子 |
| `--start_timesteps` | `25000` | 随机探索阶段步数 |
| `--eval_freq` | `5000` | 评估频率 |
| `--max_timesteps` | `1000000` | 最大训练步数 |
| `--batch_size` | `256` | 训练批大小 |
| `--ip` | `0.2` | 空间优先级剪裁参数 |
| `--k` | `10` | 初始聚类数 |
| `--save_plot` | `False` | 是否保存曲线图 |

## 论文中的实验环境

论文主要在以下 MuJoCo 连续控制任务上验证 STP：

- Ant
- HalfCheetah
- Hopper
- Humanoid
- Walker2d

## 参考

论文标题：

**Spatiotemporally prioritized experience replay for off-policy reinforcement learning**

如需引用，请以论文正式发表版本为准。
