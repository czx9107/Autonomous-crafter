# 01-basic_reward 阶段进度总结

## 2026-05-11 补充更新

本次更新保留 2026-05-10 的阶段记录，并在这里追加说明最近一轮主要变化。

### 重要经验

当前实验中有一个很重要的观察：

> 即使要很多步后才能成功，但是 RL 开始训练的阶段已经能反映出问题，甚至可以通过开始阶段的表现来判断后期训练的成功与否。

这意味着后续实验不应只等几百万步后看最终成功率，而要尽早观察前期平均步数、低状态行为、补给时机、睡觉、种树苗、反复横跳等偏置。如果训练早期已经表现出明显错误激励，继续堆时间步往往只是把错误策略训练得更稳定。

### 近期主要改动

- 奖励相关 CLI 参数已经删除，奖励实验现在直接修改 `SurvivalRewardConfig` 和 `get_survival_reward(...)`。
- observation 从旧版单帧输入升级为 history-stacked Dict：
  - `map`: `(4, 9, 9)`
  - `stats`: `(4, 23)`
- `stats` 现在包含 `crafter.constants.items` 中的全部物品和状态，而不是只包含关键物品。
- map 使用 token embedding + CNN；stats 使用 MLP；policy/value head 使用 `256 x 256`。
- action mask 仍然独立于 observation history，只基于当前底层环境状态实时计算。
- 当前算法为 `MaskablePPO("MultiInputPolicy")`。
- 当前默认训练参数包含：
  - `episode_steps = 512`
  - `n_steps = 512`
  - `gamma = 0.997`
  - `ent_coef = 0.01`
  - `clip_range = 0.3`
- 当前奖励设计已经多次调整：
  - `health_delta = 8.0`
  - `food_delta = 4.0`
  - `drink_delta = 4.0`
  - `energy_delta = 0.0`
  - 正向恢复奖励按恢复前状态加权，低状态恢复更值钱，高状态恢复更便宜。
  - `alive_bonus = 0.04`
  - `action_cost = 0.0`
  - `death_penalty = 15.0`
  - `max_steps_bonus = 20.0`
  - `food/drink/health_low_status_penalty = 0.375`
  - `energy_low_status_penalty = 0.0`

### 当前判断

4M 时间步后成功率仍然很低，说明问题不只是训练时间不够。复杂模型可以提高表达能力，但不能单独解决稀疏成功信号和错误奖励几何的问题。

后续更适合优先关注：

- 训练早期行为诊断。
- 低状态惩罚是否仍然过强。
- 是否采用课程学习，例如先训练 128 或 256 步生存，再逐步增加到 512 步。
- reward normalization 只作为数值稳定工具，不应替代奖励语义设计。
- reward 和课程稳定后，再考虑 recurrent policy 或更强记忆结构。

---

## 2026-05-10 阶段记录

## 目标

本阶段围绕 Crafter 环境构建一个强化学习智能体训练入口，重点不是追求 Crafter 原始成就分数，而是先建立一个只关注智能体自身生存状态的 PPO 训练框架。当前实验脚本为 `Crafter/01-basic_reward.py`，演示脚本为 `Crafter/01-demo.py`。

## 环境与代码组织

当前 `Crafter/` 已经被整理成一个相对独立的 Python package：

- `crafter/`：从 Conan playground 迁移出的 Crafter 风格环境代码。
- `01-basic_reward.py`：当前主要训练脚本。
- `01-demo.py`：加载模型权重并用 GUI 演示智能体行为。
- `setup.py`：声明运行、GUI、训练依赖。
- `saves/01-basic_reward/`：默认训练输出目录。

训练依赖使用 conda 环境 `crafter`。训练 extra 已加入 `sb3-contrib`，用于 `MaskablePPO`。

## 奖励设计

当前奖励函数暴露在 `get_survival_reward(...)`，方便后续直接修改。奖励只使用智能体自身生存指标，不使用原 Crafter 成就奖励。

四项核心生存指标：

- `health`
- `food`
- `drink`
- `energy`

当时默认奖励项：

- 生存指标变化奖励：
  - `health_delta = 4.0`
  - `food_delta = 2.0`
  - `drink_delta = 2.0`
  - `energy_delta = 2.0`
- 每步动作成本：`action_cost = 0.01`
- 死亡惩罚：`death_penalty = 2.0`
- 达到最大 episode 步数且仍存活：`max_steps_bonus = 4.0`
- 低生存值惩罚：
  - `low_status_penalty = 0.05`
  - `low_status_threshold = 0.5`
  - `low_status_power = 2.0`

低生存值惩罚形式为：

```text
penalty = -low_status_penalty * ((threshold - level) / threshold) ^ low_status_power
```

原 Crafter 环境的 reward 被关闭并记录为 `base_reward_ignored`，不参与训练。

## Action Mask 设计

当前训练算法已经从普通 PPO 改为 `sb3_contrib.MaskablePPO`。

动作空间仍然固定为 `Discrete(25)`。没有动态改变动作空间，因为 PPO 策略网络输出维度必须固定；如果每一步改变动作空间大小，会破坏 logits、rollout buffer 和模型参数形状。

当前方案是在环境中暴露：

```python
def action_masks(self):
    return get_action_mask(self.env)
```

MaskablePPO 会在采样和训练时读取 mask，使当前非法动作不会被采样。这样效果上等价于“当前只允许合法动作”，但模型输出维度仍保持 25 维。

默认 `invalid_action_penalty` 已改为 `0.0`。非法动作现在主要由 action mask 处理，避免奖励曲线上升主要来自“少做非法动作”。

已特别 review 和修复的 mask 边界：

- `do` 面向 Plant 等未处理对象时不再误判为合法。
- 睡眠中且 `energy` 未满时，环境会强制执行 `sleep`，因此 mask 只允许 `sleep`。
- 随机按 mask 采样 50 step，验证 `invalid=0`。

## Agent 输入与输出

当时 agent 输入是一个 `float32` 向量，shape 为 `(93,)`。

输入由两部分组成：

1. 局部 symbolic map：
   - 原 Crafter symbolic observation 展平。
   - 按 `/255.0` 归一化。

2. 额外状态特征：
   - 生存指标：`health, food, drink, energy`
   - 关键背包物品：`apple, beef, steak, bucket, water_bucket, bed, wood, stone`
   - 每项按其最大值归一化。

Agent 输出仍为 25 个离散 Crafter 动作之一：

- `noop`
- `move_left`
- `move_right`
- `move_up`
- `move_down`
- `do`
- `sleep`
- `place_*`
- `make_*`
- `eat_*`
- `drink`

MaskablePPO 会在输出采样前屏蔽当前不可执行的动作。

## 训练架构

当时训练脚本支持：

- `MaskablePPO(MlpPolicy)`
- `--n-envs` 并行环境数量，默认 8
- `--vec-env subproc|dummy`
- `SubprocVecEnv` 并行训练
- `VecMonitor`/`Monitor` 记录训练统计
- `--episode-steps` 控制每个 episode 最大步数
- `--log-interval` 控制 Stable Baselines 表格打印频率
- `--render-every` 控制训练中探针渲染频率
- `--resume-from` 断点续训
- 续训时检查显式传入的 PPO 超参数是否与 checkpoint 冲突

并行环境的 seed 设计为：

```text
env_rank_seed = base_seed + rank
```

这样同一轮训练可复现，同时不同子环境不会完全重复。

## 渲染探针

训练中的 render 探针来自真实训练 episode，不再额外跑单独推理 episode。

并行训练时 render 计数使用主进程维护的全局 episode 编号。文件名包含 env 编号，避免多子环境同一 timestep 保存时冲突：

```text
episode_0001_env_00_step_20.gif
episode_0002_env_01_step_20.gif
```

后续优化中，render 已改为只渲染命中的 env，而不是每次渲染所有子环境，减少并行训练开销。

## 存档与断点续训

每次运行训练前，会把旧输出归档到：

```text
saves/01-basic_reward/old/
```

归档内容包括：

- csv 日志
- renders
- 模型 zip

归档包数量默认最多保留 3 个。断点续训流程已经调整为先加载 checkpoint，再归档旧输出，避免 resume 权重先被打包走导致无法加载。

续训时，如果命令行显式指定了与 checkpoint 保存值冲突的关键超参数，会直接报错，而不是静默覆盖。

## Demo 脚本

`01-demo.py` 用于加载训练好的模型并 GUI 演示。

当前支持：

- 优先按 `MaskablePPO` 加载新模型。
- 推理时传入 `action_masks=env.action_masks()`。
- 若环境缺少 `sb3_contrib` 或模型为旧 PPO 权重，则尝试按普通 PPO 加载。

注意：旧 PPO 权重没有 action mask 约束；新 MaskablePPO 权重在 demo 中会使用 mask。

## 对 Crafter 环境的改动与封装

主要环境改动集中在 wrapper 层 `BasicSurvivalRewardEnv`：

- 关闭原 Crafter reward。
- 重写 reward 为 survival-only reward。
- 将 Gym/Gymnasium step 接口适配为：
  - `terminated`
  - `truncated`
  - `info`
- 暴露 `action_masks()`。
- 暴露恢复事件探针打印，例如饮水、吃东西、睡觉导致的生存值恢复。
- observation 从纯 symbolic map 扩展为 map + 生存值 + 关键背包物品。

Crafter package 本身也做过必要兼容修复：

- `ruamel.yaml` 新 API 兼容。
- `render(size=None)` 的 size 处理修复。

## 已完成验证

已跑通的关键验证：

```text
conda run -n crafter python -m py_compile 01-basic_reward.py 01-demo.py
conda run -n crafter python 01-basic_reward.py --smoke-test
conda run -n crafter python 01-basic_reward.py --total-timesteps 40 --episode-steps 10 --n-steps 10 --batch-size 20 --n-envs 2 --vec-env subproc --render-every 0
conda run -n crafter python 01-basic_reward.py --total-timesteps 40 --episode-steps 10 --n-steps 10 --batch-size 20 --n-envs 2 --vec-env subproc --render-every 1
```

额外验证：

- `sb3_contrib` 可导入，版本为 `2.8.0`。
- `MaskablePPO + SubprocVecEnv` 短训练通过。
- `MaskablePPO + n_envs=1` 自动 `DummyVecEnv` 路径通过。
- 保存后的 MaskablePPO 权重可用 `action_masks` 正常 predict。
- 随机按 mask 采样 50 step，未出现 invalid action。

## 当时阶段结论

当时已经形成一个较干净的 survival-only RL 训练基线：

- 奖励不再依赖 Crafter 成就。
- 输入包含局部地图、生存状态和关键资源。
- 输出仍为 25 个固定动作。
- 非法动作通过 MaskablePPO action mask 屏蔽。
- 并行训练、断点续训、训练中渲染和模型演示都已打通。

接下来更适合关注奖励尺度、低状态惩罚强度、episode 长度、并行数和学习率等训练参数，而不是继续让 agent 先花大量样本学习动作合法性。
