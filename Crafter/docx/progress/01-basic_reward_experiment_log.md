# 01-basic_reward 实验记录

## 实验基线说明

本文档用于记录 `Crafter/01-basic_reward.py` 的训练实验变化。实验记录只描述与训练结果解释相关的参数、奖励、观测和行为现象；较完整的阶段性背景见 `01-basic_reward_progress.md`。

当前实验初始状态以代码中的 `01-basic_reward.py` 为准，并特别固定以下四个参数：

| 参数 | 初始值 | 位置 |
| --- | ---: | --- |
| `PPO_ENT_COEF` | `0.02` | PPO policy 熵系数 |
| `PPO_CLIP_RANGE` | `0.20` | PPO clip range |
| `SurvivalRewardConfig.rate` | `1.0` | survival potential 权重基准 |
| `SurvivalRewardConfig.alive_bonus` | `0.04` | 每个非死亡存活步奖励 |

调整前的初始状态效果很差。早期训练没有看到稳定改善迹象，表现为 `ep_len_mean` 没有稳定上升、`ep_rew_mean` 长期处于明显负值、`success_rate` 很低，并且 agent 仍未形成稳定的食物管理、怪物规避和长期生存策略。

## 记录 1：将初始状态调整为当前代码状态

时间：2026-05-11

### 调整内容

本次记录从初始基线状态调整到更激进实验状态的四个关键参数变化：

| 参数 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `PPO_ENT_COEF` | `0.02` | `0.00` |
| `PPO_CLIP_RANGE` | `0.20` | `0.30` |
| `SurvivalRewardConfig.rate` | `1.0` | `2.0` |
| `SurvivalRewardConfig.alive_bonus` | `0.04` | `0.20` |

### 调整动机

这次调整的主要动机是测试：让生存状态相关的训练信号更激进一些，是否能够帮助 agent 更快学到食物管理、怪物规避和长期生存策略。

具体来说：

- 将 `PPO_ENT_COEF` 从 `0.02` 降到 `0.00`，减少熵正则对策略选择的干扰，观察更确定的策略是否能更快利用生存信号。
- 将 `PPO_CLIP_RANGE` 从 `0.20` 提高到 `0.30`，允许 PPO 做更大幅度的策略更新。
- 将 `rate` 从 `1.0` 提高到 `2.0`，增强 health/food/drink 的状态 potential 信号。
- 将 `alive_bonus` 从 `0.04` 提高到 `0.20`，显著增强“多活一步”的直接奖励。

### 当前待观察指标

- `ep_len_mean` 是否能稳定上升，而不是在高奖励短 episode 和低奖励长 episode 间交替。
- `ep_rew_mean` 是否与生存长度更一致。
- `success_rate` 是否在早期就出现持续改善迹象。
- `entropy_loss` 是否下降过快。
- `approx_kl` 和 `clip_fraction` 是否仍偏高。
- 死因中饿死、缺水死亡、怪物击杀的比例是否变化。

### 当前判断

初步结果看，这种调整有一点改善：训练中至少出现了稀疏成功，而不是完全无法成功。但改善幅度仍然有限，agent 还没有形成稳定的食物管理、怪物规避和长期生存策略。

同时，第一次调整后的 `clip_fraction` 仍然很大，说明 PPO 更新仍然偏激进，策略变化幅度较大。也就是说，稀疏成功的出现不能完全说明该组参数稳定有效，还需要继续关注 `clip_fraction`、`approx_kl` 和策略是否过快滑向局部行为。

仅仅把生存状态信号调得更激进一点，还不足以解决当前问题。

## 记录 2：降低 alive bonus，并减缓 PPO 更新

时间：2026-05-11

### 调整内容

| 参数 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `SurvivalRewardConfig.alive_bonus` | `0.20` | `0.12` |
| `batch_size` | `256` | `512` |
| `learning_rate` | `3e-4` | `1e-4` |
| `n_epochs` | `10` | `5` |

### 调整动机

激进版本中，agent 又学会了持续种树。这个现象说明较高的 `alive_bonus` 可能仍然在鼓励 agent 通过无意义动作拖延时间，而不是主动学习食物管理、怪物规避和有效探索。

本次将 `alive_bonus` 稍微降低到 `0.12`，希望减少“只要活着就有足够收益”的诱导，让 agent 更有压力区分有效动作和无用动作。

同时，第一次调整后的 `clip_fraction` 仍然很大，说明 PPO 在同一批 rollout 数据上的策略更新仍偏激进。因此本次把 `batch_size` 增大到 `512`，将 `learning_rate` 降到 `1e-4`，并将 `n_epochs` 降到 `5`，希望让每次更新更平滑，减少策略过快滑向局部行为。

### 当前待观察指标

- `place_plant` 或种树相关行为是否下降。
- `ep_len_mean` 是否还能维持或上升，而不是因为 alive bonus 降低后明显缩短。
- `ep_rew_mean` 是否减少和生存长度的冲突。
- `clip_fraction` 和 `approx_kl` 是否下降。
- 死因比例是否从怪物击杀或饥饿进一步恶化。
- 稀疏成功是否仍然存在。

### 结果与当前判断

第二次调整完全失败。最终策略表现为完全不移动。

因此当前代码已回滚到第一次调整状态：

- `SurvivalRewardConfig.alive_bonus = 0.20`
- `batch_size = 256`
- `learning_rate = 3e-4`
- `n_epochs = 10`

这说明第二次调整同时降低 alive bonus 和降低 PPO 更新强度，可能把训练推向了“保守但无探索/无行动”的局部策略。后续不应再把这些变量一次性同向改动，应拆分实验。

## 记录 3：只调整 alive bonus，保持更新幅度不变

时间：2026-05-11

### 调整内容

本次在第一次调整的 PPO 更新参数基础上，只调整存活奖励：

| 参数 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `SurvivalRewardConfig.alive_bonus` | `0.20` | `0.04` |
| `batch_size` | `256` | `256` |
| `learning_rate` | `3e-4` | `3e-4` |
| `n_epochs` | `10` | `10` |

### 调整动机

第二次调整同时降低了 `alive_bonus` 并减弱了 PPO 更新幅度，结果完全失败，最终策略表现为不移动。因此这次做更干净的消融实验：只削弱存活奖励，不改变 PPO 更新强度。

当前怀疑是：较高的 `alive_bonus` 可能给了站桩 `do grass`、种树和其他无意义拖步行为过强的近端正反馈。将 `alive_bonus` 降到 `0.04` 后，可以观察这些行为是否明显减少。

### 当前待观察指标

- 站桩 `do grass` 和种树行为是否下降。
- 移动动作占比是否上升。
- `ep_len_mean` 是否明显下降。
- 稀疏成功是否仍然存在。
- 饿死、渴死、怪物袭击而死的比例如何变化。

### 当前判断

这是对 `alive_bonus` 是否导致站桩策略的因果验证。若站桩行为明显减少，说明存活奖励确实是主要诱因；若策略仍然站桩，则问题更可能来自 action mask、局部地图观察、`do grass` 的环境收益或探索结构。

## 记录 4：降低怪物生成率

时间：2026-05-11

### 调整内容

| 参数 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `MONSTER_SPAWN_RATE` | `1.0` | `0.0` |
| `learning_rate` | `3e-4` | `2e-4` |

### 调整动机

当前怪物击杀已经成为主要死亡原因。由于离出生点越远通常越危险，agent 学到的保守策略可能是合理反应：少移动可以降低短期遇怪风险，但也会错失寻找水源和食物的机会。

本次将怪物生成率暴露为顶部超参数，并把默认训练值设为 `0.0`。`1.0` 表示原始怪物生成率，`0.0` 表示不生成新的 Zombie/Skeleton。这样可以先验证：在没有怪物压力时，当前奖励函数是否足以驱动 agent 学会找水、找食物和维持长期生存。

同时将 `learning_rate` 从 `3e-4` 调整为 `2e-4`，让更新幅度介于第一次调整的激进更新和第二次调整的保守更新之间。

### 当前待观察指标

- 怪物袭击死亡占比是否降到接近 0。
- `ep_len_mean` 是否上升。
- 移动和探索行为是否增加。
- 饿死/渴死占比是否因为探索改善而下降，或因为怪物压力移除后暴露出资源管理问题。
- 稀疏成功率是否比原始怪物生成率下更稳定。

### 当前判断

这是一个课程学习方向的调整：先完全移除怪物压力，验证奖励函数本身是否能学出基础生存策略；如果无怪物环境仍无法稳定生存，说明主要问题不在怪物，而在奖励、观察或探索结构。若无怪物环境可学，再逐步把 `MONSTER_SPAWN_RATE` 从 `0.0` 提升回 `1.0`。

## 记录 5：增强 episode stats 输出，并降低 clip range

时间：2026-05-11

### 调整内容

| 参数/功能 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `PPO_CLIP_RANGE` | `0.30` | `0.20` |
| `learning_rate` | `2e-4` | `1.5e-4` |
| episode stats 输出 | 进食次数、死因占比 | 增加饮水次数、无进食且无饮水 episode 占比，并改为多行输出 |

### 调整动机

此前只统计进食次数和死因占比，难以判断 agent 是否真正学会了饮水，也难以区分“没有补给行为”与“补给行为无效”。本次在每次 render/report 时增加：

- 每个 episode 平均饮水次数。
- 既没有饮水也没有进食的 episode 占比。
- 更清晰的多行 `[episode_stats]` 输出。

同时，第一次激进调整后曾观察到 `clip_fraction` 很大，说明 PPO 策略更新偏猛。本次将 `PPO_CLIP_RANGE` 从 `0.30` 降到 `0.20`，并将 `learning_rate` 从 `2e-4` 降到 `1.5e-4`，希望把更新幅度调到更温和的中间档，减少过快滑向局部行为。

### 当前待观察指标

- `recent_avg_drinks` 是否上升。
- `recent_no_eat_no_drink_rate` 是否下降。
- `clip_fraction` 和 `approx_kl` 是否下降。
- 降低 clip range 后，稀疏成功是否仍能出现。
- 无怪物环境下，agent 是否能通过进食/饮水维持更长 episode。

## 记录 6：修正饮水统计，并增大 batch size

时间：2026-05-11

### 调整内容

| 参数/功能 | 调整前 | 调整后 |
| --- | ---: | ---: |
| 饮水统计 | 只统计 `drink` 消耗 `water_bucket` | 统计所有 `drink` 值上升，包括对水源执行 `do` |
| episode stats 补给表 | 单独输出无进食且无饮水占比 | 输出进食/饮水 2x2 比例表 |
| `batch_size` | `256` | `1024` |

### 调整动机

饮水统计原先只通过 `water_bucket` 是否减少判断，因此会漏掉一种关键情况：agent 面对水源执行 `do` 时，环境会直接让 `drink` 上升，但不会消耗 `water_bucket`。本次将饮水统计改为只要 `drink` 值上升就计为一次成功饮水，从而同时覆盖：

- 对水源执行 `do`。
- 使用 `water_bucket` 执行 `drink`。

同时，将 `batch_size` 大幅增至 `1024`，希望每次梯度更新使用更多样本，降低 minibatch 噪声，让策略更新更稳定。

本次还把补给行为统计改成 2x2 比例表，行表示是否进食，列表示是否饮水：

```text
rows=eat(no,yes), cols=drink(no,yes)
[no eat + no drink, no eat + drink]
[eat + no drink,    eat + drink]
```

### 当前待观察指标

- `recent_avg_drinks` 是否更准确反映实际饮水行为。
- 2x2 表格中 `no eat + no drink` 占比是否下降。
- `eat + drink` 占比是否上升。
- `clip_fraction` 是否因更大的 batch size 更稳定。
- 策略是否减少大幅来回摆动。

## 记录 7：重新开启轻量低状态惩罚

时间：2026-05-11

### 调整内容

| 参数 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `enable_low_status_penalty` | `0` | `1` |
| `food_low_status_penalty` | `0.0/关闭` | `0.05` |
| `drink_low_status_penalty` | `0.0/关闭` | `0.05` |
| `health_low_status_penalty` | `0.0/关闭` | `0.05` |
| `energy_low_status_penalty` | `0.0` | `0.0` |
| `food_low_status_threshold` | `0.6` | `0.6` |
| `drink_low_status_threshold` | `0.6` | `0.6` |
| `health_low_status_threshold` | `0.75` | `0.75` |
| `low_status_power` | `2.0` | `2.0` |

### 调整动机

30w 时间步以后，平均进食/饮水次数反而走低，并且在进食饮水次数少的 rollout 中，奖励没有明显下降。这说明当前 reward 没有稳定地把“缺少补给行为”转化为更差的优势信号。

因此本次重新开启轻量低状态惩罚，让 food、drink、health 进入低状态时产生连续压力。目标不是强行惩罚长 episode，而是让 agent 更早感受到 food/drink/health 变差的风险，从而增加补给和规避危险的动机。

energy 低状态惩罚仍保持 `0.0`，避免重新诱导睡觉相关策略。

### 当前待观察指标

- 2x2 补给表中 `no eat + no drink` 占比是否下降。
- `recent_avg_eats` 和 `recent_avg_drinks` 是否上升。
- 进食/饮水少的 rollout 是否开始出现更低 reward。
- `ep_len_mean` 是否因为低状态惩罚重新开启而下降。
- food/drink 相关死亡占比是否下降。

## 记录 8：加入显式局部 memory map

时间：2026-05-11

### 调整内容

| 参数/功能 | 调整前 | 调整后 |
| --- | --- | --- |
| observation 输入 | `map` + `stats` | `map` + `memory` + `stats` |
| `memory` 形状 | 无 | `(6, 32, 32)` |
| 特征提取器 | 地图 CNN + 状态 MLP | 地图 CNN + memory CNN + 状态 MLP |
| PPO 前共享特征维度 | `192` | `256` |

### 调整动机

当前 agent 的局部观察窗口只有 9x9，即使叠加历史帧，也主要保存短期运动痕迹，不能稳定记住已经见过的水源、食物、障碍物和危险区域。对于 Crafter 这种需要在地图中寻找资源并返回资源点的任务，缺少空间记忆会让策略很容易表现为短视搜索、局部徘徊或重复无效动作。

本次加入一个以玩家为中心裁剪的显式 memory map。环境在每个 step 把当前 9x9 视野写入全局记忆，再裁剪出 32x32 的局部记忆交给策略网络。memory map 当前包含 6 个通道：

- seen：该格是否曾被观察到。
- visited：玩家是否曾到达过该格。
- water：已观察到的水源位置。
- food：已观察到的 apple tree / cow 等食物来源。
- hostile：已观察到的 zombie / skeleton。
- obstacle：已观察到的不可通行格子。

### 当前 observation 大小

```text
map:    (4, 9, 9)      = 324 个离散 token
memory: (6, 32, 32)    = 6144 个 float
stats:  (4, 23)        = 92 个 float
raw total             = 6560 个输入元素
```

网络内部会先把 `map` token 做 embedding，因此地图分支进入 CNN 前的实际张量通道数是 `4 * 8 = 32`，形状等价于 `(32, 9, 9)`。三路特征最终拼接为：

```text
map CNN features:    128
memory CNN features: 64
stats MLP features:  64
total features:      256
```

### 当前待观察指标

- 发现水源或食物后，agent 是否能更稳定地回到相关区域。
- 2x2 补给表中 `eat + drink` 占比是否上升。
- `no eat + no drink` 占比是否下降。
- `ep_len_mean` 和成功率是否比无 memory map 时更稳定。
- 训练速度是否因 memory CNN 和更大的 observation 明显下降。

### 阶段观察

截至约 500000 time steps，加入 memory map 后效果尚可，agent 处于稳步学习状态，成功率约为 `5.5%`。这说明显式地图记忆至少初步缓解了“发现周边没有任何食物来源后原地摆烂”的问题：agent 有机会利用 seen/visited/food/water 等通道判断当前区域资源不足，并尝试向未探索区域移动。

暂不加入 frontier bonus、首次访问 bonus、已访问区域惩罚等直接探索奖励。当前项目目标是构建具有自主探索能力的智能体，后续更希望通过 meta reward，例如环境压力、社会评价等更高层反馈，驱动 agent 主动探索。如果现在加入手工探索奖励，会把“主动探索能力”部分提前写进奖励函数，削弱后续实验对 meta reward 的解释力。

### 后续诊断假设

测试 memory map 版本时需要重点观察：agent 是否会保守地待在水边，而不是继续探索寻找食物。如果出现这种行为，可能说明当前 `rate` 下的状态恢复/状态变化奖励仍然过强，agent 更倾向于维持 drink 安全区的局部最优；同时 food 低状态压力不够早、不够强，导致离开水边寻找食物的长期收益没有清晰反映到训练信号中。

下一轮可考虑：适当降低 `rate`，减少单次状态变化带来的即时吸引力；同时提高 food/drink/health 的低状态惩罚，让资源短缺的风险更早进入优势估计。但这一步应在 memory map 单独测试后再做，避免同时改动过多因素。

## 记录 9：降低状态变化奖励，后移低状态阈值

时间：2026-05-11

状态：当前最优基线

### 调整内容

| 参数 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `SurvivalRewardConfig.rate` | `2.0` | `1.5` |
| `health_delta` | `4.0` | `3.0` |
| `food_delta` | `2.0` | `1.5` |
| `drink_delta` | `2.0` | `1.5` |
| `energy_delta` | `0.0` | `0.0` |
| `food_low_status_penalty` | `0.05` | `0.10` |
| `drink_low_status_penalty` | `0.05` | `0.10` |
| `health_low_status_penalty` | `0.05` | `0.10` |
| `energy_low_status_penalty` | `0.0` | `0.0` |
| `food_low_status_threshold` | `0.6` | `0.4` |
| `drink_low_status_threshold` | `0.6` | `0.4` |
| `health_low_status_threshold` | `0.75` | `0.6` |

### 调整动机

memory map 版本虽然已经能稳步学习，但仍可能出现保守地待在水边、不主动探索食物来源的问题。上一版将 `rate` 降到 `1.0` 并提高低状态惩罚后效果不好，因此本记录直接覆盖为新的折中方案。

本次将 `rate` 从 `2.0` 降到 `1.5`，适度降低 health/food/drink 的 potential-based 状态变化奖励，减少“守着水源刷稳定状态”的即时吸引力。同时保留较强的 food/drink/health 低状态惩罚权重，但把 food/drink/health 的触发阈值分别后移到 `0.4/0.4/0.6`，避免过早惩罚导致 agent 变得过度保守。energy 仍不加低状态惩罚，避免重新诱导睡觉策略。

### 当前待观察指标

- agent 是否减少长期停留水边的行为。
- 离开水源后的探索距离是否增加。
- `recent_avg_eats` 是否上升。
- 2x2 补给表中 `eat + drink` 占比是否上升。
- `ep_len_mean` 是否因低状态惩罚增强而明显下降。
- 成功率是否继续高于 memory map 加入前的水平。

### 阶段结果

早期观察中，本次策略看起来没有提升成功率，并且让 agent 表现得更加保守，因此曾判断不适合作为后续起始点。后续更长训练显示该判断过早：到约 `1536` 个 episode、`492136` time steps 附近，`recent_success_rate` 达到 `0.156`，`total_success_rate` 达到 `0.042`，说明该策略存在延迟起势现象，值得继续训练观察。

关键阶段指标：

```text
training_episode=1536
timesteps=492136
recent_success_rate=0.156
total_success_rate=0.042
ep_len_mean=342
ep_rew_mean=-24.7
recent_avg_eats=1.078
recent_avg_drinks=4.891
recent_death_ratios=starvation:0.593,dehydration:0.407,monster_attack:0.000
approx_kl=0.0050
clip_fraction=0.0556
```

当前重新将本策略作为主线继续训练，重点观察成功率是否能在更长时间步上稳定维持或继续上升。

### 当前最优标记

截至记录 16，本记录仍是当前最优实验基线。它在约 `1536` 个 episode、`492136` time steps 附近达到过 `recent_success_rate=0.156`，明显优于后续 rate-only、balance potential 等消融尝试。

后续实验应优先以本记录参数为起点，只做单变量修改，避免同时改变奖励尺度、低状态惩罚、观察结构和探索压力。

## 记录 10：回滚到 memory map 基线并建立 git 起始点

时间：2026-05-11

### 回滚内容

| 参数 | 失败策略 | 回滚后 |
| --- | ---: | ---: |
| `SurvivalRewardConfig.rate` | `1.5` | `2.0` |
| `health_delta` | `3.0` | `4.0` |
| `food_delta` | `1.5` | `2.0` |
| `drink_delta` | `1.5` | `2.0` |
| `food_low_status_penalty` | `0.10` | `0.05` |
| `drink_low_status_penalty` | `0.10` | `0.05` |
| `health_low_status_penalty` | `0.10` | `0.05` |
| `food_low_status_threshold` | `0.4` | `0.6` |
| `drink_low_status_threshold` | `0.4` | `0.6` |
| `health_low_status_threshold` | `0.6` | `0.75` |

### 备注

本次回滚到记录 8 的 memory map 基线：保留显式局部 memory map、action mask、episode stats 和当前 PPO 训练架构，但撤销记录 9 中对奖励尺度和低状态阈值的修改。该版本作为后续实验的 git 起始点。

## 记录 11：只降低 rate，保持低状态惩罚不变

时间：2026-05-12

### 调整内容

| 参数 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `SurvivalRewardConfig.rate` | `2.0` | `1.5` |
| `health_delta` | `4.0` | `3.0` |
| `food_delta` | `2.0` | `1.5` |
| `drink_delta` | `2.0` | `1.5` |
| `food_low_status_penalty` | `0.05` | `0.05` |
| `drink_low_status_penalty` | `0.05` | `0.05` |
| `health_low_status_penalty` | `0.05` | `0.05` |
| `food_low_status_threshold` | `0.6` | `0.6` |
| `drink_low_status_threshold` | `0.6` | `0.6` |
| `health_low_status_threshold` | `0.75` | `0.75` |

### 候选动机

记录 9 曾在早期被误判为整体失败，但后续长训显示其 recent success rate 能达到 `15.6%`。此后继续观察发现，记录 9 虽能在较好地图条件下成功，但后劲不足，并且策略仍存在刻板行为，例如种树苗、单方向移动，以及临近低状态才补给。

因此本次将该候选转为当前实验方案，采用更干净的折中消融：只把 `rate` 从 `2.0` 降到 `1.5`，降低 health/food/drink 的 potential-based 状态变化奖励强度；低状态惩罚权重和阈值保持 memory map 基线不变。目标是单独验证“降低 potential shaping 强度”是否足以保留饱食度回复学习迹象，同时减少记录 9 中较强低状态压力带来的保守性。

### 当前待观察指标

- 是否仍能学到进食和饱食度回复。
- 是否比记录 9 更愿意离开水边探索。
- 成功率是否高于记录 10 的 memory map 基线。
- `recent_avg_eats`、`recent_avg_drinks` 和 2x2 补给表是否改善。
- episode 长度是否保持稳定，而不是转向更短或更保守的策略。

## 记录 12：恢复记录 9 策略并继续长训

时间：2026-05-12

### 当前代码状态

| 参数 | 当前值 |
| --- | ---: |
| `SurvivalRewardConfig.rate` | `1.5` |
| `health_delta` | `3.0` |
| `food_delta` | `1.5` |
| `drink_delta` | `1.5` |
| `food_low_status_penalty` | `0.10` |
| `drink_low_status_penalty` | `0.10` |
| `health_low_status_penalty` | `0.10` |
| `food_low_status_threshold` | `0.4` |
| `drink_low_status_threshold` | `0.4` |
| `health_low_status_threshold` | `0.6` |

### 调整说明

根据记录 9 后续长训结果，当前恢复到回滚前版本：`rate=1.5`，food/drink/health 低状态惩罚为 `0.10`，触发阈值为 `0.4/0.4/0.6`。该版本在约 `512000` total timesteps 时已经出现 `recent_success_rate=0.156`，说明其学习曲线可能需要更长时间才能体现优势。

接下来优先继续训练该版本，不急于切换到记录 11 的 rate-only 候选。记录 11 暂作为后续预计修改队列中的消融方案。

### 后续补充

继续观察后发现，记录 9 虽然在约 `1536` episode 附近达到过 `recent_success_rate=0.156`，并且在好环境条件下能成功，但后劲不足。agent 仍然较难兼顾进食和饮水，并出现种树苗、固定方向移动、临近低状态才补给等刻板行为。因此后续切换到记录 11 的 rate-only 消融方案。

## 记录 13：应用记录 11 的 rate-only 消融

时间：2026-05-12

### 当前代码状态

| 参数 | 当前值 |
| --- | ---: |
| `SurvivalRewardConfig.rate` | `1.5` |
| `health_delta` | `3.0` |
| `food_delta` | `1.5` |
| `drink_delta` | `1.5` |
| `food_low_status_penalty` | `0.05` |
| `drink_low_status_penalty` | `0.05` |
| `health_low_status_penalty` | `0.05` |
| `food_low_status_threshold` | `0.6` |
| `drink_low_status_threshold` | `0.6` |
| `health_low_status_threshold` | `0.75` |

### 调整说明

当前正式应用记录 11 的消融方案：保留 `rate=1.5`，但将 food/drink/health 的低状态惩罚和触发阈值恢复到 memory map 基线。这样可以单独检验降低 potential shaping 强度的效果，而不再叠加记录 9 中更强、更晚触发的低状态压力。

### 当前待观察指标

- 是否能保留记录 9 中出现的进食/饱食度回复学习迹象。
- 是否降低种树苗和固定方向移动等刻板行为。
- 是否改善 food/drink 兼顾能力，而不是只在临近低状态时补给。
- 成功率是否能超过记录 9 的短期峰值，或至少表现出更稳定的后期增长。

### 阶段结果

本次效果很差，agent 又退回到种树相关刻板行为。说明只降低 `rate`、同时把低状态惩罚恢复到较轻基线，并没有缓解当前的后劲不足问题，反而可能削弱了对资源短缺的有效压力。

## 记录 14：调整低状态惩罚并加入 food-drink balance potential

时间：2026-05-12

### 调整内容

| 参数/功能 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `SurvivalRewardConfig.rate` | `1.5` | `1.5` |
| `food_low_status_penalty` | `0.05` | `0.10` |
| `drink_low_status_penalty` | `0.05` | `0.10` |
| `health_low_status_penalty` | `0.05` | `0.10` |
| `food_low_status_threshold` | `0.6` | `0.4` |
| `drink_low_status_threshold` | `0.6` | `0.4` |
| `health_low_status_threshold` | `0.75` | `0.5` |
| `food_drink_balance_delta` | 无 | `0.5` |

### 调整动机

记录 13 中 agent 再次退回种树，说明单纯降低 potential shaping 强度不能解决 food/drink 兼顾问题。当前重新增强低状态压力，但将 health 阈值调低到 `0.5`，避免 health 惩罚过早压制探索。

同时加入轻量 food-drink balance potential：

```text
balance_level = min(food_level, drink_level)
Phi += food_drink_balance_delta * balance_level
```

该项不是额外 step penalty，而是进入 potential-based shaping：

```text
reward += gamma * Phi(s') - Phi(s)
```

因此它表达的是“food 和 drink 的短板越高，整体状态越好”，而不是分别重复惩罚 food 或 drink。目标是缓解 agent 只围绕水源或食物单项资源形成局部策略的问题。

### 当前待观察指标

- `eat + drink` episode 比例是否上升。
- starvation/dehydration 死因是否更均衡且总体下降。
- agent 是否减少种树苗、固定方向移动等刻板行为。
- recent success rate 是否能超过记录 9 的 `0.156` 短期峰值。
- 是否出现新的偏置，例如过度追求短板补给而不探索。

## 记录 15：统一 PPO gamma 与 potential gamma

时间：2026-05-12

### 调整内容

| 参数/功能 | 调整前 | 调整后 |
| --- | --- | --- |
| PPO 折扣因子 | `--gamma` CLI，默认 `0.997` | 顶部常量 `PPO_GAMMA = 0.997` |
| `SurvivalRewardConfig.potential_gamma` | 独立默认 `0.997` | 默认引用 `PPO_GAMMA` |
| resume 固定超参校验 | 检查 `ent_coef/clip_range/n_epochs` | 额外检查 `gamma` |
| CLI 参数 | 可通过 `--gamma` 修改 | 移除 `--gamma` |

### 调整动机

potential-based shaping 的标准形式是：

```text
F(s, s') = gamma * Phi(s') - Phi(s)
```

这里的 gamma 应与 PPO 计算折扣回报时使用的 gamma 保持一致。此前两者默认都为 `0.997`，但分别位于 `SurvivalRewardConfig.potential_gamma` 和 CLI `--gamma`，存在未来只修改其中一边的风险。

本次将 gamma 提升为顶部常量 `PPO_GAMMA`，PPO 训练和 reward potential 共同引用该常量。这样可以避免 potential shaping gamma 与 PPO gamma 分叉，保持 reward shaping 的理论含义更清晰。

## 记录 16：回退到记录 9 参数，保留 balance 代码但关闭

时间：2026-05-12

### 调整内容

| 参数/功能 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `SurvivalRewardConfig.rate` | `1.5` | `1.5` |
| `food_low_status_penalty` | `0.10` | `0.10` |
| `drink_low_status_penalty` | `0.10` | `0.10` |
| `health_low_status_penalty` | `0.10` | `0.10` |
| `food_low_status_threshold` | `0.4` | `0.4` |
| `drink_low_status_threshold` | `0.4` | `0.4` |
| `health_low_status_threshold` | `0.5` | `0.6` |
| `food_drink_balance_delta` | `0.5` | `0.0` |

### 调整动机

记录 14 的 balance potential 效果仍然不理想，agent 没有表现出明显规划能力提升。因此当前回退到记录 9 的奖励参数：`rate=1.5`，food/drink/health 低状态惩罚为 `0.10`，触发阈值为 `0.4/0.4/0.6`。

balance 相关代码暂时保留，但通过 `food_drink_balance_delta=0.0` 关闭实际影响。这样后续如果要重新测试短板项，只需要调整常量，不需要重新改 reward 结构。

### 当前判断

当前主要瓶颈不再像是简单 reward scale 问题，而是 agent 缺少更强的规划能力。后续更可能需要从观察结构、记忆使用方式、动作统计诊断或高层 meta reward 入手，而不是继续单纯增加 food/drink 的局部奖励项。

## 记录 17：加入二维世界坐标输入

时间：2026-05-12

### 调整内容

| 参数/功能 | 调整前 | 调整后 |
| --- | --- | --- |
| `stats` observation | inventory/status，形状 `(4, 23)` | inventory/status + `x/y`，形状 `(4, 25)` |
| 坐标编码 | 无 | `x / (width - 1)`, `y / (height - 1)` |
| CNN/map 输入 | 不变 | 不变 |
| memory map 输入 | 不变 | 不变 |

### 调整动机

当前 agent 出现固定方向移动、局部资源依赖和缺少规划能力的问题。memory map 是以 agent 为中心裁剪的局部记忆，虽然能提供附近已见资源和障碍信息，但缺少全局位置锚点。

本次加入二维归一化世界坐标，让 MLP 分支获得当前位置的全局参考。目标不是告诉 agent 资源在哪里，而是帮助它建立基础方向感，减少固定方向坍缩，并让同样的局部观察在地图不同区域具有可区分性。

### 当前 observation 大小

```text
map:    (4, 9, 9)
memory: (6, 32, 32)
stats:  (4, 25)
```

### 当前待观察指标

- 固定向下移动或单方向移动占比是否下降。
- 资源稀疏地图中是否更愿意改变探索方向。
- 成功率是否高于记录 9 的当前最优峰值。
- 是否出现坐标过拟合，例如只在某些固定区域行动。

## 记录 18：加入 `--new-map` 维护式 token 地图实验

时间：2026-05-12

### 当前调整定位

记录 17 和记录 18 合并为当前阶段的观察结构调整：在记录 9 当前最优奖励参数基础上，加入二维世界坐标，并保留两套地图表示用于分支对照。当前不再继续改 reward，而是比较 observation 结构是否能改善 agent 的方向感、地图记忆利用和规划能力。

当前有两个分支实验待完成：

| 分支 | 启动方式 | observation | 实验目的 |
| --- | --- | --- | --- |
| 分支 A：旧地图 + 坐标 | 默认运行，不加 `--new-map` | `map (4, 9, 9)` + `memory (6, 32, 32)` + `stats (4, 25)` | 测试仅加入 `x/y` 坐标后，旧语义 memory map 是否足以减少固定方向移动和种树刻板行为 |
| 分支 B：新 token map + 坐标 | 加 `--new-map` | `map (1, 17, 17)` + `stats (4, 25)` | 测试由 wrapper 维护的 17x17 token 地图和 unknown token 是否比 6-channel memory 更利于探索和回访资源 |

### 调整内容

默认模式保持不变：

```text
map:    (4, 9, 9)      当前 9x9 视野历史
memory: (6, 32, 32)    wrapper 维护的语义 memory map
stats:  (4, 25)        inventory/status + x/y
```

新增 `--new-map` 模式：

```text
map:   (1, 17, 17)     wrapper 维护的 token 地图裁剪
stats: (4, 25)         inventory/status + x/y
```

`--new-map` 下不再使用旧的 6-channel `memory` 分支。wrapper 维护一张全局 token map，初始全部填充为 `UNKNOWN_MAP_TOKEN_ID`；agent 每步仍然只能看到环境返回的 9x9 视野，wrapper 将这 9x9 写入全局 token map，再以 agent 当前位置为中心裁剪 17x17 作为 observation。

### 设计动机

旧 memory map 是人工压缩后的语义通道：

```text
seen / visited / water / food / hostile / obstacle
```

它信息密度低、先验强，可能丢失了材料和对象的细粒度差异。新 token map 保留环境符号 token，并给未知区域一个独立 token 与 embedding，让网络自己学习未知、已知、资源、障碍之间的关系。

该改动用于比较两种地图表示：

- 旧模式：更强人工归纳偏置，更轻量。
- 新模式：更接近显式地图记忆，信息更完整，但需要网络自己学语义。

### 当前待观察指标

- `--new-map` 是否减少固定方向移动和种树苗刻板行为。
- `--new-map` 是否提升资源稀疏地图中的探索半径。
- 17x17 token map 是否比 6-channel memory 更容易学会回访水源/食物。
- 训练速度是否因 17x17 CNN map 增大而下降。

## 记录 19：增加 visited area 与 reward component 诊断输出

时间：2026-05-12

### 调整内容

每次 render/report 后的 `[episode_stats]` 增加两类诊断：

```text
recent_avg_visited_area / total_avg_visited_area
reward_components avg_per_episode/abs_share
```

visited area 定义为单个 episode 内 agent 走过的不同世界格子数量，即 visited position set 的大小。

reward component 分为四类：

| 类别 | 包含项 |
| --- | --- |
| `terminal` | `death_penalty + max_steps_bonus` |
| `low_status` | `low_status_penalty` |
| `alive` | `alive_bonus` |
| `potential` | `status_potential_reward` |

输出格式为：

```text
terminal=avg_per_episode/abs_share
```

其中 `avg_per_episode` 是该类 reward 在统计窗口内的每 episode 平均有符号贡献，`abs_share` 是该类绝对值贡献占四类绝对值总贡献的比例。使用绝对值占比是为了避免正奖励和负惩罚互相抵消后失去解释性。

### 调整动机

此前只能看到成功率、进食/饮水次数和死因，难以判断 reward 到底由哪些部分主导。加入 reward component 占比后，可以判断策略是否主要靠 `alive_bonus` 和 `max_steps_bonus` 推动，还是确实从 potential-based 状态改善中获得学习信号。

visited area 用于衡量探索范围。若成功率提升但 visited area 很低，说明策略可能仍然是局部保守生存；若 visited area 提升但成功率下降，说明探索增加但资源管理尚未跟上。

### 阶段经验

探针非常重要。当前任务中，仅凭成功率、平均奖励或平均步数，很容易误判策略到底学到了什么。例如记录 9 曾被早期现象误判为失败，但更长训练和更细的 episode stats 显示它其实出现过明显成功率峰值；后续 reward component、visited area、进食/饮水 2x2 表也能帮助区分“真的学会规划”和“只是靠保守策略拖到终局”。

后续实验应优先增加低干扰探针，尽量在不改变训练目标的情况下获取足够精准的信息。只有当探针显示问题来源明确后，再做 reward、observation 或模型结构调整。

## 记录 20：增加四方向移动占比探针

时间：2026-05-12

### 调整内容

每次 render/report 后的 `[episode_stats]` 增加移动模式诊断：

```text
recent_avg_move_actions / total_avg_move_actions
recent_move_ratios left/right/up/down
total_move_ratios left/right/up/down
```

其中 `avg_move_actions` 是统计窗口内每个 episode 平均执行移动动作的次数；`move_ratios` 的分母只包括四个移动动作本身，即：

```text
move_left + move_right + move_up + move_down
```

它不把 `noop`、采集、放置、进食、睡觉等动作放进分母。这样可以单独观察“只要 agent 决定移动，它在四个方向上的分布是否已经固化”。

### 调整动机

当前策略仍可能出现只往下走、在局部区域反复横跳、种树苗后固定移动等刻板模式。四方向移动占比可以和 visited area 一起使用：

- `visited area` 低且单方向占比极高：可能是移动策略坍缩。
- `visited area` 低但四方向较均衡：可能是在局部来回移动或被障碍/资源点限制。
- `visited area` 高但成功率低：可能探索增加了，但进食/饮水规划仍没跟上。

这个探针不改变 reward，只用于识别移动模式是否固化。

## 记录 21：降低 alive bonus 占比

时间：2026-05-12

### 调整内容

| 参数 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `SurvivalRewardConfig.alive_bonus` | `0.04` | `0.01` |

### 调整动机

reward component 探针显示，在终局项已经占据约一半绝对奖励贡献的情况下，`alive` 项仍然能达到约 `20%` 的贡献，占比偏高。这样的奖励结构容易鼓励 agent 追求“持续活着并拖步”，而不是主动寻找食物、水源或扩大探索范围。

本次只降低 `alive_bonus`，不改 potential-based 状态奖励、低状态惩罚、终局惩罚或 observation 结构。这样可以更干净地观察存活步奖励是否是导致保守策略、种树拖步和移动模式固化的主要诱因之一。

### 当前待观察指标

- `alive` 在 reward component 中的绝对占比是否明显下降。
- `visited area` 和四方向移动占比是否改善。
- 平均 episode 步数是否明显下降；若下降过多，说明 `alive_bonus` 仍承担了稳定训练的作用。
- 进食/饮水 2x2 表是否改善，尤其是 `eat_drink` episode 比例是否上升。

## 记录 22：扩大 new-map token 地图窗口到 25x25

时间：2026-05-12

### 调整内容

| 参数 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `NEW_MAP_SIZE` | `17` | `25` |

本次只影响 `--new-map` 分支。默认 oldmap 分支仍保持：

```text
map:    (4, 9, 9)
memory: (6, 32, 32)
stats:  (4, 25)
```

`--new-map` 分支改为：

```text
map:   (1, 25, 25)
stats: (4, 25)
```

当前代码中的 `alive_bonus` 已恢复为 `0.04`，本次不再改动 reward，仅扩大 wrapper 维护的 token map 裁剪范围。

### 调整动机

`17x17` new-map 测试中，方向占比反而更不均衡，成功也比较稀疏。一个可能原因是新地图分支需要更多数据才能学会 token 语义；另一个可能原因是 `17x17` 仍不足以让 agent 同时看到水源、食物、障碍和自己移动轨迹之间的空间关系。

因此将 token map 窗口扩大到 `25x25`，让 agent 在不改变真实视野 `9x9` 的前提下获得更大的历史地图裁剪。这个实验用于测试更大的显式记忆范围是否能改善探索半径、资源回访和方向固化问题。

### 当前待观察指标

- `recent_avg_visited_area` 是否高于 `17x17` new-map。
- 四方向 `move_ratios` 是否比 `17x17` 更均衡。
- `eat_drink` episode 比例是否提升。
- 成功率是否从稀疏成功变为更稳定上升。
- 训练速度是否因为 map CNN 输入从 `17x17` 增至 `25x25` 明显下降。

## 记录 23：当前剩余核心问题：方向偏好与不探索

时间：2026-05-12

### 当前现象

截至 `oldmap`、`newmap 17x17`、`newmap 25x25` 的对照观察，当前最稳定暴露出的两个问题是：

1. 方向偏好明显。

   四方向移动占比探针显示，agent 在较早训练阶段就可能形成明显单方向或轴向偏好。例如曾观察到：

   ```text
   recent_move_ratios left/right/up/down:
     0.077/0.215/0.505/0.203
   ```

   这说明 agent 并非只是随机探索，而是在移动策略上产生了偏置。若同时 `visited_area` 不高，则更可能是移动模式固化或碰壁后重复尝试同一方向。

2. 不探索或探索不足。

   即使加入 explicit memory map、世界坐标和 new-map token map，agent 仍可能保守停留在局部区域，例如水边、出生点附近、草地/树苗附近。`newmap` 并没有立刻改善该问题，反而可能因为输入更复杂、数据不足或 reward 信号较弱而更难训练。

### 当前判断

这两个问题可能是同一件事的两个表现：agent 没有学到“探索本身能够提高未来生存机会”的可用价值函数，只学到了一些局部短期行为，例如靠近水源、重复采集/种植、沿某个方向移动或少动。当前奖励虽然只围绕自身生存指标设计，但 survival reward 的反馈仍然偏稀疏，且食物/水源的收益通常延迟很多步才体现出来。

`alive_bonus` 的对照说明：过低会让长 episode 更难维持，过高又可能鼓励拖步。`potential` 项也有隐含时间惩罚：在 `gamma < 1` 时，即使状态不变，`gamma * Phi(s') - Phi(s)` 也会产生负项。因此 agent 可能同时受到“活得久有奖励”和“活得久也持续扣 potential”的混合信号，学习压力比较混乱。

### 优先解决方向

#### 方向一：继续增强诊断，而不是马上加探索奖励

下一步最稳妥的是继续加低干扰探针，用于区分几种不同失败模式：

| 探针 | 作用 |
| --- | --- |
| `net_dx/net_dy` | 判断 episode 终点相对起点是否有持续方向漂移 |
| `start_to_end_distance` | 区分真实探索和局部来回抖动 |
| `visited_area / move_actions` | 衡量每次移动带来的新区域发现效率 |
| `blocked_move_count` | 判断方向偏好是否来自反复撞墙或 action mask 未屏蔽障碍方向 |
| `unknown_area_ratio` | 对 new-map 统计局部地图中 unknown token 占比，判断 agent 是否仍待在已知区域 |

其中最建议优先加入 `net_dx/net_dy` 与 `visited_area / move_actions`。它们不改变 reward，却能判断方向偏好到底是“有方向地远离出生点”，还是“在局部固化”。

#### 方向二：拆分 potential 诊断

当前 reward component 中的 `potential` 应继续拆开：

```text
potential_decay = -(1 - gamma) * Phi(s')
potential_delta = Phi(s') - Phi(s)
```

这样可以判断 `potential` 负项到底来自状态真实恶化，还是来自 `gamma < 1` 带来的隐含时间惩罚。如果主要来自 decay，就不应该继续简单调大/调小 food、drink 权重，而应考虑是否将这部分单独显式化，或者改用更接近 `Phi(s') - Phi(s)` 的状态变化项做实验。

#### 方向三：先不直接加入探索内在奖励

由于项目目标是构建具有自主探索能力的智能体，后续可能要引入 meta reward，例如环境压力、社会评价等。因此当前不宜直接添加 visited bonus、unknown bonus 这类强手工探索奖励，否则会提前把“探索动机”写死进环境。

但可以先通过探针记录探索行为，然后在后续 meta reward 阶段把这些统计作为诊断指标或候选输入，而不是现在直接作为 reward。

#### 方向四：观察结构继续小步对照

`newmap 25x25` 仍值得跑一段，但需要接受它可能比 oldmap 更慢起效。若 `25x25` 仍方向更不均衡且成功稀疏，则说明更大地图窗口本身不能解决不探索问题。此时优先回到 oldmap 当前最优设置，再处理 reward/potential 与诊断拆分，而不是继续盲目增大地图。

### 下一步建议

下一步不要再同时改 reward、map size 和 PPO 超参。建议先做一个诊断型改动：

1. 加入 `net_dx/net_dy`、`start_to_end_distance`、`visited_area / move_actions` 三个 episode stats。
2. 加入 `potential_decay` 与 `potential_delta` 的 reward component 拆分。
3. 用 `oldmap + alive_bonus=0.04` 和 `newmap 25x25 + alive_bonus=0.04` 分别跑短程对照。

如果 oldmap 明显更稳定，则 newmap 暂时降级为备选分支；如果 newmap 的 `visited_area / move_actions` 明显更高但成功率低，则说明地图确实促进探索，但补给规划和 reward 信号仍需处理。

## 记录 24：移除 survival potential 中的 gamma

时间：2026-05-12

### 调整内容

| 项目 | 调整前 | 调整后 |
| --- | --- | --- |
| PPO 折扣因子 | `PPO_GAMMA = 0.997` | 保持不变 |
| survival potential reward | `gamma * Phi(s') - Phi(s)` | `Phi(s') - Phi(s)` |
| `SurvivalRewardConfig.potential_gamma` | 存在，引用 `PPO_GAMMA` | 删除 |

本次只改变 survival reward 中状态 potential 的计算方式，不改变 PPO 自身的折扣因子。PPO 仍使用 `gamma=0.997` 计算回报和优势。

代码中 `status_potential_reward` 现在等于：

```text
status_potential_delta = Phi(s') - Phi(s)
```

并额外保留诊断字段：

```text
status_potential_decay = 0.0
status_potential_delta = Phi(s') - Phi(s)
```

### 调整动机

此前 potential-based 形式使用：

```text
gamma * Phi(s') - Phi(s)
```

在 `gamma < 1` 时，即使 agent 的四项生存指标完全不变，也会产生负项。例如满状态 `Phi≈6` 时，每步约为：

```text
(0.997 - 1) * 6 = -0.018
```

这会让长 episode 额外承受持续负反馈，和 `alive_bonus` 的正反馈同时存在，形成“活得久有奖励，但活得久也被 potential 扣分”的混合信号。当前任务中 agent 已表现出不探索、反复横跳、方向偏好等问题，因此先测试移除这部分隐含时间惩罚是否能让食物/饮水/血量变化信号更清晰。

### survival-only 边界

本次不加入资源距离、未知区域、visited bonus 或其他外部探索奖励。reward 仍只依赖 agent 自身四项生存指标及终局：

- `health`
- `food`
- `drink`
- `energy`
- 死亡/最大步数成功

### 当前待观察指标

- `potential` 组件占比是否下降到更合理范围。
- 平均 episode 长度是否上升，且不是单纯站桩拖步。
- `recent_avg_eats`、`recent_avg_drinks` 和 `eat_drink` episode 比例是否改善。
- `visited_area / move_actions` 若后续加入，应观察是否比旧 potential 更有效探索。
- 是否出现新的问题：由于状态不变不再扣分，agent 是否更容易无意义拖步。

## 记录 25：置零 alive bonus，检验方向偏好的奠基者效应

时间：2026-05-12

### 调整内容

| 参数 | 调整前 | 调整后 |
| --- | ---: | ---: |
| `SurvivalRewardConfig.alive_bonus` | `0.04` | `0.00` |

当前已在代码中手动改为：

```text
alive_bonus = 0.00
```

记录 24 已经移除了 survival potential 中的 gamma，因此在四项生存指标不变、没有低状态惩罚、没有死亡/成功终局时：

```text
status_potential_reward = 0
alive_bonus = 0
step reward = 0
```

也就是说，不动或执行对生存状态没有影响的动作，不再因为“多活一步”获得直接正奖励。

### 调整动机

当前怀疑方向偏好可能来自早期训练中的“奠基者效应”：在随机策略阶段，某一个方向动作如果偶然比其他动作更频繁地获得 `alive_bonus` 带来的正回报，PPO 可能会在早期就把该方向动作权重推高。之后该方向动作更常被采样，继续获得相似回报，最终形成路径依赖和方向固化。

这种机制不需要该方向真的更有利；它只需要在早期随机波动中占到一点优势，然后被 alive reward 放大。因此本次把 `alive_bonus` 置零，用来检验方向动作是否还会因为非状态变化奖励而被固定。

### survival-only 边界

本次仍然保持 survival-only：不加入资源距离、visited bonus、unknown bonus 或任何外部探索奖励。reward 只来自：

- 四项生存指标的真实变化：`Phi(s') - Phi(s)`
- 低状态惩罚
- 死亡惩罚
- 最大步数成功奖励

### 当前待观察指标

- 四方向 `move_ratios` 是否更均衡，尤其是否仍出现单方向 50% 以上。
- `recent_avg_move_actions` 是否下降；如果明显下降，说明 alive reward 可能确实在维持移动频率。
- `visited_area` 是否下降；若下降，说明 alive reward 也承担了鼓励行动/拖长 episode 的作用。
- 平均 episode 长度是否明显缩短。
- 成功率是否更稀疏；如果成功率下降但方向偏好减弱，说明后续需要寻找不背离 survival-only 的方式增强长期信用分配。

### 可能解释

若置零 alive 后方向偏好明显减弱，说明此前的方向固化很可能不是地图结构本身导致，而是早期 alive reward 对任意动作的正反馈造成了路径依赖。

若置零 alive 后方向偏好仍然存在，则问题更可能来自 action mask、地图分布、出生点局部结构、PPO 更新早期熵下降，或 observation 编码中某些方向相关偏置。
