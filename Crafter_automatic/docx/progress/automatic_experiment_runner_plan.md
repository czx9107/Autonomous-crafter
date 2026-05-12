# Plan

目标是在 `Crafter_automatic` 中实现一个上层自动实验控制脚本，让它能够运行 `01-basic_reward.py`、保存完整日志、解析关键探针、根据提前退出条件停止无效实验，并把每轮实验的参数与结果写入实验记录。正式实验开始前，先只完成 runner 设计和验证流程；实验主线从记录 26 的 oldmap 当前最优基线出发。

## Scope

- In:
  - 新增自动实验 runner：`Crafter_automatic/01-auto_experiment_runner.py`。
  - 默认调用 `Crafter_automatic/01-basic_reward.py`，不加 `--new-map`，以 oldmap 基线开始。
  - 将 stdout/stderr 同时输出到终端和日志文件。
  - 解析 `[success]`、`[episode_stats]`、SB3 rollout/train 表格中的关键指标。
  - 支持提前退出条件：成功率达标、长期无改善、方向占比坍缩、训练异常退出。
  - 每轮实验生成结构化 summary，并追加写入实验记录文档。
  - 保持 survival-only reward，不加入资源距离、visited bonus、unknown bonus。

- Out:
  - 不在 runner 第一版中自动设计复杂新 reward。
  - 不做无限后台常驻训练。
  - 不把 `newmap` 作为默认主线。
  - 不自动删除旧模型、旧 gif 或旧实验结果。
  - 不把外部资源位置、未知区域比例或 visited area 写入 reward。

## Action Items

[ ] Add runner scaffold  
新增 `Crafter_automatic/01-auto_experiment_runner.py`。脚本职责是启动训练、保存日志、解析指标、判断是否提前退出、写 summary。它不替代 `01-basic_reward.py`，只作为上层控制器。

[ ] Add experiment workspace layout  
每轮实验输出到独立目录，建议格式：

```text
Crafter_automatic/saves/auto_experiments/<timestamp>_<experiment_id>/
  train_stdout.log
  train_summary.json
  params.json
  renders/
  model/
```

其中 `train_stdout.log` 保存完整 stdout/stderr，`train_summary.json` 保存解析后的最后结果、最佳结果、提前退出原因和是否达标。

[ ] Add log parser  
解析以下日志块：

```text
[success] episodes=... timesteps=... recent_success_rate=... total_success_rate=...
[episode_stats]
  recent_avg_eats=...
  recent_avg_drinks=...
  recent_avg_visited_area=...
  recent_avg_move_actions=...
  recent_move_ratios left/right/up/down:
  recent_reward_components avg_per_episode/abs_share ...
  recent_supply_table ...
  recent_deaths=... recent_death_ratios=...
```

同时从 SB3 表格中提取：

```text
ep_len_mean
ep_rew_mean
total_timesteps
approx_kl
clip_fraction
entropy_loss
explained_variance
value_loss
```

[ ] Add early-stop rules  
runner 第一版使用保守提前退出，不轻易误杀可能后期起势的实验。

建议默认规则：

```text
target_success_rate = 0.10
target_timesteps = 250000
min_timesteps_before_early_stop = 80000
patience_reports = 4
min_success_improvement = 0.01
direction_collapse_threshold = 0.70
direction_collapse_patience = 2
min_visited_area_for_direction_check = 12
```

提前退出条件：

- 达标退出：`recent_success_rate >= 0.10` 且 `timesteps <= 250000`。
- 无改善退出：超过最小步数后，连续 `patience_reports` 次 report 中 `recent_success_rate` 没有提升 `0.01`，且 `recent_avg_visited_area`、`recent_avg_eats/drinks` 也没有明显改善。
- 方向坍缩退出：连续 `direction_collapse_patience` 次 report 中，四方向最大占比 `>0.70`，同时 `recent_avg_visited_area < 12`。如果 visited area 很高，不直接判定为坍缩，因为可能是有方向地探索。
- 异常退出：训练进程非零退出、日志解析不到任何 `[success]` 块、出现 traceback。

[ ] Add parameter-change protocol  
每轮只允许修改 1-2 个变量。第一版不让 runner 随意编辑源码，而是采用“候选实验清单 + 受控 patch”的方式：

```text
experiment_id
base: record_26_oldmap_best
changes:
  - name: PPO_ENT_COEF
    from: 0.00
    to: 0.01
  - name: PPO_CLIP_RANGE
    from: 0.2
    to: 0.15
```

runner 在运行前检查源码中原值是否匹配 `from`，不匹配则拒绝 patch，避免在错误基线上实验。每轮 patch diff 写入 `params.json` 和实验记录。

[ ] Add smoke-test gate  
每次应用参数改动后，先运行：

```powershell
conda run -n crafter python -B Crafter_automatic/01-basic_reward.py --smoke-test
```

smoke 通过后才运行训练。若 smoke 失败，记录失败原因并停止该轮。

[ ] Add training command template  
默认训练命令建议：

```powershell
conda run -n crafter python -B Crafter_automatic/01-basic_reward.py `
  --total-timesteps 250000 `
  --episode-steps 512 `
  --n-envs 8 `
  --vec-env subproc `
  --n-steps 512 `
  --batch-size 1024 `
  --learning-rate 0.00015 `
  --log-interval 5 `
  --render-every 64 `
  --output-dir Crafter_automatic/saves/auto_experiments/<run_id>/model
```

runner 应允许通过配置覆盖 `n-envs`、`total-timesteps`、`batch-size`、`learning-rate`，但每轮仍需记录实际命令。

[ ] Add experiment-log writer  
每轮开始时，向 `Crafter_automatic/docx/progress/01-basic_reward_experiment_log.md` 追加：

```text
实验编号
基线
修改变量
假设
训练命令
smoke 结果
```

每轮结束时，追加：

```text
最终/最佳 recent_success_rate
触发的停止条件
关键 episode_stats
是否达标
下一轮建议
```

[ ] Add first experiment queue  
正式实验从记录 26 oldmap 基线开始，不立即运行 newmap。候选队列只放少量、单变量或双变量实验：

```text
E00 baseline: 记录 26，不改参数
E01 entropy: PPO_ENT_COEF 0.00 -> 0.01
E02 conservative_update: PPO_CLIP_RANGE 0.2 -> 0.15
E03 entropy_plus_clip: PPO_ENT_COEF 0.01 + PPO_CLIP_RANGE 0.15
```

是否实际执行 E01-E03，应由 E00 的日志结果决定。

## Validation

- 编译检查：

```powershell
conda run -n crafter python -B -c "import pathlib; compile(pathlib.Path('Crafter_automatic/01-auto_experiment_runner.py').read_text(encoding='utf-8'), 'Crafter_automatic/01-auto_experiment_runner.py', 'exec'); compile(pathlib.Path('Crafter_automatic/01-basic_reward.py').read_text(encoding='utf-8'), 'Crafter_automatic/01-basic_reward.py', 'exec'); print('compile ok')"
```

- 训练脚本 smoke：

```powershell
conda run -n crafter python -B Crafter_automatic/01-basic_reward.py --smoke-test
```

- runner dry-run：

```powershell
conda run -n crafter python -B Crafter_automatic/01-auto_experiment_runner.py --dry-run
```

- runner 短程集成测试：

```powershell
conda run -n crafter python -B Crafter_automatic/01-auto_experiment_runner.py --max-runs 1 --total-timesteps 2048 --no-patch
```

该短测只验证日志重定向、解析、summary 写入、实验记录追加，不用它判断策略效果。

## Edge Cases / Risks

- 训练曲线可能后期起势，提前退出过严会误杀有效实验。解决：设置 `min_timesteps_before_early_stop=80000`，且方向坍缩必须同时满足 visited area 低。
- SB3 表格是纯文本，解析可能受换行和空格影响。解决：优先解析我们自己的 `[success]` 和 `[episode_stats]`，SB3 表格只作辅助。
- Codex 无法直接观看训练生成的 gif，因此正式自动实验不要依赖人工看 render 来判断策略。解决：正式 runner 可以使用 `--render-every 0` 关闭渲染，同时保留 `--report-every 64` 输出结构化 episode stats，主要依靠 `train_summary.json`、移动占比、visited area、进食/饮水 2x2 表、reward component 和死因比例判断策略。
- 如果需要替代 gif 观察，应优先增加文本探针，例如 episode 级路径统计、补给时机、成功/死亡前若干步摘要，而不是增加更多 gif 输出。
- Windows 下中断子进程可能留下子进程。解决：runner 使用 `subprocess.Popen`，提前退出时显式 terminate；必要时记录需要人工检查。
- 受控 patch 可能因源码变动失败。解决：patch 前检查旧值，失败就停止，不做猜测替换。
- 多进程训练 stdout 可能缓冲。解决：以行模式读取 stdout，并把 stderr 合并到 stdout。
- 正式训练会写入大量 gif/model/csv。解决：每轮单独 output dir，不清理旧结果。

## Open Questions

- 第一轮 E00 baseline 是否直接跑满 `250000` steps，还是先用 `100000` steps 检查 runner 和曲线？
- 方向坍缩阈值是否用 `0.70`，还是更严格的 `0.60`？
- 是否允许 runner 自动执行 E01/E02/E03，还是每轮结束后先停下来等待人工确认？

## Proposed Formal Workflow

1. 确认本计划。
2. 实现 `01-auto_experiment_runner.py`。
3. 跑 compile、smoke、dry-run、短程集成测试。
4. 启动 E00 baseline；正式自动实验建议加 `--render-every 0`，减少 gif 写入并把判断依据转移到日志探针。
5. runner 等训练结束或提前退出。
6. runner 解析日志，写 `train_summary.json` 和实验记录。
7. 如果 `recent_success_rate >= 0.10` 且 `timesteps <= 250000`，停止并标记达标。
8. 如果未达标，结合日志选择下一轮只改 1-2 个变量。
9. 每轮都重复 smoke -> train -> parse -> record。

## Initial Recommendation

第一轮正式实验建议先跑 E00 baseline 到 `250000` steps，不加 `--new-map`，不改 reward。原因是 `Crafter_automatic` 刚恢复到记录 26，需要先得到自动 runner 下的可复现实验曲线，后续所有修改才有参照。

如果你希望节省时间，可以先跑 `100000` steps 作为 runner 校准；但它不能作为是否达标的正式判断。
