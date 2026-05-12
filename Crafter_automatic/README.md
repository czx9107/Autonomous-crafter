# AIDR Crafter Environment

这是从 `Conan/conan/playground/` 抽出的独立 Crafter 风格环境，用于后续强化学习训练。它保留了 Conan 当前源码中的环境机制、材料、物品、trace、贴图和 Gym 风格 `Env` 接口，但不包含 Conan 的 Vandal/Detective 任务生成、视觉语言推理和论文复现实验代码。

## 包含内容

```text
Crafter/
├── crafter/
│   ├── assets/
│   ├── constants.py
│   ├── data.yaml
│   ├── engine.py
│   ├── env.py
│   ├── objects.py
│   ├── recorder.py
│   ├── run_gui.py
│   ├── run_random.py
│   └── worldgen.py
├── setup.py
├── MANIFEST.in
└── README.md
```

## 环境规则

不要在 base Python 或 base Conda 环境里安装依赖。建议使用独立环境：

```powershell
conda --no-plugins create -n crafter python=3.10
conda activate crafter
cd C:\Users\17014\Desktop\Code\AIDR\project\Crafter
pip install -e .
```

如果需要 GUI：

```powershell
pip install -e ".[gui]"
```

如果需要接 Stable Baselines3：

```powershell
pip install -e ".[training]"
```

## 最小用法

```python
import crafter

env = crafter.Env(length=100, seed=0, view_type="symbolic")
obs = env.reset()
done = False

while not done:
    action = env.action_space.n - 1
    obs, reward, done, info = env.step(action)

print(info["inventory"])
print(info["achievements"])
```

更推荐训练时用：

```python
action = env.action_names.index("noop")
obs, reward, done, info = env.step(action)
```

## Smoke Test

随机动作：

```powershell
cd C:\Users\17014\Desktop\Code\AIDR\project\Crafter
python -m crafter.run_random --episodes 1 --length 20 --area 32 32 --seed 0
```

GUI：

```powershell
cd C:\Users\17014\Desktop\Code\AIDR\project\Crafter
python -m crafter.run_gui --length 500 --boss True --footprints True
```

## 强化学习入口

核心接口是 `crafter.Env`：

- `reset() -> obs`
- `step(action) -> obs, reward, done, info`
- `render(size=None) -> rgb`
- `get_detailed_view() -> mat_map, obj_map`
- `action_names -> list[str]`

默认 symbolic observation 是局部 `9 x 9` `uint8` 矩阵。`info` 中包含 `inventory`、`achievements`、`semantic`、`player_pos` 和原始 `reward`。

后续训练建议先从 symbolic observation + inventory 向量开始，再逐步加入 action mask、frame stacking 或 visual observation。
