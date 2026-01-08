# Velocity Tracking Reward Function（速度跟踪奖励函数）

本文档基于 `limxtron1lab-main` 当前代码结构，梳理“速度跟踪奖励（Velocity Tracking Reward）”在项目中的**配置入口**、**调用路径**与**实现位置（可追溯到的部分）**，用于对应报告中的：

- **B. Reward Function Design**
- **Listing 1: Velocity Tracking Reward Function**

> 说明：本仓库将部分通用 MDP 组件复用自 `isaaclab_tasks.manager_based.locomotion.velocity`（外部/上游 IsaacLab Tasks 包）。因此 `track_lin_vel_xy_exp` 的“函数源码”不在本仓库内，而是通过 `from isaaclab_tasks... import *` 的方式引入。

---

## 1. Reward Function Design 在哪里配置？

项目的 reward 设计采用 IsaacLab 的 `RewardTermCfg`（在代码中别名为 `RewTerm`）进行声明式配置，然后由环境管理器统一执行与加权求和。

### 1.1 配置入口（SF SoleFoot 版本示例）

文件：`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py`

在 `RewardsCfg` 中定义了 reward 项：

- `keep_balance`
- `rew_lin_vel_xy`（线速度跟踪）
- `rew_ang_vel_z`（角速度跟踪）
- 各类 penalty（姿态、扭矩、关节限制、接触等）

其中速度跟踪项是：

- `rew_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=..., params={...})`
- `rew_ang_vel_z  = RewTerm(func=mdp.track_ang_vel_z_exp,  weight=..., params={...})`

该结构正对应报告中的 **B. Reward Function Design**：即 reward 由若干 term 组成，每个 term 有 `func + params + weight`。

### 1.2 其他机器人/任务配置对比

同样的 reward term 在以下 cfg 里也存在（权重/超参不同）：

- `exts/bipedal_locomotion/.../cfg/PF/limx_base_env_cfg.py`
- `exts/bipedal_locomotion/.../cfg/WF/limx_base_env_cfg.py`

此外，机器人特定 cfg 里可能会 **二次覆盖** weight/params，比如：

- `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_solefoot_env_cfg.py`
- `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/limx_pointfoot_env_cfg.py`

---

## 2. Listing 1: Velocity Tracking Reward Function 对应哪段代码？

报告里的 “Velocity Tracking Reward Function” 在本项目中，对应的是 reward term：

- `mdp.track_lin_vel_xy_exp`（线速度 xy 跟踪的指数型奖励）

### 2.1 在本项目 cfg 中的声明（用于复现 Listing 1 的入口）

文件：`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/SF/limx_base_env_cfg.py`

关键字段：

- `func = mdp.track_lin_vel_xy_exp`
- `params = {"command_name": "base_velocity", "std": sqrt(...)}`
- `weight = ...`

> `command_name` 表示跟踪的目标速度来自 command manager（这里是 `base_velocity` 指令）。

### 2.2 mdp 命名空间是如何解析到奖励函数的？

文件：`exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/__init__.py`

该文件第一行：

```python
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *
```

表示：

- `bipedal_locomotion.tasks.locomotion.mdp` 会把上游 `isaaclab_tasks.manager_based.locomotion.velocity.mdp` 中的所有 mdp 函数（包括 rewards）导入到本地命名空间。
- 所以 `mdp.track_lin_vel_xy_exp` 实际来源是 **上游 IsaacLab Tasks 包**，而不是本仓库里的 `mdp/rewards.py`。

因此：

- **Listing 1 的真正源码位置**应在 Python 环境的 site-packages 中：
  `isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`（或同名模块）

本仓库中无法直接打开该文件（因为它不在 repo 内）。

> 但本仓库仍然保留了它的调用方式、权重与参数（std、command_name），足以在报告中解释“奖励函数设计”和“如何使用该 reward”。

---

## 3. 与日志 env.yaml 的对应关系（训练时实际使用的 reward 配置）

训练启动后，最终生效的配置会被保存到日志目录的：

- `logs/rsl_rl/.../params/env.yaml`

你可以在 `env.yaml` 里看到同名的 reward term（例如 `rew_lin_vel_xy`）以及最终的：

- `func: isaaclab.envs.mdp.rewards:track_lin_vel_xy_exp`
- `params: {command_name: base_velocity, std: ...}`
- `weight: ...`

这说明 reward term 在运行时会被序列化为“模块路径 + 函数名”，便于复现。

---

## 4. 一句总结（可直接写入报告）

- **B. Reward Function Design**：在 `.../cfg/*/limx_base_env_cfg.py` 的 `RewardsCfg` 中以 `RewardTermCfg(RewTerm)` 对每个奖励项进行声明式配置（函数、参数、权重），环境在每一步对各项 reward 计算并按权重加权求和。
- **Listing 1: Velocity Tracking Reward Function**：对应 `mdp.track_lin_vel_xy_exp`（线速度跟踪指数奖励）。本仓库通过 `bipedal_locomotion.tasks.locomotion.mdp.__init__` 引入上游 `isaaclab_tasks...velocity.mdp` 的实现，因此函数源码位于外部 IsaacLab Tasks 包中；本仓库可追溯到其调用入口与超参（`command_name`、`std`、`weight`）。
