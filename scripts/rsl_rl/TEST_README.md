# 模型测试脚本使用说明 / Model Testing Script Usage

## 概述 / Overview

`test.py` 脚本用于按照课程要求对训练好的模型进行测试评估，包括三个主要测试项目：

1. **速度跟踪测试** (Velocity Tracking Test)
2. **抗干扰测试** (Disturbance Rejection Test)  
3. **复杂地形测试** (Terrain Traversal Test)

## 使用方法 / Usage

### 基本用法 / Basic Usage

```bash
# 运行所有测试 / Run all tests
python scripts/rsl_rl/test.py --task=Isaac-Limx-PF-Blind-Flat-Play-v0 --checkpoint_path=path/to/checkpoint

# 运行特定测试 / Run specific test
python scripts/rsl_rl/test.py --test_mode=velocity_tracking --task=Isaac-Limx-PF-Blind-Flat-Play-v0 --checkpoint_path=path/to/checkpoint

# 运行抗干扰测试 / Run disturbance test
python scripts/rsl_rl/test.py --test_mode=disturbance --task=Isaac-Limx-PF-Blind-Flat-Play-v0 --checkpoint_path=path/to/checkpoint

# 运行地形测试 / Run terrain test (需要使用地形环境)
python scripts/rsl_rl/test.py --test_mode=terrain --task=Isaac-Limx-PF-Blind-Rough-Play-v0 --checkpoint_path=path/to/checkpoint
```

### 参数说明 / Parameters

- `--test_mode`: 测试模式
  - `velocity_tracking`: 仅速度跟踪测试
  - `disturbance`: 仅抗干扰测试
  - `terrain`: 仅复杂地形测试
  - `all`: 运行所有测试（默认）

- `--checkpoint_path`: 模型检查点路径（必需）
  
- `--task`: 任务名称
  - 速度跟踪和抗干扰测试: `Isaac-Limx-PF-Blind-Flat-Play-v0`
  - 地形测试: `Isaac-Limx-PF-Blind-Rough-Play-v0` 或 `Isaac-Limx-PF-Blind-Stair-Play-v0`

- `--test_duration`: 测试时长（秒），默认60秒

- `--num_envs`: 并行环境数量，默认1（单机器人测试）

- `--disturbance_prob`: 抗干扰测试中每步施加干扰的概率，默认0.01

- `--disturbance_force_range`: 干扰力范围（N），默认[-500, 500]

- `--video`: 是否录制视频

- `--seed`: 随机种子

## 测试指标 / Test Metrics

### 速度跟踪测试指标 / Velocity Tracking Metrics

- **速度跟踪误差 (MSE)**:
  - `mse_vx`: v_x方向的均方误差
  - `mse_vy`: v_y方向的均方误差  
  - `mse_omega_z`: 角速度的均方误差
  - `mse_total`: 总体均方误差

- **姿态稳定性**:
  - `roll_mean/std/max`: Roll角的均值/标准差/最大值
  - `pitch_mean/std/max`: Pitch角的均值/标准差/最大值

- **存活率**:
  - `survival_rate`: 存活率（百分比）
  - `total_steps`: 总步数
  - `alive_steps`: 存活步数

### 抗干扰测试指标 / Disturbance Rejection Metrics

- **干扰统计**:
  - `num_disturbances`: 施加的干扰次数
  - `max_force_magnitude`: 最大干扰力大小（N）
  - `avg_recovery_time`: 平均恢复时间（秒）

- **存活率**: 同速度跟踪测试

### 复杂地形测试指标 / Terrain Traversal Metrics

- **通过情况**:
  - `survival_rate`: 存活率
  - `distance_traveled`: 前进距离（米）
  - `total_steps`: 总步数

## 输出结果 / Output Results

测试结果会：
1. 在控制台实时显示
2. 保存到 `logs/rsl_rl/{experiment_name}/{run_name}/test_results.txt`

## 示例 / Examples

```bash
# 示例1: 完整的速度跟踪测试（60秒）
python scripts/rsl_rl/test.py \
    --test_mode=velocity_tracking \
    --task=Isaac-Limx-PF-Blind-Flat-Play-v0 \
    --checkpoint_path=logs/rsl_rl/pf_flat/2024-01-01_12-00-00/model_15000.pt \
    --test_duration=60.0 \
    --video

# 示例2: 抗干扰测试（自定义参数）
python scripts/rsl_rl/test.py \
    --test_mode=disturbance \
    --task=Isaac-Limx-PF-Blind-Flat-Play-v0 \
    --checkpoint_path=logs/rsl_rl/pf_flat/2024-01-01_12-00-00/model_15000.pt \
    --test_duration=60.0 \
    --disturbance_prob=0.02 \
    --disturbance_force_range -800 800

# 示例3: 地形测试
python scripts/rsl_rl/test.py \
    --test_mode=terrain \
    --task=Isaac-Limx-PF-Blind-Rough-Play-v0 \
    --checkpoint_path=logs/rsl_rl/pf_flat/2024-01-01_12-00-00/model_15000.pt \
    --test_duration=120.0
```

## 注意事项 / Notes

1. **环境选择**: 
   - 速度跟踪和抗干扰测试使用平地环境（Flat）
   - 地形测试需要使用粗糙地形或楼梯环境（Rough/Stair）

2. **测试时长**: 
   - 速度跟踪测试建议60秒（课程要求约1分钟）
   - 地形测试可以设置更长时间

3. **干扰参数**: 
   - 干扰概率和力范围可以根据需要调整
   - 较高的干扰力可以测试机器人的极限抗干扰能力

4. **性能**: 
   - 单环境测试（`--num_envs=1`）适合详细分析
   - 多环境测试可以加快评估速度但可能影响数据精度

