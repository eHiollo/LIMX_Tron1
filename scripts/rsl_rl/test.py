"""模型测试脚本 - 按照课程要求进行模型评估 / Model testing script - evaluate model according to course requirements."""

"""首先启动Isaac Sim仿真器 / Launch Isaac Sim Simulator first."""

import argparse
import os
import time
import numpy as np
import torch
from typing import Dict, List, Tuple

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 添加argparse参数 / Add argparse arguments
parser = argparse.ArgumentParser(description="Test RL agent with various evaluation metrics.")
parser.add_argument("--test_mode", type=str, default="all", 
                    choices=["velocity_tracking", "disturbance", "terrain", "all"],
                    help="Test mode: velocity_tracking, disturbance, terrain, or all")
parser.add_argument("--checkpoint_path", type=str, default="output/play/2025-12-09_20-04-02_rough_from_flat_3000/model_2600.pt", 
                    help="Relative path to checkpoint file.")
parser.add_argument("--task", type=str, default="Isaac-Limx-PF-Blind-Flat-Play-v0",
                    help="Task name for testing")
parser.add_argument("--num_envs", type=int, default=512, 
                    help="Number of environments to simulate (1 for single robot testing)")
parser.add_argument("--test_duration", type=float, default=10,
                    help="Test duration in seconds (default: 60s for velocity tracking, 120s for disturbance, 180s for terrain)")
parser.add_argument("--disturbance_prob", type=float, default=0.03,
                    help="Probability of applying disturbance per step (default: 0.03, increased from 0.01)")
parser.add_argument("--disturbance_force_range", type=float, nargs=2, default=[-500, 500],
                    help="Disturbance force range in N")
parser.add_argument("--disturbance_min_interval", type=int, default=30,
                    help="Minimum steps between disturbances (default: 30, reduced from 50)")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during testing")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for testing")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 自动检测SSH连接，如果是SSH环境则自动启用无头模式
# Auto-detect SSH connection, enable headless mode automatically if in SSH environment
if os.environ.get('SSH_CONNECTION') or os.environ.get('SSH_CLIENT') or os.environ.get('SSH_TTY'):
    if hasattr(args_cli, 'headless') and not args_cli.headless:
        args_cli.headless = True
        print("[INFO] 检测到SSH连接，自动启用无头模式 / SSH connection detected, enabling headless mode automatically")

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runner import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg


class TestMetrics:
    """测试指标收集器 / Test metrics collector"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标 / Reset all metrics"""
        self.velocity_errors = []  # 速度跟踪误差 / Velocity tracking errors
        self.commanded_velocities = []  # 指令速度 / Commanded velocities
        self.actual_velocities = []  # 实际速度 / Actual velocities
        self.base_orientations = []  # 基座姿态 (roll, pitch, yaw) / Base orientations
        self.is_alive = []  # 是否存活 / Survival status
        self.disturbance_forces = []  # 施加的干扰力 / Applied disturbance forces
        self.recovery_times = []  # 恢复时间 / Recovery times after disturbance
        self.positions = []  # 位置记录 / Position records
        self.step_count = 0
        self.start_time = None
    
    def record_step(self, env, infos):
        """记录一步的数据（所有环境） / Record data for one step (all environments)"""
        # 获取机器人资产 / Get robot asset
        robot: Articulation = env.unwrapped.scene["robot"]
        
        # 获取环境数量 / Get number of environments
        num_envs = robot.num_instances
        
        # 记录速度命令和实际速度（所有环境） / Record velocity commands and actual velocities (all environments)
        if "observations" in infos:
            obs_dict = infos["observations"]
            if "commands" in obs_dict:
                # 获取所有环境的速度命令 / Get velocity commands for all environments
                cmd_vel_all = obs_dict["commands"].cpu().numpy()  # [num_envs, 3] = [v_x, v_y, omega_z]
                self.commanded_velocities.append(cmd_vel_all)
        
        # 实际速度（所有环境） / Actual velocity (all environments)
        actual_vel_all = robot.data.root_lin_vel_b.cpu().numpy()  # [num_envs, 3] 基座坐标系下的速度
        actual_ang_vel_all = robot.data.root_ang_vel_b.cpu().numpy()  # [num_envs, 3] 角速度
        actual_vel_xy_all = actual_vel_all[:, :2]  # [num_envs, 2] = [v_x, v_y]
        actual_ang_vel_z_all = actual_ang_vel_all[:, 2]  # [num_envs] = omega_z
        # 组合成 [num_envs, 3] 格式 / Combine into [num_envs, 3] format
        actual_velocities_step = np.column_stack([
            actual_vel_xy_all[:, 0],  # v_x
            actual_vel_xy_all[:, 1],  # v_y
            actual_ang_vel_z_all      # omega_z
        ])
        self.actual_velocities.append(actual_velocities_step)
        
        # 计算速度跟踪误差（所有环境） / Calculate velocity tracking error (all environments)
        if len(self.commanded_velocities) > 0:
            cmd_all = self.commanded_velocities[-1]  # [num_envs, 3]
            error_all = cmd_all - actual_velocities_step  # [num_envs, 3]
            self.velocity_errors.append(error_all)
        
        # 记录姿态 (Roll, Pitch)（所有环境） / Record orientation (Roll, Pitch) (all environments)
        quat_all = robot.data.root_quat_w.cpu().numpy()  # [num_envs, 4]
        orientations_step = []
        
        for i in range(num_envs):
            w, x, y, z = quat_all[i, 0], quat_all[i, 1], quat_all[i, 2], quat_all[i, 3]
            # 计算Roll (绕X轴) 和 Pitch (绕Y轴) / Calculate Roll (around X-axis) and Pitch (around Y-axis)
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1:
                pitch = np.copysign(np.pi / 2, sinp)
            else:
                pitch = np.arcsin(sinp)
            
            orientations_step.append([roll.item(), pitch.item()])
        
        self.base_orientations.append(np.array(orientations_step))  # [num_envs, 2]
        
        # 记录位置（所有环境） / Record position (all environments)
        pos_all = robot.data.root_pos_w.cpu().numpy()  # [num_envs, 3]
        self.positions.append(pos_all)
        
        # 检查是否存活（所有环境） / Check if alive (all environments)
        base_heights = pos_all[:, 2]  # [num_envs]
        orientations_array = np.array(orientations_step)  # [num_envs, 2]
        roll_abs = np.abs(orientations_array[:, 0])  # [num_envs]
        pitch_abs = np.abs(orientations_array[:, 1])  # [num_envs]
        
        alive_all = (base_heights > 0.3) & (roll_abs < 0.5) & (pitch_abs < 0.5)  # [num_envs] 阈值可调
        self.is_alive.append(alive_all)
        
        self.step_count += 1
    
    def compute_velocity_tracking_metrics(self) -> Dict:
        """计算速度跟踪指标（所有环境平均） / Compute velocity tracking metrics (averaged over all environments)"""
        if len(self.velocity_errors) == 0:
            return {}
        
        # 将所有步骤和所有环境的数据合并 / Concatenate data from all steps and all environments
        # errors: list of [num_envs, 3] arrays
        errors_all = np.concatenate(self.velocity_errors, axis=0)  # [total_samples, 3]
        
        # 计算每个分量的MSE（所有环境平均） / Compute MSE for each component (averaged over all environments)
        mse = np.mean(errors_all ** 2, axis=0)  # [3] = [mse_vx, mse_vy, mse_omega_z]
        mse_total = np.mean(np.sum(errors_all ** 2, axis=1))  # 总体MSE
        
        # 计算总样本数（步数 × 环境数） / Calculate total samples (steps × num_envs)
        total_samples = errors_all.shape[0]
        
        return {
            "mse_vx": float(mse[0]),
            "mse_vy": float(mse[1]),
            "mse_omega_z": float(mse[2]),
            "mse_total": float(mse_total),
            "rmse_vx": float(np.sqrt(mse[0])),
            "rmse_vy": float(np.sqrt(mse[1])),
            "rmse_omega_z": float(np.sqrt(mse[2])),
            "total_samples": int(total_samples),  # 总样本数 / Total number of samples
        }
    
    def compute_stability_metrics(self) -> Dict:
        """计算姿态稳定性指标（所有环境平均） / Compute stability metrics (averaged over all environments)"""
        if len(self.base_orientations) == 0:
            return {}
        
        # 将所有步骤和所有环境的数据合并 / Concatenate data from all steps and all environments
        # orientations: list of [num_envs, 2] arrays
        orientations_all = np.concatenate(self.base_orientations, axis=0)  # [total_samples, 2]
        roll = orientations_all[:, 0]  # [total_samples]
        pitch = orientations_all[:, 1]  # [total_samples]
        
        total_samples = orientations_all.shape[0]
        
        return {
            "roll_mean": float(np.mean(np.abs(roll))),
            "roll_std": float(np.std(roll)),
            "roll_max": float(np.max(np.abs(roll))),
            "pitch_mean": float(np.mean(np.abs(pitch))),
            "pitch_std": float(np.std(pitch)),
            "pitch_max": float(np.max(np.abs(pitch))),
            "total_samples": int(total_samples),  # 总样本数 / Total number of samples
        }
    
    def compute_survival_metrics(self) -> Dict:
        """计算存活率指标（所有环境平均） / Compute survival metrics (averaged over all environments)"""
        if len(self.is_alive) == 0:
            return {}
        
        # 将所有步骤和所有环境的数据合并 / Concatenate data from all steps and all environments
        # is_alive: list of [num_envs] boolean arrays
        alive_array_all = np.concatenate(self.is_alive, axis=0)  # [total_samples]
        survival_rate = np.mean(alive_array_all)
        
        # 计算最大连续存活步数（对于所有环境的平均值，按步骤计算） / Compute max consecutive alive (average across environments per step)
        # 首先计算每个步骤的平均存活率 / First compute average survival rate per step
        alive_per_step = np.array([np.mean(step_alive) for step_alive in self.is_alive])
        consecutive_alive = self._max_consecutive(alive_per_step >= 0.5, True)
        
        total_samples = alive_array_all.shape[0]
        
        return {
            "survival_rate": float(survival_rate),
            "total_steps": len(self.is_alive),
            "total_samples": int(total_samples),  # 总样本数（步数 × 环境数） / Total samples (steps × num_envs)
            "alive_steps": int(np.sum(alive_array_all)),
            "max_consecutive_alive": int(consecutive_alive),
        }
    
    def compute_disturbance_metrics(self) -> Dict:
        """计算干扰测试指标 / Compute disturbance metrics"""
        if len(self.disturbance_forces) == 0:
            return {}
        
        forces = np.array(self.disturbance_forces)
        max_force_magnitude = np.max(np.linalg.norm(forces, axis=1)) if len(forces) > 0 else 0.0
        
        recovery_times_array = np.array(self.recovery_times) if len(self.recovery_times) > 0 else np.array([])
        avg_recovery_time = float(np.mean(recovery_times_array)) if len(recovery_times_array) > 0 else 0.0
        
        return {
            "num_disturbances": len(self.disturbance_forces),
            "max_force_magnitude": float(max_force_magnitude),
            "avg_recovery_time": avg_recovery_time,
            "total_recovery_time": float(np.sum(recovery_times_array)),
        }
    
    def _max_consecutive(self, arr, value):
        """计算连续True的最大长度 / Compute max consecutive True"""
        max_count = 0
        current_count = 0
        for v in arr:
            if v == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count


def _create_test_env(task_name, env_cfg, enable_video, video_folder, test_name, duration):
    """为每个测试创建独立的环境实例（支持视频录制）
    Create separate environment instance for each test (with video recording support)
    """
    # 创建基础环境 / Create base environment
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if enable_video else None)
    
    # 如果启用视频录制，在 RslRlVecEnvWrapper 之前添加 RecordVideo wrapper
    # If video recording is enabled, add RecordVideo wrapper before RslRlVecEnvWrapper
    if enable_video and video_folder is not None:
        video_kwargs = {
            "video_folder": os.path.join(video_folder, test_name),
            "step_trigger": lambda step: step == 0,  # 在测试开始时录制 / Record at test start
            "video_length": int(duration / env.unwrapped.step_dt),
            "disable_logger": True,
        }
        print(f"[INFO] Setting up video recording for {test_name} test: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # wrap around environment for rsl-rl (RecordVideo must be before this)
    env = RslRlVecEnvWrapper(env)
    
    return env


def apply_disturbance(env, force_range: Tuple[float, float], env_id: int = 0):
    """施加干扰力到机器人基座 / Apply disturbance force to robot base"""
    robot: Articulation = env.unwrapped.scene["robot"]
    
    # 随机生成力 / Randomly generate force
    force_magnitude = np.random.uniform(force_range[0], force_range[1])
    angle = np.random.uniform(0, 2 * np.pi)
    force_x = force_magnitude * np.cos(angle)
    force_y = force_magnitude * np.sin(angle)
    force_z = 0.0
    
    # 创建力和力矩张量 / Create force and torque tensors
    forces = torch.zeros((1, 1, 3), device=robot.device)
    forces[0, 0, 0] = force_x
    forces[0, 0, 1] = force_y
    forces[0, 0, 2] = force_z
    
    torques = torch.zeros((1, 1, 3), device=robot.device)
    
    # 应用到基座 / Apply to base
    env_ids = torch.tensor([env_id], device=robot.device)
    body_ids = torch.tensor([0], device=robot.device)  # base_Link通常是第一个body
    
    robot.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=body_ids)
    
    return np.array([force_x, force_y, force_z])


def test_velocity_tracking(env, policy, encoder, metrics: TestMetrics, duration: float):
    """测试速度跟踪性能 / Test velocity tracking performance"""
    print(f"\n{'='*60}")
    print(f"开始速度跟踪测试 (测试时长: {duration}秒) / Starting velocity tracking test (duration: {duration}s)")
    print(f"{'='*60}\n")
    
    metrics.reset()
    metrics.start_time = time.time()
    
    # 重置环境 / Reset environment
    # RslRlVecEnvWrapper 使用 get_observations 而不是 reset
    # RslRlVecEnvWrapper uses get_observations instead of reset
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    if obs_history is not None:
        obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")
    
    step_count = 0
    dt = env.unwrapped.step_dt  # 仿真时间步 / Simulation timestep
    
    while simulation_app.is_running() and (time.time() - metrics.start_time) < duration:
        with torch.inference_mode():
            # 获取当前观测和命令 / Get current observations and commands
            if obs_history is not None:
                est = encoder(obs_history)
                obs_input = torch.cat((est, obs, commands), dim=-1).detach()
            else:
                obs_input = torch.cat((obs, commands), dim=-1).detach()
            
            # 执行策略 / Execute policy
            actions = policy(obs_input)
            
            # 环境步进 / Environment step
            obs, _, _, infos = env.step(actions)
            
            # 更新观测历史 / Update observation history
            if "observations" in infos:
                obs_dict = infos["observations"]
                if "obsHistory" in obs_dict:
                    obs_history = obs_dict["obsHistory"]
                    obs_history = obs_history.flatten(start_dim=1)
                if "commands" in obs_dict:
                    commands = obs_dict["commands"]
            
            # 记录数据 / Record data
            metrics.record_step(env, infos)
            
            step_count += 1
            if step_count % 100 == 0:
                elapsed = time.time() - metrics.start_time
                print(f"进度: {elapsed:.1f}/{duration:.1f}秒, 步数: {step_count} / Progress: {elapsed:.1f}/{duration:.1f}s, Steps: {step_count}")
    
    # 计算指标 / Compute metrics
    print(f"\n计算测试指标... / Computing test metrics...")
    velocity_metrics = metrics.compute_velocity_tracking_metrics()
    stability_metrics = metrics.compute_stability_metrics()
    survival_metrics = metrics.compute_survival_metrics()
    
    print(f"\n{'='*60}")
    print("速度跟踪测试结果 / Velocity Tracking Test Results")
    print(f"{'='*60}")
    print(f"速度跟踪误差 (MSE):")
    print(f"  v_x MSE: {velocity_metrics.get('mse_vx', 0):.6f} m²/s²")
    print(f"  v_y MSE: {velocity_metrics.get('mse_vy', 0):.6f} m²/s²")
    print(f"  omega_z MSE: {velocity_metrics.get('mse_omega_z', 0):.6f} rad²/s²")
    print(f"  总体MSE: {velocity_metrics.get('mse_total', 0):.6f}")
    
    print(f"\n姿态稳定性 (Roll/Pitch):")
    print(f"  Roll - 均值: {stability_metrics.get('roll_mean', 0):.4f} rad, "
          f"标准差: {stability_metrics.get('roll_std', 0):.4f} rad, "
          f"最大值: {stability_metrics.get('roll_max', 0):.4f} rad")
    print(f"  Pitch - 均值: {stability_metrics.get('pitch_mean', 0):.4f} rad, "
          f"标准差: {stability_metrics.get('pitch_std', 0):.4f} rad, "
          f"最大值: {stability_metrics.get('pitch_max', 0):.4f} rad")
    
    print(f"\n存活率:")
    print(f"  存活率: {survival_metrics.get('survival_rate', 0)*100:.2f}%")
    print(f"  总步数: {survival_metrics.get('total_steps', 0)}")
    print(f"  存活步数: {survival_metrics.get('alive_steps', 0)}")
    print(f"\n样本统计:")
    print(f"  总样本数: {velocity_metrics.get('total_samples', 0)} (步数 × 环境数)")
    print(f"{'='*60}\n")
    
    return {**velocity_metrics, **stability_metrics, **survival_metrics}


def test_disturbance_rejection(env, policy, encoder, metrics: TestMetrics, duration: float, 
                                disturbance_prob: float, force_range: Tuple[float, float],
                                min_interval: int = 30):
    """测试抗干扰能力 / Test disturbance rejection capability"""
    print(f"\n{'='*60}")
    print(f"开始抗干扰测试 (测试时长: {duration}秒) / Starting disturbance rejection test (duration: {duration}s)")
    print(f"干扰概率: {disturbance_prob:.3f}, 最小间隔: {min_interval}步 / Disturbance prob: {disturbance_prob:.3f}, Min interval: {min_interval} steps")
    print(f"{'='*60}\n")
    
    metrics.reset()
    metrics.start_time = time.time()
    
    # 重置环境 / Reset environment
    # RslRlVecEnvWrapper 使用 get_observations 而不是 reset
    # RslRlVecEnvWrapper uses get_observations instead of reset
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    if obs_history is not None:
        obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")
    
    step_count = 0
    last_disturbance_step = -1000
    recovery_start_step = None
    
    while simulation_app.is_running() and (time.time() - metrics.start_time) < duration:
        with torch.inference_mode():
            # 随机施加干扰 / Randomly apply disturbance
            if np.random.rand() < disturbance_prob and (step_count - last_disturbance_step) > min_interval:
                force = apply_disturbance(env, force_range)
                metrics.disturbance_forces.append(force)
                last_disturbance_step = step_count
                recovery_start_step = step_count
                print(f"步数 {step_count}: 施加干扰力 {np.linalg.norm(force):.2f}N / Step {step_count}: Applied disturbance force {np.linalg.norm(force):.2f}N")
            
            # 获取当前观测和命令 / Get current observations and commands
            if obs_history is not None:
                est = encoder(obs_history)
                obs_input = torch.cat((est, obs, commands), dim=-1).detach()
            else:
                obs_input = torch.cat((obs, commands), dim=-1).detach()
            
            # 执行策略 / Execute policy
            actions = policy(obs_input)
            
            # 环境步进 / Environment step
            obs, _, _, infos = env.step(actions)
            
            # 更新观测历史 / Update observation history
            if "observations" in infos:
                obs_dict = infos["observations"]
                if "obsHistory" in obs_dict:
                    obs_history = obs_dict["obsHistory"]
                    obs_history = obs_history.flatten(start_dim=1)
                if "commands" in obs_dict:
                    commands = obs_dict["commands"]
            
            # 记录数据 / Record data
            metrics.record_step(env, infos)
            
            # 检查恢复（所有环境的平均姿态） / Check recovery (average orientation across all environments)
            if recovery_start_step is not None and step_count > recovery_start_step:
                # 简单恢复判断：姿态稳定 / Simple recovery check: stable orientation
                if len(metrics.base_orientations) >= 2:
                    current_orient_all = metrics.base_orientations[-1]  # [num_envs, 2]
                    # 计算所有环境的平均姿态角度 / Compute average orientation angle across all environments
                    avg_orient = np.mean(np.abs(current_orient_all), axis=0)  # [2] = [avg_roll, avg_pitch]
                    if np.max(avg_orient) < 0.1:  # 阈值可调
                        recovery_time = (step_count - recovery_start_step) * env.unwrapped.step_dt
                        metrics.recovery_times.append(recovery_time)
                        recovery_start_step = None
            
            step_count += 1
            if step_count % 100 == 0:
                elapsed = time.time() - metrics.start_time
                print(f"进度: {elapsed:.1f}/{duration:.1f}秒, 步数: {step_count} / Progress: {elapsed:.1f}/{duration:.1f}s, Steps: {step_count}")
    
    # 计算指标 / Compute metrics
    print(f"\n计算测试指标... / Computing test metrics...")
    disturbance_metrics = metrics.compute_disturbance_metrics()
    survival_metrics = metrics.compute_survival_metrics()
    
    print(f"\n{'='*60}")
    print("抗干扰测试结果 / Disturbance Rejection Test Results")
    print(f"{'='*60}")
    print(f"干扰统计:")
    print(f"  干扰次数: {disturbance_metrics.get('num_disturbances', 0)}")
    print(f"  最大干扰力: {disturbance_metrics.get('max_force_magnitude', 0):.2f}N")
    print(f"  平均恢复时间: {disturbance_metrics.get('avg_recovery_time', 0):.3f}秒")
    
    print(f"\n存活率:")
    print(f"  存活率: {survival_metrics.get('survival_rate', 0)*100:.2f}%")
    print(f"  总样本数: {survival_metrics.get('total_samples', 0)} (步数 × 环境数)")
    print(f"{'='*60}\n")
    
    return {**disturbance_metrics, **survival_metrics}


def test_terrain_traversal(env, policy, encoder, metrics: TestMetrics, duration: float):
    """测试复杂地形适应能力 / Test terrain traversal capability"""
    print(f"\n{'='*60}")
    print(f"开始复杂地形测试 (测试时长: {duration}秒) / Starting terrain traversal test (duration: {duration}s)")
    print(f"{'='*60}\n")
    
    metrics.reset()
    metrics.start_time = time.time()
    
    # 重置环境 / Reset environment
    # RslRlVecEnvWrapper 使用 get_observations 而不是 reset
    # RslRlVecEnvWrapper uses get_observations instead of reset
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    if obs_history is not None:
        obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands")
    
    initial_pos = None
    step_count = 0
    
    while simulation_app.is_running() and (time.time() - metrics.start_time) < duration:
        with torch.inference_mode():
            # 获取当前观测和命令 / Get current observations and commands
            if obs_history is not None:
                est = encoder(obs_history)
                obs_input = torch.cat((est, obs, commands), dim=-1).detach()
            else:
                obs_input = torch.cat((obs, commands), dim=-1).detach()
            
            # 执行策略 / Execute policy
            actions = policy(obs_input)
            
            # 环境步进 / Environment step
            obs, _, _, infos = env.step(actions)
            
            # 更新观测历史 / Update observation history
            if "observations" in infos:
                obs_dict = infos["observations"]
                if "obsHistory" in obs_dict:
                    obs_history = obs_dict["obsHistory"]
                    obs_history = obs_history.flatten(start_dim=1)
                if "commands" in obs_dict:
                    commands = obs_dict["commands"]
            
            # 记录数据 / Record data
            metrics.record_step(env, infos)
            
            # 记录初始位置（所有环境的平均位置） / Record initial position (average across all environments)
            if initial_pos is None and len(metrics.positions) > 0:
                initial_pos_all = metrics.positions[0]  # [num_envs, 3]
                initial_pos = np.mean(initial_pos_all, axis=0)  # [3] 所有环境的平均初始位置
            
            step_count += 1
            if step_count % 100 == 0:
                elapsed = time.time() - metrics.start_time
                if len(metrics.positions) > 0 and initial_pos is not None:
                    current_pos_all = metrics.positions[-1]  # [num_envs, 3]
                    current_pos_avg = np.mean(current_pos_all, axis=0)  # [3] 所有环境的平均当前位置
                    # 计算所有环境的前进距离，然后取平均 / Compute distance for all envs, then average
                    distances = np.linalg.norm(current_pos_all[:, :2] - initial_pos[:2], axis=1)  # [num_envs]
                    avg_distance = np.mean(distances)
                    max_distance = np.max(distances)
                    print(f"进度: {elapsed:.1f}/{duration:.1f}秒, 步数: {step_count}, 平均前进距离: {avg_distance:.2f}m (最大: {max_distance:.2f}m) / Progress: {elapsed:.1f}/{duration:.1f}s, Steps: {step_count}, Avg Distance: {avg_distance:.2f}m (Max: {max_distance:.2f}m)")
    
    # 计算指标 / Compute metrics
    print(f"\n计算测试指标... / Computing test metrics...")
    survival_metrics = metrics.compute_survival_metrics()
    
    # 计算前进距离（所有环境的平均和最大） / Compute traversal distance (average and max across all environments)
    if len(metrics.positions) > 0 and initial_pos is not None:
        final_pos_all = metrics.positions[-1]  # [num_envs, 3]
        # 计算每个环境的前进距离 / Compute distance for each environment
        distances = np.linalg.norm(final_pos_all[:, :2] - initial_pos[:2], axis=1)  # [num_envs]
        distance_traveled_avg = np.mean(distances)  # 平均前进距离
        distance_traveled_max = np.max(distances)   # 最大前进距离
        distance_traveled_min = np.min(distances)   # 最小前进距离
    else:
        distance_traveled_avg = 0.0
        distance_traveled_max = 0.0
        distance_traveled_min = 0.0
    
    print(f"\n{'='*60}")
    print("复杂地形测试结果 / Terrain Traversal Test Results")
    print(f"{'='*60}")
    print(f"通过情况:")
    print(f"  存活率: {survival_metrics.get('survival_rate', 0)*100:.2f}%")
    print(f"  前进距离 (平均): {distance_traveled_avg:.2f}m")
    print(f"  前进距离 (最大): {distance_traveled_max:.2f}m")
    print(f"  前进距离 (最小): {distance_traveled_min:.2f}m")
    print(f"  总步数: {survival_metrics.get('total_steps', 0)}")
    print(f"  总样本数: {survival_metrics.get('total_samples', 0)}")
    print(f"{'='*60}\n")
    
    return {
        **survival_metrics, 
        "distance_traveled_avg": distance_traveled_avg,
        "distance_traveled_max": distance_traveled_max,
        "distance_traveled_min": distance_traveled_min,
    }


def main():
    """主测试函数 / Main test function"""
    # 解析配置 / Parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    env_cfg.seed = args_cli.seed
    
    # 指定日志实验目录 / Specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
        if not os.path.exists(resume_path):
            print(f"[ERROR] Checkpoint file not found: {resume_path}")
            print(f"[ERROR] Please specify a valid checkpoint path with --checkpoint_path")
            return
    
    if not os.path.exists(resume_path):
        print(f"[ERROR] Checkpoint file not found: {resume_path}")
        return
    
    log_dir = os.path.dirname(resume_path)
    
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    
    # 为视频录制确定测试时长（如果未指定，使用默认值60秒）/ Determine test duration for video recording
    if args_cli.test_duration is not None:
        video_test_duration = args_cli.test_duration
    else:
        # 根据测试模式使用默认时长 / Use default duration based on test mode
        if args_cli.test_mode == "velocity_tracking":
            video_test_duration = 60.0
        elif args_cli.test_mode == "disturbance":
            video_test_duration = 60.0
        elif args_cli.test_mode == "terrain":
            video_test_duration = 60.0
        else:  # all
            video_test_duration = 60.0  # 默认使用速度跟踪的时长
    
    # 创建指标收集器 / Create metrics collector
    metrics = TestMetrics()
    
    # 根据测试模式运行测试 / Run tests according to test mode
    test_mode = args_cli.test_mode
    results = {}
    
    # 为不同测试模式设置默认时长 / Set default duration for different test modes
    # 定义各测试的默认时长 / Define default durations for each test
    velocity_duration = 60.0   # 速度跟踪测试60秒
    disturbance_duration = 120.0  # 抗干扰测试120秒（增加干扰次数）
    terrain_duration = 180.0   # 地形测试180秒（增加前进距离）
    
    if args_cli.test_duration is not None:
        # 如果用户指定了时长，所有测试使用相同时长
        velocity_duration = args_cli.test_duration
        disturbance_duration = args_cli.test_duration
        terrain_duration = args_cli.test_duration
    
    # 视频录制基础路径 / Base path for video recording
    video_base_folder = os.path.join(log_dir, "videos", "test") if args_cli.video else None
    
    # 创建测试环境（只创建一次，避免多个环境冲突）
    # Create test environment (only once to avoid conflicts from multiple envs)
    # 注意：如果启用视频录制，需要在 RslRlVecEnvWrapper 之前添加 RecordVideo wrapper
    # Note: If video recording is enabled, RecordVideo wrapper must be added before RslRlVecEnvWrapper
    print(f"[INFO] Creating test environment...")
    test_env_base = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    if isinstance(test_env_base.unwrapped, DirectMARLEnv):
        test_env_base = multi_agent_to_single_agent(test_env_base)
    
    # 如果启用视频录制，添加 RecordVideo wrapper（必须在 RslRlVecEnvWrapper 之前）
    # If video recording is enabled, add RecordVideo wrapper (must be before RslRlVecEnvWrapper)
    if args_cli.video and video_base_folder is not None:
        os.makedirs(video_base_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_base_folder,
            "step_trigger": lambda step: step == 0,  # 在重置时录制 / Record on reset
            "video_length": int(max(velocity_duration, disturbance_duration, terrain_duration) / test_env_base.unwrapped.step_dt),
            "disable_logger": True,
            "name_prefix": "test",  # 视频文件名前缀 / Video filename prefix
        }
        print(f"[INFO] Setting up video recording: {video_base_folder}")
        test_env_base = gym.wrappers.RecordVideo(test_env_base, **video_kwargs)
    
    # 包装为 RSL-RL 环境
    # Wrap as RSL-RL environment
    test_env = RslRlVecEnvWrapper(test_env_base)
    
    # 加载模型
    # Load model
    print(f"[INFO] Loading model from checkpoint...")
    ppo_runner = OnPolicyRunner(test_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    
    # 获取训练好的策略用于推理
    # Obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=test_env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=test_env.unwrapped.device)
    
    print(f"[INFO] Model loaded successfully")
    
    # 运行测试 / Run tests
    if test_mode == "velocity_tracking" or test_mode == "all":
        duration = velocity_duration
        print(f"\n[INFO] Starting velocity_tracking test...")
        try:
            results["velocity_tracking"] = test_velocity_tracking(
                test_env, policy, encoder, metrics, duration
            )
            print(f"[INFO] Completed velocity_tracking test")
        except Exception as e:
            print(f"[ERROR] Failed to run velocity_tracking test: {e}")
            import traceback
            traceback.print_exc()
            results["velocity_tracking"] = {"error": str(e)}
    
    if test_mode == "disturbance" or test_mode == "all":
        duration = disturbance_duration
        print(f"\n[INFO] Starting disturbance test...")
        try:
            results["disturbance"] = test_disturbance_rejection(
                test_env, policy, encoder, metrics, duration,
                args_cli.disturbance_prob, tuple(args_cli.disturbance_force_range),
                args_cli.disturbance_min_interval
            )
            print(f"[INFO] Completed disturbance test")
        except Exception as e:
            print(f"[ERROR] Failed to run disturbance test: {e}")
            import traceback
            traceback.print_exc()
            results["disturbance"] = {"error": str(e)}
    
    if test_mode == "terrain" or test_mode == "all":
        if "Rough" not in args_cli.task and "Stair" not in args_cli.task:
            print("[WARNING] Terrain test recommended with terrain environment (e.g., Isaac-Limx-PF-Blind-Rough-Play-v0)")
        duration = terrain_duration
        print(f"\n[INFO] Starting terrain test...")
        try:
            results["terrain"] = test_terrain_traversal(
                test_env, policy, encoder, metrics, duration
            )
            print(f"[INFO] Completed terrain test")
        except Exception as e:
            print(f"[ERROR] Failed to run terrain test: {e}")
            import traceback
            traceback.print_exc()
            results["terrain"] = {"error": str(e)}
    
    # 保存结果 / Save results
    results_file = os.path.join(log_dir, "test_results.txt")
    with open(results_file, "w") as f:
        f.write("Test Results / 测试结果\n")
        f.write("="*60 + "\n\n")
        for test_name, test_results in results.items():
            f.write(f"{test_name.upper()}\n")
            f.write("-"*60 + "\n")
            for key, value in test_results.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    print(f"测试结果已保存到: {results_file} / Test results saved to: {results_file}")
    
    # 注意：不需要关闭环境，因为会在 finally 块中关闭 simulation_app
    # Note: Don't need to close env, as simulation_app will be closed in finally block


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被用户中断 / Test interrupted by user")
    finally:
        # close sim app
        simulation_app.close()

