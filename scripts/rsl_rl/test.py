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
parser.add_argument("--checkpoint_path", type=str, default=None, 
                    help="Relative path to checkpoint file.")
parser.add_argument("--task", type=str, default="Isaac-Limx-PF-Blind-Flat-Play-v0",
                    help="Task name for testing")
parser.add_argument("--num_envs", type=int, default=1, 
                    help="Number of environments to simulate (1 for single robot testing)")
parser.add_argument("--test_duration", type=float, default=60.0,
                    help="Test duration in seconds (default: 60s for velocity tracking)")
parser.add_argument("--disturbance_prob", type=float, default=0.01,
                    help="Probability of applying disturbance per step")
parser.add_argument("--disturbance_force_range", type=float, nargs=2, default=[-500, 500],
                    help="Disturbance force range in N")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record videos during testing")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for testing")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
        """记录一步的数据 / Record data for one step"""
        # 获取机器人资产 / Get robot asset
        robot: Articulation = env.unwrapped.scene["robot"]
        
        # 记录速度命令和实际速度 / Record velocity commands and actual velocities
        if "observations" in infos:
            obs_dict = infos["observations"]
            if "commands" in obs_dict:
                cmd_vel = obs_dict["commands"].cpu().numpy()[0]  # [v_x, v_y, omega_z]
                self.commanded_velocities.append(cmd_vel)
        
        # 实际速度 / Actual velocity
        actual_vel = robot.data.root_lin_vel_b.cpu().numpy()[0]  # 基座坐标系下的速度
        actual_ang_vel = robot.data.root_ang_vel_b.cpu().numpy()[0]  # 角速度
        actual_vel_xy = actual_vel[:2]  # [v_x, v_y]
        actual_ang_vel_z = actual_ang_vel[2]  # omega_z
        self.actual_velocities.append([actual_vel_xy[0], actual_vel_xy[1], actual_ang_vel_z])
        
        # 计算速度跟踪误差 / Calculate velocity tracking error
        if len(self.commanded_velocities) > 0:
            cmd = self.commanded_velocities[-1]
            error = np.array([cmd[0] - actual_vel_xy[0], 
                            cmd[1] - actual_vel_xy[1], 
                            cmd[2] - actual_ang_vel_z])
            self.velocity_errors.append(error)
        
        # 记录姿态 (Roll, Pitch) / Record orientation (Roll, Pitch)
        # 使用投影重力来获取姿态信息（更简单且稳定） / Use projected gravity for orientation (simpler and more stable)
        quat = robot.data.root_quat_w[0].cpu()
        # 简化的Roll/Pitch计算（从四元数） / Simplified Roll/Pitch calculation (from quaternion)
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # 计算Roll (绕X轴) 和 Pitch (绕Y轴) / Calculate Roll (around X-axis) and Pitch (around Y-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        self.base_orientations.append([roll.item(), pitch.item()])  # Roll, Pitch
        
        # 记录位置 / Record position
        pos = robot.data.root_pos_w.cpu().numpy()[0]
        self.positions.append(pos)
        
        # 检查是否存活（基座高度和姿态） / Check if alive (base height and orientation)
        base_height = pos[2]
        roll_abs = abs(roll.item())
        pitch_abs = abs(pitch.item())
        alive = (base_height > 0.3) and (roll_abs < 0.5) and (pitch_abs < 0.5)  # 阈值可调
        self.is_alive.append(alive)
        
        self.step_count += 1
    
    def compute_velocity_tracking_metrics(self) -> Dict:
        """计算速度跟踪指标 / Compute velocity tracking metrics"""
        if len(self.velocity_errors) == 0:
            return {}
        
        errors = np.array(self.velocity_errors)
        mse = np.mean(errors ** 2, axis=0)  # 每个分量的MSE
        mse_total = np.mean(np.sum(errors ** 2, axis=1))  # 总体MSE
        
        return {
            "mse_vx": float(mse[0]),
            "mse_vy": float(mse[1]),
            "mse_omega_z": float(mse[2]),
            "mse_total": float(mse_total),
            "rmse_vx": float(np.sqrt(mse[0])),
            "rmse_vy": float(np.sqrt(mse[1])),
            "rmse_omega_z": float(np.sqrt(mse[2])),
        }
    
    def compute_stability_metrics(self) -> Dict:
        """计算姿态稳定性指标 / Compute stability metrics"""
        if len(self.base_orientations) == 0:
            return {}
        
        orientations = np.array(self.base_orientations)
        roll = orientations[:, 0]
        pitch = orientations[:, 1]
        
        return {
            "roll_mean": float(np.mean(np.abs(roll))),
            "roll_std": float(np.std(roll)),
            "roll_max": float(np.max(np.abs(roll))),
            "pitch_mean": float(np.mean(np.abs(pitch))),
            "pitch_std": float(np.std(pitch)),
            "pitch_max": float(np.max(np.abs(pitch))),
        }
    
    def compute_survival_metrics(self) -> Dict:
        """计算存活率指标 / Compute survival metrics"""
        if len(self.is_alive) == 0:
            return {}
        
        alive_array = np.array(self.is_alive)
        survival_rate = np.mean(alive_array)
        consecutive_alive = self._max_consecutive(alive_array, True)
        
        return {
            "survival_rate": float(survival_rate),
            "total_steps": len(self.is_alive),
            "alive_steps": int(np.sum(alive_array)),
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
    print(f"{'='*60}\n")
    
    return {**velocity_metrics, **stability_metrics, **survival_metrics}


def test_disturbance_rejection(env, policy, encoder, metrics: TestMetrics, duration: float, 
                                disturbance_prob: float, force_range: Tuple[float, float]):
    """测试抗干扰能力 / Test disturbance rejection capability"""
    print(f"\n{'='*60}")
    print(f"开始抗干扰测试 (测试时长: {duration}秒) / Starting disturbance rejection test (duration: {duration}s)")
    print(f"{'='*60}\n")
    
    metrics.reset()
    metrics.start_time = time.time()
    
    # 重置环境 / Reset environment
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
            if np.random.rand() < disturbance_prob and (step_count - last_disturbance_step) > 50:
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
            
            # 检查恢复 / Check recovery
            if recovery_start_step is not None and step_count > recovery_start_step:
                # 简单恢复判断：姿态稳定 / Simple recovery check: stable orientation
                if len(metrics.base_orientations) >= 2:
                    current_orient = np.array(metrics.base_orientations[-1])
                    if np.max(np.abs(current_orient)) < 0.1:  # 阈值可调
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
            
            # 记录初始位置 / Record initial position
            if initial_pos is None and len(metrics.positions) > 0:
                initial_pos = np.array(metrics.positions[0])
            
            step_count += 1
            if step_count % 100 == 0:
                elapsed = time.time() - metrics.start_time
                if len(metrics.positions) > 0:
                    current_pos = np.array(metrics.positions[-1])
                    distance = np.linalg.norm(current_pos[:2] - initial_pos[:2]) if initial_pos is not None else 0
                    print(f"进度: {elapsed:.1f}/{duration:.1f}秒, 步数: {step_count}, 前进距离: {distance:.2f}m / Progress: {elapsed:.1f}/{duration:.1f}s, Steps: {step_count}, Distance: {distance:.2f}m")
    
    # 计算指标 / Compute metrics
    print(f"\n计算测试指标... / Computing test metrics...")
    survival_metrics = metrics.compute_survival_metrics()
    
    # 计算前进距离 / Compute traversal distance
    if len(metrics.positions) > 0 and initial_pos is not None:
        final_pos = np.array(metrics.positions[-1])
        distance_traveled = np.linalg.norm(final_pos[:2] - initial_pos[:2])
    else:
        distance_traveled = 0.0
    
    print(f"\n{'='*60}")
    print("复杂地形测试结果 / Terrain Traversal Test Results")
    print(f"{'='*60}")
    print(f"通过情况:")
    print(f"  存活率: {survival_metrics.get('survival_rate', 0)*100:.2f}%")
    print(f"  前进距离: {distance_traveled:.2f}m")
    print(f"  总步数: {survival_metrics.get('total_steps', 0)}")
    print(f"{'='*60}\n")
    
    return {**survival_metrics, "distance_traveled": distance_traveled}


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
    log_dir = os.path.dirname(resume_path)
    
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    
    # 创建isaac环境 / Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "test"),
            "step_trigger": lambda step: step == 0,
            "video_length": int(args_cli.test_duration / env.unwrapped.step_dt),
            "disable_logger": True,
        }
        print("[INFO] Recording videos during testing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    
    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)
    
    # 创建指标收集器 / Create metrics collector
    metrics = TestMetrics()
    
    # 根据测试模式运行测试 / Run tests according to test mode
    test_mode = args_cli.test_mode
    results = {}
    
    if test_mode == "velocity_tracking" or test_mode == "all":
        results["velocity_tracking"] = test_velocity_tracking(
            env, policy, encoder, metrics, args_cli.test_duration
        )
    
    if test_mode == "disturbance" or test_mode == "all":
        # 对于抗干扰测试，使用较小的环境数量 / For disturbance test, use smaller number of envs
        results["disturbance"] = test_disturbance_rejection(
            env, policy, encoder, metrics, args_cli.test_duration,
            args_cli.disturbance_prob, tuple(args_cli.disturbance_force_range)
        )
    
    if test_mode == "terrain" or test_mode == "all":
        # 对于地形测试，需要使用地形环境 / For terrain test, need terrain environment
        if "Rough" not in args_cli.task and "Stair" not in args_cli.task:
            print("[WARNING] Terrain test recommended with terrain environment (e.g., Isaac-Limx-PF-Blind-Rough-Play-v0)")
        results["terrain"] = test_terrain_traversal(
            env, policy, encoder, metrics, args_cli.test_duration
        )
    
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
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被用户中断 / Test interrupted by user")
    finally:
        # close sim app
        simulation_app.close()

