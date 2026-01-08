"""RSL-RL智能体检查点播放脚本 / Script to play a checkpoint of an RL agent from RSL-RL."""

"""首先启动Isaac Sim仿真器 / Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# 添加argparse参数 / Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")

# 添加RSL-RL命令行参数 / Append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# 添加AppLauncher命令行参数 / Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# 启动Omniverse应用 / Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit


def main():
    """使用RSL-RL智能体进行测试 / Play with RSL-RL agent."""
    # 解析配置 / Parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    ) # type: ignore
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # 指定日志实验目录 / Specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)
    
    # 创建输出目录 / Create output directory
    # 从checkpoint路径提取实验名和运行名 / Extract experiment name and run name from checkpoint path
    checkpoint_basename = os.path.basename(os.path.dirname(resume_path))
    output_dir = os.path.join("output", "play", checkpoint_basename)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output will be saved to: {output_dir}")

    # 创建isaac环境 / Create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    video_folder = None
    if args_cli.video:
        video_folder = os.path.join(output_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)  # 确保视频目录存在 / Ensure video directory exists
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print(f"[INFO] Video will be saved to: {video_folder}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 如果RL算法需要，转换为单智能体实例 / Convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 为rsl-rl包装环境 / Wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # 加载先前训练的模型 / Load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # 获取训练好的策略用于推理 / Obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

     # 导出策略到onnx / Export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(output_dir, "exported")
        os.makedirs(export_model_dir, exist_ok=True)  # 确保导出目录存在 / Ensure export directory exists
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
    # 重置环境 / Reset environment
    obs, obs_dict = env.get_observations()
    obs_history = obs_dict["observations"].get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict["observations"].get("commands") 
    # 仿真环境 / Simulate environment
    step_count = 0
    try:
        while simulation_app.is_running():
            # 在推理模式下运行所有操作 / Run everything in inference mode
            with torch.inference_mode():
                # 智能体步进 / Agent stepping
                est = encoder(obs_history)
                actions = policy(torch.cat((est, obs, commands), dim=-1).detach())
                # 环境步进 / Env stepping
                obs, _, _, infos = env.step(actions)
                obs_history = infos["observations"].get("obsHistory")
                obs_history = obs_history.flatten(start_dim=1)
                commands = infos["observations"].get("commands") 
                step_count += 1
                
                # 每100步打印一次进度 / Print progress every 100 steps
                if step_count % 100 == 0:
                    print(f"[INFO] Playback step: {step_count}")
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        # close the simulator (RecordVideo 会在 env.close() 时自动保存视频)
        # Close simulator (RecordVideo will automatically save video when env.close() is called)
        env.close()
        print(f"[INFO] Environment closed after {step_count} steps.")
        if args_cli.video and video_folder:
            print(f"[INFO] Video should be saved to: {video_folder}")
            print(f"[INFO] Please check for rl-video-step-0.mp4 in the video folder.")


if __name__ == "__main__":
    EXPORT_POLICY = True
    # 运行主程序 / Run the main execution
    main()
    # 关闭仿真应用 / Close sim app
    simulation_app.close()
