import math

from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.pointfoot_cfg import POINTFOOT_CFG
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_base_env_cfg import PFEnvCfg
from bipedal_locomotion.tasks.locomotion.cfg.PF.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
)

from isaaclab.sensors import RayCasterCfg, patterns
from bipedal_locomotion.tasks.locomotion import mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg


######################
# 双足机器人基础环境 / Pointfoot Base Environment
######################


@configclass
class PFBaseEnvCfg(PFEnvCfg):
    """双足机器人基础环境配置 - 所有变体的共同基础 / Base environment configuration for pointfoot robot - common foundation for all variants"""
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = POINTFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }
        # 调整基座质量随机化参数 / Adjust base mass randomization parameters
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        # 设置基座接触终止条件 / Set base contact termination condition
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"
        
        # 更新视口相机设置 / Update viewport camera settings
        self.viewer.origin_type = "env"  # 相机跟随环境 / Camera follows environment


@configclass
class PFBaseEnvCfg_PLAY(PFBaseEnvCfg):
    """双足机器人基础测试环境配置 - 用于策略评估 / Base play environment configuration - for policy evaluation"""
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


############################
# 双足机器人盲视平地环境 / Pointfoot Blind Flat Environment
############################


@configclass
class PFBlindFlatEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None
        
        # 增强抗冲击训练 - 增加扰动频率和强度 / Enhance impact resistance training - increase disturbance frequency and intensity
        # 减小间隔时间，增加扰动频率 / Reduce interval time, increase disturbance frequency
        self.events.push_robot.interval_range_s = (3.0, 6.0)  # 从 (10.0, 15.0) 减少到 (3.0, 6.0)
        # 增加扰动概率 / Increase disturbance probability
        self.events.push_robot.params["probability"] = 0.05  # 从 0.005 增加到 0.05 (10倍)
        # 可以适当增加力的范围以增强抗冲击能力 / Can increase force range to enhance impact resistance
        # self.events.push_robot.params["force_range"]["x"] = (-1000.0, 1000.0)  # 可选：进一步增加
        # self.events.push_robot.params["force_range"]["y"] = (-1000.0, 1000.0)
        
        # 优化步态稳定性和平滑性 - 解决"跳着走"和"不稳定"问题 / Optimize gait stability and smoothness - fix "hopping" and "unstable" issues
        # 1. 增强高度稳定性 / Enhance height stability
        self.rewards.pen_base_height.weight = -30.0  # 从 -20.0 增加到 -30.0（更严格的高度控制，减少跳跃）
        # 2. 增强姿态稳定性 / Enhance orientation stability  
        self.rewards.pen_flat_orientation.weight = -15.0  # 从 -10.0 增加到 -15.0（更严格的姿态要求）
        # 3. 减少垂直速度（防止跳跃） / Reduce vertical velocity (prevent hopping)
        self.rewards.pen_lin_vel_z.weight = -1.0  # 从 -0.5 增加到 -1.0（更严格惩罚垂直运动）
        # 4. 增强着陆柔顺性 / Enhance landing softness
        self.rewards.foot_landing_vel.weight = -0.5  # 从 -0.2 增加到 -0.5（减少砸地）
        # 5. 增强动作平滑性 / Enhance action smoothness
        self.rewards.pen_action_rate.weight = -0.05  # 从 -0.03 增加到 -0.05（减少动作突变）
        self.rewards.pen_action_smoothness.weight = -0.08  # 从 -0.04 增加到 -0.08（更平滑的动作）
        # 6. 增强关节平滑性 / Enhance joint smoothness
        self.rewards.pen_joint_vel_l2.weight = -0.002  # 从 -0.001 增加到 -0.002（减少关节速度突变）
        # 7. 保持平衡奖励（稳定站立） / Keep balance reward (stable standing)
        self.rewards.keep_balance.weight = 3.0  # 从 1.0 增加到 3.0（增强稳定性优先）
        # 8. 适度降低速度跟踪权重（稳定性优先于速度） / Slightly reduce velocity tracking weight (stability over speed)
        self.rewards.rew_lin_vel_xy.weight = 5.0  # 从 7.0 降低到 5.0（减少为了追速度而跳跃）
        self.rewards.rew_ang_vel_z.weight = 3.0  # 从 4.0 降低到 3.0


@configclass
class PFBlindFlatEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


#############################
# 双足机器人盲视粗糙环境 / Pointfoot Blind Rough Environment
#############################

import math
from isaaclab.utils import configclass
from bipedal_locomotion.tasks.locomotion import mdp


# @configclass
# class PFBlindRoughEnvCfg(PFBaseEnvCfg):
#     """Blind Rough Env: 复杂地形 + 不输入高度观测（policy/critic 都不看 heights）"""

#     def __post_init__(self):
#         super().__post_init__()

#         # -------------------------
#         # 1) Blind：不使用高度扫描/高度观测
#         # -------------------------
#         self.scene.height_scanner = None
#         self.observations.policy.heights = None
#         self.observations.critic.heights = None

#         # -------------------------
#         # 2) Rough terrain：启用 generator
#         # -------------------------
#         self.scene.terrain.terrain_type = "generator"
#         self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG

#         # 不开启地形课程
#         if hasattr(self, "curriculum") and hasattr(self.curriculum, "terrain_levels"):
#             self.curriculum.terrain_levels = None

#         # -------------------------
#         # 3) 速度指令范围：rough 先收窄，稳了再放开
#         # -------------------------
#         self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
#             lin_vel_x=(0.0, 0.6),         # 先练前进为主
#             lin_vel_y=(-0.2, 0.2),        # 横向先小
#             ang_vel_z=(-0.6, 0.6),        # 转向先小
#             heading=(-math.pi, math.pi),
#         )

#         # -------------------------
#         # 4) Reward 权重：只做小幅偏置（稳、落脚软、少抖）
#         # -------------------------
#         # 存活奖励更重要一点
#         self.rewards.keep_balance.weight = 2.0            # 1.0 -> 2.0

#         # rough 更容易 roll/pitch 摇，稍微更严格
#         self.rewards.pen_flat_orientation.weight = -12.0  # -10 -> -12

#         # 高度稍微更严格，减少“蹲-弹-跳”
#         self.rewards.pen_base_height.weight = -22.0       # -20 -> -22

#         # 落脚软一点（如果有这个项就改）
#         if hasattr(self.rewards, "foot_landing_vel"):
#             self.rewards.foot_landing_vel.weight = -0.25  # -0.2 -> -0.25

#         # 平滑项只加一点点（你这个文件里是 pen_action_smoothness）
#         if hasattr(self.rewards, "pen_action_smoothness"):
#             self.rewards.pen_action_smoothness.weight = -0.05  # -0.04 -> -0.05

#         # 追踪项 rough 初期别太追（稳了再加回去）
#         self.rewards.rew_lin_vel_xy.weight = 5.0          # 7.0 -> 5.0
#         self.rewards.rew_ang_vel_z.weight = 3.0           # 4.0 -> 3.0

#         # yaw 漂移稍微更严格（有就改）
#         if hasattr(self.rewards, "pen_yaw_drift"):
#             self.rewards.pen_yaw_drift.weight = -0.25     # -0.2 -> -0.25


#############################
# 双足机器人盲视粗糙环境 / 修复抖动和速度追踪不稳
#############################

@configclass
class PFBlindRoughEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # blind：不要 height scanner
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # rough terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG

        # ✅（可选但推荐）开启地形课程
        # 如果你原来 rough 把 curriculum 关了，这里就打开
        # self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        # 或者如果 base 里已经有了，就别覆盖 None

        # ✅ 速度命令范围：rough 先收一点，稳定后再放开
        self.commands.base_velocity.ranges.lin_vel_x = (-1.2, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # ✅ 命令重采样别太快（减少急变导致的抖动）
        self.commands.base_velocity.resampling_time_range = (1.0, 5.0)

        
        self.rewards.pen_action_rate.weight = -0.02
        self.rewards.pen_action_smoothness.weight = -0.06
        self.rewards.pen_joint_accel.weight = -5e-07
        self.rewards.foot_landing_vel.weight = -0.35

        # tracking reward：更重、更严格
        self.rewards.rew_lin_vel_xy.weight = 8.0
        self.rewards.rew_lin_vel_xy.params["std"] = math.sqrt(0.12)

        # ✅ yaw 跟踪：优先加 drift 惩罚（更稳）
        self.rewards.pen_yaw_drift.weight = -0.30
        self.rewards.rew_ang_vel_z.weight = 5.0
        

        # （可选）如果你觉得转向还是跟不住，再加一点 tracking
        # self.rewards.rew_ang_vel_z.weight = 5.0




@configclass
class PFBlindRoughEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG


##############################
# 双足机器人盲视楼梯环境 / Pointfoot Blind Stairs Environment
##############################


@configclass
class PFBlindStairEnvCfg(PFBaseEnvCfg):
    """盲视楼梯环境配置 - 专门训练爬楼梯能力 / Blind stairs environment configuration - specialized for stair climbing training"""
    
    def __post_init__(self):
        """后初始化 - 配置楼梯训练环境 / Post-initialization - configure stairs training environment"""
        super().__post_init__()
        
        # 移除视觉组件 / Remove vision components
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # 调整速度命令范围以适应楼梯环境 / Adjust velocity command ranges for stairs environment
        self.commands.base_velocity.ranges.lin_vel_x = (0, 0.3)      # 前进速度：0.5-1.0 m/s / Forward velocity: 0.5-1.0 m/s
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)     # 横向速度：0（仅直行）/ Lateral velocity: 0 (straight only)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)  # 转向：±30度 / Turning: ±30 degrees

        # 调整奖励权重以适应楼梯爬升 / Adjust reward weights for stair climbing
        self.rewards.rew_lin_vel_xy.weight = 2.0          # 增加线速度跟踪奖励 / Increase linear velocity tracking reward
        self.rewards.rew_ang_vel_z.weight = 1.5           # 增加角速度跟踪奖励 / Increase angular velocity tracking reward
        self.rewards.pen_lin_vel_z.weight = -1.0          # 增加Z方向速度惩罚 / Increase Z velocity penalty
        self.rewards.pen_ang_vel_xy.weight = -0.05        # XY角速度惩罚 / XY angular velocity penalty
        self.rewards.pen_action_rate.weight = -0.01       # 动作变化率惩罚 / Action rate penalty
        self.rewards.pen_flat_orientation.weight = -2.5   # 姿态保持惩罚 / Orientation keeping penalty
        self.rewards.pen_undesired_contacts.weight = -1.0 # 不期望接触惩罚 / Undesired contact penalty

        # 设置楼梯地形 / Set up stairs terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG

@configclass
class PFBlindStairEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    """盲视楼梯测试环境配置 / Blind stairs play environment configuration"""
    
    def __post_init__(self):
        """后初始化 - 配置楼梯测试环境 / Post-initialization - configure stairs testing environment"""
        super().__post_init__()
        
        # 移除视觉组件 / Remove vision components
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # 设置测试专用的速度命令 / Set testing-specific velocity commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)    # 固定前进速度范围 / Fixed forward velocity range
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)   # 无横向移动 / No lateral movement
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)   # 无转向 / No turning

        # 固定重置姿态（无偏航角变化）/ Fixed reset pose (no yaw variation)
        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # 设置测试楼梯地形 / Set up testing stairs terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        # 设置中等难度的楼梯测试环境 / Set medium difficulty stairs testing environment
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))


#############################
# 带高度扫描的双足机器人楼梯环境 / Pointfoot Stairs Environment with Height Scanning
#############################

@configclass
class PFStairEnvCfgv1(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
                    clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class PFStairEnvCfgv1_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))