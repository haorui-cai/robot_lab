import gymnasium as gym

from . import agents

# Independent controls; existing registrations and deployment defaults stay unchanged.
for suffix, module, cls in (
    ("RewardV4Scratch", "reward_v4_env_cfg", "TQBot2RewardV4EnvCfg"),
    ("FreeGaitScratch", "free_gait_env_cfg", "TQBot2FreeGaitEnvCfg"),
):
    gym.register(
        id=f"RobotLab-Isaac-Velocity-Rough-TQBot2-{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{module}:{cls}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TQBot2StandardWalkPPORunnerCfg",
        },
    )

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-TQBot2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TQBot2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TQBot2FlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-TQBot2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:TQBot2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TQBot2RoughPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standard_walk_env_cfg:TQBot2StandardWalkRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TQBot2StandardWalkPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-Climb-BlindBox-TQBot2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.blind_box_env_cfg:TQBot2BlindBoxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.blind_box_ppo_cfg:TQBot2BlindBoxPPORunnerCfg",
    },
)
