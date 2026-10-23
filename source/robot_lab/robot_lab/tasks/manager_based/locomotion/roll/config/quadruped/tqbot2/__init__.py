# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

gym.register(
    id="RobotLab-Isaac-Roll-LeftThigh-Slope-TQBot2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.left_thigh_slope_env_cfg:TQBot2RollLeftThighSlopeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TQBot2RollLeftThighSlopePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Roll-LeftThigh-Cycle-TQBot2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.left_thigh_cycle_env_cfg:TQBot2RollLeftThighCycleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TQBot2RollLeftThighCyclePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Roll-HipOnly-Cycle-TQBot2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hip_only_cycle_env_cfg:TQBot2RollHipOnlyCycleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TQBot2RollHipOnlyCyclePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Roll-Rough-TQBot2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:TQBot2RollRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TQBot2RollPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Roll-HipOnly-TQBot2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hip_only_env_cfg:TQBot2RollHipOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TQBot2RollHipOnlyPPORunnerCfg",
    },
)
