# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

for suffix, config in (("", "TQBot2RecoverEnvCfg"), ("-Play", "TQBot2RecoverEnvCfg_PLAY")):
    gym.register(
        id=f"RobotLab-Isaac-Recover-Flat-TQBot2{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.recover_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TQBot2RecoverPPORunnerCfg",
        },
    )
