# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Compare a checkpoint with zero actions from identical, explicitly supine starts."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--output", default="outputs/recover_validation/supine_evaluation.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import importlib.metadata
import json
from pathlib import Path

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from robot_lab.tasks.manager_based.locomotion.recover import mdp
from robot_lab.tasks.manager_based.locomotion.recover.config.tqbot2.agents.rsl_rl_ppo_cfg import (
    TQBot2RecoverPPORunnerCfg,
)
from robot_lab.tasks.manager_based.locomotion.recover.config.tqbot2.recover_env_cfg import TQBot2RecoverEnvCfg_PLAY
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

cfg = TQBot2RecoverEnvCfg_PLAY()
cfg.seed = 2026
cfg.scene.num_envs = args.num_envs
cfg.terminations.success = None
cfg.terminations.time_out = None
cfg.terminations.escaped = None
cfg.rewards.success_bonus = None
env = gym.make("RobotLab-Isaac-Recover-Flat-TQBot2-Play-v0", cfg=cfg).unwrapped
wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
agent = handle_deprecated_rsl_rl_cfg(TQBot2RecoverPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)
result = {"checkpoint": str(Path(args.checkpoint).resolve()), "num_envs": args.num_envs,
          "horizon_seconds": 16.0, "quiet_hold_seconds": 4.0, "criterion_version": "legs_down_quiet_hold_v3"}

with torch.inference_mode():
    for mode in ("zero_action", "policy"):
        env.reset(seed=2026)
        obs = wrapped.get_observations()
        gravity_z = env.scene["robot"].data.projected_gravity_b[:, 2]
        assert (gravity_z > 0.99).all(), "Every evaluation episode must start feet-up"
        assert not mdp.is_prone(env).any()
        start_range = [gravity_z.min().item(), gravity_z.max().item()]
        streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        first_time = torch.full((env.num_envs,), float("nan"), device=env.device)
        final_pose, final_speed, final_target_rate = [], [], []
        previous_target = None
        for step in range(800):
            actions = torch.zeros(env.num_envs, 12, device=env.device) if mode == "zero_action" else policy(obs)
            obs, _, _, _ = wrapped.step(actions)
            good = mdp.is_quiet_prone(env) & (env.episode_length_buf >= 50)
            velocity = env.scene["robot"].data.joint_vel
            streak = torch.where(good, streak + 1, 0)
            first = (streak >= 200) & first_time.isnan()
            first_time[first] = (step + 1) * env.step_dt
            target = env.action_manager.get_term("joint_pos").processed_actions.clone()
            if step >= 600:
                final_pose.append(mdp.is_prone(env).clone())
                final_speed.append(velocity.square().mean(-1).sqrt())
                final_target_rate.append(((target - previous_target) / env.step_dt).square().mean(-1).sqrt())
            previous_target = target
        succeeded = first_time.isfinite()
        result[mode] = {
            "initial_gravity_z_range": start_range,
            "successes": succeeded.sum().item(),
            "success_rate": succeeded.float().mean().item(),
            "mean_recovery_seconds": first_time[succeeded].mean().item() if succeeded.any() else None,
            "final_upright_fraction": (env.scene["robot"].data.projected_gravity_b[:, 2] < -0.95).float().mean().item(),
            "final_4s_prone_fraction": torch.stack(final_pose).float().mean().item(),
            "final_4s_joint_rms_speed_rad_s": torch.stack(final_speed).mean().item(),
            "final_4s_target_rms_rate_rad_s": torch.stack(final_target_rate).mean().item(),
        }
        print("EVALUATION", mode, result[mode], flush=True)
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
env.close()
app.close()
