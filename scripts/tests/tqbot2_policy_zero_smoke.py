"""Verify independent policy/standing references using real managers and PhysX."""

import argparse
import json
import re
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--policy-zero", choices=("legacy", "new"), default="legacy")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import robot_lab.tasks  # noqa: F401
from robot_lab.assets.tqbot_walk_stance import WALK_LEGACY_POLICY_ZERO_JOINT_POS, WALK_POLICY_ZERO_JOINT_POS
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import (
    TQBot2StandardWalkRoughEnvCfg,
)


def main():
    cfg = TQBot2StandardWalkRoughEnvCfg()
    reference = WALK_LEGACY_POLICY_ZERO_JOINT_POS if args.policy_zero == "legacy" else WALK_POLICY_ZERO_JOINT_POS
    if args.policy_zero == "new":
        cfg.actions.joint_pos.offset = reference.copy()
        for observations in (cfg.observations.policy, cfg.observations.critic):
            observations.joint_pos.params["reference_joint_pos"] = reference.copy()
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.height_scanner = None
    cfg.scene.height_scanner_base = None
    cfg.observations.critic.height_scan = None
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.terrain_breakdown = None
    cfg.events.randomize_push_robot = None
    cfg.events.randomize_apply_external_force_torque = None
    cfg.events.randomize_reset_base.params = {"pose_range": {}, "velocity_range": {}}
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.heading_command = False
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    cfg.log_dir = None
    env = gym.make("RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0", cfg=cfg).unwrapped
    try:
        env.reset(seed=42)
        robot = env.scene["robot"]
        action = env.action_manager.get_term("joint_pos")
        zeros = torch.zeros(env.num_envs, 12, device=env.device)
        standing = robot.data.default_joint_pos.clone()
        policy_zero = torch.tensor([
            next(value for pattern, value in reference.items() if re.fullmatch(pattern, name))
            for name in robot.joint_names
        ], device=env.device).expand_as(standing)
        thigh_ids = [i for i, name in enumerate(robot.joint_names) if "_thigh_" in name]
        torch.testing.assert_close(standing[:, thigh_ids], torch.full_like(standing[:, thigh_ids], .8))
        expected_thigh = .8 if args.policy_zero == "legacy" else 1.35
        torch.testing.assert_close(policy_zero[:, thigh_ids], torch.full_like(policy_zero[:, thigh_ids], expected_thigh))
        torch.testing.assert_close(robot.data.joint_pos, standing)

        action.process_actions(zeros)
        torch.testing.assert_close(action.processed_actions, policy_zero[:, action._joint_ids])
        standing_action = (standing[:, action._joint_ids] - action._offset) / action._scale
        action.process_actions(standing_action)
        torch.testing.assert_close(action.processed_actions, standing[:, action._joint_ids])
        action_thigh_ids = [i for i, name in enumerate(action._joint_names) if "_thigh_" in name]
        torch.testing.assert_close(standing_action[:, action_thigh_ids],
                                   torch.full_like(standing_action[:, action_thigh_ids], 0. if args.policy_zero == "legacy" else -2.2))

        command = env.command_manager.get_term("base_velocity")
        command.vel_command_b[:] = 0
        command.is_standing_env[:] = True
        command.is_heading_env[:] = False

        def reward_value(name):
            term = env.reward_manager.get_term_cfg(name)
            return term.func(env, **term.params)

        def check_observations(expected):
            for group in ("policy", "critic"):
                manager = env.observation_manager
                index = manager.active_terms[group].index("joint_pos")
                term = manager._group_obs_term_cfgs[group][index]
                actual = term.func(env, **term.params)
                torch.testing.assert_close(actual, expected[:, term.params["asset_cfg"].joint_ids])

        check_observations(standing - policy_zero)
        for name in ("stand_still", "joint_pos_penalty"):
            torch.testing.assert_close(reward_value(name), torch.zeros(env.num_envs, device=env.device))

        # Changing the policy coordinate origin cannot move either reward's
        # minimum or the standing/reset reference.
        original_offset = action._offset.clone()
        action._offset += .1
        for name in ("stand_still", "joint_pos_penalty"):
            torch.testing.assert_close(reward_value(name), torch.zeros(env.num_envs, device=env.device))
        torch.testing.assert_close(robot.data.default_joint_pos, standing)
        action._offset.copy_(original_offset)

        # The retained new zero differs from the reward minimum; the legacy
        # zero coincides with it. Check both profiles against the real terms.
        robot.write_joint_state_to_sim(policy_zero, torch.zeros_like(policy_zero))
        env.sim.forward()
        env.scene.update(env.physics_dt)
        check_observations(torch.zeros_like(standing))
        l1 = 0. if args.policy_zero == "legacy" else 2.2
        l2 = 0. if args.policy_zero == "legacy" else 1.1
        torch.testing.assert_close(reward_value("stand_still"), torch.full((env.num_envs,), l1, device=env.device))
        torch.testing.assert_close(reward_value("joint_pos_penalty"), torch.full((env.num_envs,), 5*l2, device=env.device))
        torch.testing.assert_close(reward_value("joint_pos_limits"), torch.zeros(env.num_envs, device=env.device))
        command.vel_command_b[:, 0] = .5
        torch.testing.assert_close(reward_value("stand_still"), torch.zeros(env.num_envs, device=env.device))
        torch.testing.assert_close(reward_value("joint_pos_penalty"), torch.full((env.num_envs,), l2, device=env.device))
        height = env.reward_manager.get_term_cfg("feet_height_body")
        assert abs(height.params["target_height"] + .165) < 1e-12 and height.weight == -5.

        env._reset_idx(torch.tensor([0, 2], device=env.device))
        torch.testing.assert_close(robot.data.joint_pos[[0, 2]], standing[[0, 2]])
        torch.testing.assert_close(action._offset, original_offset)
        env.reset()
        for step in range(40):
            raw = standing_action + .05 * torch.sin(torch.full_like(standing_action, step * .2))
            obs, reward, _, _, _ = env.step(raw)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all() and torch.isfinite(env.reward_manager._step_reward).all()

        report = {
            "policy_zero_profile": args.policy_zero,
            "device": env.device, "num_envs": env.num_envs, "physics_steps": 40,
            "joint_names": robot.joint_names,
            "standing_joint_pos": standing[0].cpu().tolist(),
            "policy_zero_joint_pos": policy_zero[0].cpu().tolist(),
            "action_joint_names": action._joint_names,
            "standing_action": standing_action[0].cpu().tolist(),
            "checks": {name: True for name in (
                "thigh_zero_matches_selected_profile", "zero_action_targets_policy_zero",
                "standing_action_targets_standing_pose", "actor_and_critic_reference_joint_order",
                "rewards_minimized_at_standing_pose", "reward_reference_independent_of_action_offset",
                "policy_zero_pose_has_expected_standing_penalties", "reward_command_gates_preserved",
                "foot_height_target_preserved", "partial_reset_preserves_standing_pose_and_policy_zero",
                "finite_observations_rewards_and_steps",
            )},
            "note": "Reference semantics and integration checks only; no training or walking-performance claim.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS: independent policy/standing references, rewards, observations, reset and stepping", flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
