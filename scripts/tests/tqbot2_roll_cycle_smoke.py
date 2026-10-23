"""Validate phase/yaw curricula, assisted resets and full-cycle rewards in Isaac."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--left_thighs", action="store_true", help="Validate the six-action left-thigh variant.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils import parse_env_cfg

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.roll.mdp.hip_only_cycle import INITIAL_ROLL_RATE, course, tracker

task = "RobotLab-Isaac-Roll-LeftThigh-Cycle-TQBot2-v0" if args.left_thighs else "RobotLab-Isaac-Roll-HipOnly-Cycle-TQBot2-v0"
action_dim = 6 if args.left_thighs else 4
obs_dim = 35 if args.left_thighs else 29
cfg = parse_env_cfg(task, device=args.device, num_envs=32)
cfg.seed = 42
env = gym.make(task, cfg=cfg)
try:
    raw = env.unwrapped
    robot = raw.scene["robot"]
    ids = torch.arange(raw.num_envs, device=raw.device)
    env.reset()
    assert raw.action_manager.total_action_dim == robot.num_joints == action_dim
    expected_rewards = {"roll_cycle_tracking", "yaw_cycle_tracking", "base_linear_acceleration", "base_angular_acceleration"}
    if args.left_thighs:
        expected_rewards |= {"joint_power"}
    assert set(raw.reward_manager.active_terms) == expected_rewards
    yaw_weight = 5.0 if args.left_thighs else 2.0
    assert raw.reward_manager.get_term_cfg("yaw_cycle_tracking").weight == yaw_weight
    if args.left_thighs:
        thigh_ids, names = robot.find_joints(".*_thigh_joint")
        assert set(names) == {"FL_thigh_joint", "RL_thigh_joint"}
        limits = robot.root_physx_view.get_dof_limits()[:, thigh_ids]
        torch.testing.assert_close(limits[..., 0], torch.full_like(limits[..., 0], 1.68))
        torch.testing.assert_close(limits[..., 1], torch.full_like(limits[..., 1], 2.45))
        left_contact_ids = raw._roll_support_ids[0]
        left_contact_names = {raw.scene["contact_forces"].body_names[i] for i in left_contact_ids}
        assert left_contact_names == {"FL_hip", "RL_hip", "FL_thigh", "RL_thigh"}

    def check_reset(reset_ids, width):
        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w[reset_ids])
        error = (roll + 2 + torch.pi) % (2 * torch.pi) - torch.pi
        assert error.abs().max() <= width + 1e-5
        assert torch.allclose(pitch, torch.zeros_like(pitch), atol=1e-5)
        assert torch.allclose(yaw, torch.zeros_like(yaw), atol=1e-5)
        scale = course(raw).initial_speed_scale
        assert torch.allclose(robot.data.root_ang_vel_b[reset_ids, 0], torch.full_like(roll, INITIAL_ROLL_RATE * scale), atol=1e-4)
        assert torch.allclose(robot.data.root_lin_vel_w[reset_ids, 1], torch.full_like(roll, 0.8 * scale), atol=1e-4)
        assert torch.allclose(robot.data.root_ang_vel_w[reset_ids, 2], torch.zeros_like(roll), atol=1e-4)
        assert torch.allclose(robot.data.joint_pos[reset_ids], robot.data.default_joint_pos[reset_ids])
        assert tracker(raw).cycles[reset_ids].sum() == 0
        assert tracker(raw).roll_payment[reset_ids].sum() == 0
        assert tracker(raw).roll_acceleration[reset_ids].count_nonzero() == 0
        assert tracker(raw).yaw_acceleration[reset_ids].count_nonzero() == 0
        return error

    error = check_reset(ids, 0.10)
    assert error.std() > 0.01, "Reset phase did not randomize"
    command = raw.command_manager.get_command("base_velocity")
    assert command.shape == (32, 2)
    low, high = (-14.0, -6.0) if args.left_thighs else (-1 / 0.112, -0.7 / 0.112)
    assert torch.all((command[:, 0] >= low) & (command[:, 0] <= high))
    assert command[:, 0].min() < low + (high - low) * 0.25
    assert command[:, 1].count_nonzero() == 0

    # Verify goal positions, including extreme raw outputs, on both sides.
    term = raw.action_manager.get_term("joint_pos")
    for action in (-100.0, 0.0, 100.0):
        raw.action_manager.process_action(torch.full((32, action_dim), action, device=raw.device))
        target = term.processed_actions
        for index, name in enumerate(term._joint_names):
            offset = 2.065 if "thigh" in name else (0.52 if "R_hip" in name else -0.52)
            scale = 0.385 if "thigh" in name else 1.0
            expected = min(offset + scale, max(offset - scale, offset + scale * action))
            assert torch.allclose(target[:, index], torch.full((32,), expected, device=raw.device))

    # Exercise the maximum curriculum envelope without relying on training success.
    level = course(raw)
    if args.left_thighs:
        for assist_level in range(len(level.initial_speed_scales)):
            level.assist_level = assist_level
            raw._reset_idx(ids)
            check_reset(ids, 0.10)
        assert robot.data.root_vel_w.count_nonzero() == 0
        # Check zero-assistance physics/rewards, including fresh derivative history.
        for _ in range(8):
            _, rewards, _, _, _ = env.step(torch.zeros(32, action_dim, device=raw.device))
            assert torch.isfinite(rewards).all()
        level.assist_level = 0
    level.phase_level = len(level.phase_widths) - 1
    level.yaw_level = len(level.yaw_widths) - 1
    raw._reset_idx(ids)
    check_reset(ids, torch.pi)
    assert torch.isfinite(robot.data.root_pos_w).all()
    assert robot.data.root_pos_w[:, 2].min() > 0.09
    assert robot.data.root_pos_w[:, 2].max() < 0.15
    assert raw.command_manager.get_command("base_velocity")[:, 1].abs().max() <= 0.5
    assert raw.command_manager.get_command("base_velocity")[:, 1].std() > 0.05
    level.phase_level = level.yaw_level = 0
    raw._reset_idx(ids)
    check_reset(ids, 0.1)
    total_reward = 0.0
    tracking_reward = 0.0
    reset_count = 0
    thigh_min, thigh_max = float("inf"), -float("inf")
    penalty_totals = {name: 0.0 for name in (
        "base_linear_acceleration", "base_angular_acceleration"
    )}
    if args.left_thighs:
        penalty_totals["joint_power"] = 0.0
    for step in range(256):
        actions = torch.randn(32, action_dim, device=raw.device).clamp(-1, 1)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(rewards).all()
        for value in observations.values():
            assert value.shape[-1] == obs_dim
            assert torch.isfinite(value).all()
        total_reward += rewards.sum().item()
        tracking_reward += (8 * tracker(raw).roll_payment + yaw_weight * tracker(raw).yaw_payment).sum().item() * raw.step_dt
        reset_count += (terminated | truncated).sum().item()
        for name in penalty_totals:
            reward_cfg = raw.reward_manager.get_term_cfg(name)
            penalty = reward_cfg.func(raw, **reward_cfg.params) * reward_cfg.weight
            assert torch.isfinite(penalty).all() and (penalty <= 0).all()
            penalty_totals[name] += penalty.sum().item()
        if args.left_thighs:
            thigh_min = min(thigh_min, robot.data.joint_pos[:, thigh_ids].min().item())
            thigh_max = max(thigh_max, robot.data.joint_pos[:, thigh_ids].max().item())
        if step == 128:
            subset = torch.tensor([0, 3, 17], device=raw.device)
            raw._reset_idx(subset)
            check_reset(subset, 0.1)
    print(f"CYCLE_SMOKE reward_sum={total_reward:.6f} resets={reset_count} initial_roll_rate={INITIAL_ROLL_RATE:.6f}")
    assert reset_count > 0
    # Signed total can be negative under random actions; tracking credit must still exist.
    print(f"TRACKING_CHECK reward_sum={tracking_reward:.6f}")
    assert tracking_reward > 1.0, "Assisted starts produced no useful average-speed reward signal"
    if args.left_thighs:
        print(f"PENALTY_CHECK totals_before_dt={penalty_totals}")
        assert all(value < 0 for value in penalty_totals.values())
        print(f"LEFT_THIGH_CHECK actual_joint_range=[{thigh_min:.6f}, {thigh_max:.6f}]")
        assert thigh_max > 1.9, "Left thighs did not move"
        assert thigh_min > 1.65 and thigh_max < 2.48, "Excessive solver violation of thigh limits"
    print("ROLL_CYCLE_SMOKE_OK", flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    env.close()
    app.close()
