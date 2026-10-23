"""Validate physical slope resets, mixed curricula and finite simulation."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils import parse_env_cfg

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.roll.mdp import hip_only_cycle as cycle, slopes

task = "RobotLab-Isaac-Roll-LeftThigh-Slope-TQBot2-v0"
cfg = parse_env_cfg(task, device=args.device, num_envs=64)
cfg.seed = 42
cfg.commands.base_velocity.debug_vis = False
env = gym.make(task, cfg=cfg)
try:
    raw = env.unwrapped
    env.reset()
    terrain = raw.scene.terrain
    robot = raw.scene["robot"]
    ids = torch.arange(raw.num_envs, device=raw.device)
    assert terrain.slope_degrees.count_nonzero() == 0
    assert terrain.slope_origins.shape == (41, 3)
    assert raw.action_manager.total_action_dim == 6
    assert cycle.tracker(raw).actual_window is not None
    assert cycle.tracker(raw).yaw_min_speed == 6.0
    assert cycle.tracker(raw).yaw_tracking_std == 0.125
    assert cycle.tracker(raw).yaw_success_tolerance == 0.06
    assert cycle.tracker(raw).legacy_reward_fraction == 0.5
    assert set(raw.reward_manager.active_terms) == {
        "roll_cycle_tracking", "yaw_cycle_tracking", "joint_power",
        "base_linear_acceleration", "base_angular_acceleration",
    }
    assert set(raw.curriculum_manager.active_terms) == {"rolling", "slopes"}
    # Explicitly exercise both extreme ramps without waiting for PPO to learn.
    angles = torch.tensor([-30.0, 10.0, 0.0, -15.0], device=raw.device).repeat(16)
    terrain.assign_slopes(ids, angles)
    cycle.reset_rolling_phase(raw, ids, "left_thigh_roll_spawn_heights.json", torch.deg2rad(angles))
    normals = slopes.slope_normals(raw)
    relative = robot.data.root_pos_w - terrain.env_origins
    height = (relative * normals).sum(-1)
    assert height.min() > 0.09 and height.max() < 0.15
    tangent = torch.stack((torch.zeros_like(angles), torch.deg2rad(angles).cos(), torch.deg2rad(angles).sin()), -1)
    torch.testing.assert_close((relative * tangent).sum(-1), torch.zeros_like(angles), atol=5e-5, rtol=0)
    torch.testing.assert_close(robot.data.root_lin_vel_w, 0.8 * tangent, atol=1e-4, rtol=0)
    roll, _, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    error = (roll - torch.deg2rad(angles) + 2 + torch.pi) % (2 * torch.pi) - torch.pi
    assert error.abs().max() < 0.1001
    assert not slopes.invalid_slope_pose(raw).any(), "Valid ramp spawn rejected by absolute world height"
    assert not slopes.slope_out_of_bounds(raw).any()
    # Contact must develop on both real ramp orientations.
    max_contact = torch.zeros(64, device=raw.device)
    for _ in range(16):
        _, rewards, _, _, _ = env.step(torch.zeros(64, 6, device=raw.device))
        assert torch.isfinite(rewards).all()
        max_contact = torch.maximum(max_contact, raw.scene["contact_forces"].data.net_forces_w.norm(dim=-1).amax(-1))
    assert max_contact[angles == 10].max() > 5
    assert max_contact[angles == -30].max() > 5
    # Validate reset sampling at the full envelope, while keeping flat rehearsal.
    for name in terrain.slope_course.levels:
        terrain.slope_course.levels[name] = len(terrain.slope_course.limits[name]) - 1
    sampled = []
    for _ in range(8):
        raw._reset_idx(ids)
        sampled.append(terrain.slope_degrees.clone())
    sampled = torch.cat(sampled)
    assert sampled.min() == -30 and sampled.max() == 10
    assert (sampled == 0).float().mean() > 0.15
    reward_sum = 0.0
    for _ in range(128):
        obs, reward, _, _, _ = env.step(torch.randn(64, 6, device=raw.device).clamp(-1, 1))
        assert torch.isfinite(reward).all()
        assert all(value.shape == (64, 35) and torch.isfinite(value).all() for value in obs.values())
        reward_sum += reward.sum().item()
    untouched = robot.data.root_pos_w[1:].clone()
    histories = [raw.reward_manager.get_term_cfg(name).func.history for name in (
        "base_linear_acceleration", "base_angular_acceleration"
    )]
    other_accelerations = [h.acceleration[1:].clone() for h in histories]
    raw._reset_idx(ids[:1])
    for history, other in zip(histories, other_accelerations):
        assert not history.valid[0]
        assert history.acceleration[0].count_nonzero() == 0
        torch.testing.assert_close(history.acceleration[1:], other)
    torch.testing.assert_close(robot.data.root_pos_w[1:], untouched)
    assert cycle.tracker(raw).roll_acceleration[0] == cycle.tracker(raw).yaw_acceleration[0] == 0
    assert not cycle.tracker(raw).actual_window.valid[0]
    assert cycle.tracker(raw).actual_window.mean[0].count_nonzero() == 0
    print(f"SLOPE_SMOKE_OK sampled_range=[{sampled.min().item()}, {sampled.max().item()}] reward_sum={reward_sum:.6f}", flush=True)
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    env.close()
    app.close()
