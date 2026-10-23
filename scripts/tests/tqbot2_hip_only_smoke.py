"""Exercise hip-only actions, locked leg physics and partial episode resets."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils import parse_env_cfg

import robot_lab.tasks  # noqa: F401

task = "RobotLab-Isaac-Roll-HipOnly-TQBot2-v0"
cfg = parse_env_cfg(task, device=args.device, num_envs=32)
cfg.seed = 42
env = gym.make(task, cfg=cfg)
try:
    raw = env.unwrapped
    robot = raw.scene["robot"]
    hips, _ = robot.find_joints(".*_hip_joint")
    assert raw.action_manager.total_action_dim == 4
    assert robot.num_joints == 4
    assert all("_hip_" in name for name in robot.joint_names)
    assert set(robot.body_names) == {"trunk", "FR_hip", "FL_hip", "RR_hip", "RL_hip"}
    env.reset()

    def check_reset(ids):
        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w[ids])
        roll = (roll + torch.pi) % (2 * torch.pi) - torch.pi
        assert torch.allclose(roll, torch.full_like(roll, -2.0), atol=1e-5)
        assert torch.allclose(pitch, torch.zeros_like(pitch), atol=1e-5)
        assert torch.allclose(yaw, torch.zeros_like(yaw), atol=1e-5)
        assert torch.allclose(robot.data.root_pos_w[ids, 2], torch.full_like(roll, 0.113), atol=1e-5)
        assert torch.count_nonzero(robot.data.root_vel_w[ids]) == 0
        assert torch.allclose(robot.data.joint_pos[ids], robot.data.default_joint_pos[ids])
        assert torch.count_nonzero(robot.data.joint_vel[ids]) == 0

    check_reset(torch.arange(raw.num_envs, device=raw.device))
    assert raw.command_manager.get_term("base_velocity").cfg.debug_vis
    sensor = raw.scene["contact_forces"]
    left_ids, _ = sensor.find_bodies(["FL_hip", "RL_hip"])
    right_ids, _ = sensor.find_bodies(["FR_hip", "RR_hip"])
    first_left = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
    first_right = torch.zeros_like(first_left)
    for _ in range(5):
        env.step(torch.zeros(raw.num_envs, 4, device=raw.device))
        left_contact = sensor.data.net_forces_w[:, left_ids].norm(dim=-1).amax(dim=-1) > 1.0
        right_contact = sensor.data.net_forces_w[:, right_ids].norm(dim=-1).amax(dim=-1) > 1.0
        first_left |= left_contact
        first_right |= right_contact
    print(f"LEFT_SUPPORT_CHECK left_contact_envs={first_left.sum().item()} right_contact_envs={first_right.sum().item()}")
    assert first_left.all(), "Some robots did not land on their left support shells"
    assert not first_right.any(), "Right shells contacted the ground in the initial left-support phase"
    raw._reset_idx(torch.arange(raw.num_envs, device=raw.device))
    max_hip_motion = 0.0
    reset_count = 0
    for step in range(256):
        actions = torch.randn(32, 4, device=raw.device)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(rewards).all()
        for value in observations.values():
            assert torch.isfinite(value).all()
        max_hip_motion = max(
            max_hip_motion,
            (robot.data.joint_pos[:, hips] - robot.data.default_joint_pos[:, hips]).abs().max().item(),
        )
        reset_count += (terminated | truncated).sum().item()
        if step == 128:
            reset_ids = torch.tensor([0, 3, 17], device=raw.device)
            raw._reset_idx(reset_ids)
            check_reset(reset_ids)
    print(f"LOCK_CHECK joints={robot.joint_names} hip_motion_rad={max_hip_motion:.6f} resets={reset_count}")
    assert max_hip_motion > 0.05, "Hip actions did not move the robot"
    assert reset_count > 0
    print("HIP_ONLY_SMOKE_OK", flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    env.close()
    app.close()
