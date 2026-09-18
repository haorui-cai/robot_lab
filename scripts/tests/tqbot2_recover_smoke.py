# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""GPU integration checks for Recover. Run with the Isaac Lab Python environment.

python scripts/tests/tqbot2_recover_smoke.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math
from types import SimpleNamespace

import gymnasium as gym
import robot_lab.tasks  # noqa: F401
import torch
from robot_lab.tasks.manager_based.locomotion.recover import mdp
from robot_lab.tasks.manager_based.locomotion.recover.config.tqbot2.recover_env_cfg import (
    JOINT_NAMES,
    TQBot2RecoverEnvCfg,
)


def check_success_logic():
    """Reject bad body states but accept arbitrary joint positions and velocities."""
    count = 7
    data = SimpleNamespace(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.07]]).repeat(count, 1),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        joint_pos=torch.zeros(count, 12),
        default_joint_pos=torch.zeros(count, 12),
        joint_vel=torch.zeros(count, 12),
        root_lin_vel_b=torch.zeros(count, 3),
        root_ang_vel_b=torch.zeros(count, 3),
    )
    forces = torch.zeros(count, 13, 3)
    forces[:, 0, 2] = 50.0
    data.projected_gravity_b[1, 2] = 1.0
    data.root_pos_w[2, 2] = 0.35
    forces[3] = 0.0
    data.root_ang_vel_b[4, 0] = 1.0
    data.joint_pos[5] = torch.linspace(-2.0, 4.0, 12)
    data.joint_vel[6] = 5.0
    data.root_pos_w[5, 2] = 0.15

    class Scene(dict):
        env_origins = torch.zeros(count, 3)

    env = SimpleNamespace(
        scene=Scene(
            robot=SimpleNamespace(data=data), contact_forces=SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
        ),
        num_envs=count,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.full((count,), 100),
    )
    expected = [True, False, False, False, False, True, True]
    assert mdp.is_prone(env).tolist() == expected
    assert mdp.upright(env)[0] == 1 and mdp.upright(env)[1] == 0
    assert mdp.prone_settled(env)[0] > mdp.prone_settled(env)[1:3].max()
    assert mdp.prone_settled(env)[0] == mdp.prone_settled(env)[5] == mdp.prone_settled(env)[6]
    term = mdp.StableProne(SimpleNamespace(), env)
    for _ in range(math.ceil(0.5 / env.step_dt) - 1):
        assert not term(env).any()
    assert term(env).tolist() == expected
    term.reset(torch.tensor([0]))
    assert not term(env)[0]
    term.reset()
    forces[0] = 0
    assert not term(env).any()
    assert term._stable_steps[0] == 0
    forces[0, 0, 2] = 50
    env.episode_length_buf[:] = 0
    for _ in range(30):
        assert not term(env).any()
    # Height must be relative to each environment's origin.
    env.scene.env_origins[:, 2] = 2.0
    data.root_pos_w[:, 2] += 2.0
    assert mdp.is_prone(env)[0]
    print("PASS: joint-independent success/reward, signed gravity, dwell/reset and origin handling")


def check_simulation():
    cfg = TQBot2RecoverEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 16
    cfg.terminations.success = None
    cfg.rewards.success_bonus = None
    cfg.episode_length_s = 30.0
    cfg.observations.policy.enable_corruption = False
    env = gym.make("RobotLab-Isaac-Recover-Flat-TQBot2-v0", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (16, 42)
        assert obs["critic"].shape == (16, 46)
        robot = env.scene["robot"]
        assert "trunk" in robot.body_names, robot.body_names
        _, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
        assert names == JOINT_NAMES
        assert (robot.data.projected_gravity_b[:, 2] > 0.99).all(), "Every reset must be feet-up"
        assert torch.allclose(mdp.base_height(env), torch.full((16, 1), 0.096, device=env.device), atol=1e-4)
        assert not mdp.is_prone(env).any(), "No initial state may already satisfy recovery"
        # Zero policy output still applies the neutral PD target. It must not
        # solve this task merely by settling from a high spawn point.
        ever_prone = torch.zeros(16, dtype=torch.bool, device=env.device)
        for _ in range(200):
            env.step(torch.zeros(16, 12, device=env.device))
            ever_prone |= mdp.is_prone(env)
        assert not ever_prone.any(), "Zero-action baseline unexpectedly recovers from the initial state"
        print("PASS: 16/16 physically inverted starts; zero-action baseline 0/16 recoveries over 4 seconds")
        env.reset()
        for _ in range(200):
            obs, reward, _, _, _ = env.step(torch.rand(16, 12, device=env.device) * 2 - 1)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        print("PASS: URDF import, ordered joints, 42/46 observations, 200 random-action steps")

        # Check physical feasibility of the exact FixedLie target with zero action.
        ids = torch.arange(env.num_envs, device=env.device)
        root = robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins
        root[:, 2] += 0.3
        root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        root[:, 7:] = 0
        robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)
        robot.write_root_velocity_to_sim(root[:, 7:], env_ids=ids)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos, torch.zeros_like(robot.data.joint_vel), env_ids=ids
        )
        robot.set_joint_position_target(robot.data.default_joint_pos)
        for _ in range(200):
            env.step(torch.zeros(16, 12, device=env.device))
        good = mdp.is_prone(env)
        print("FixedLie settled:", good.tolist())
        print("Height:", mdp.base_height(env).flatten().tolist())
        print("Maximum joint errors:", (robot.data.joint_pos - robot.data.default_joint_pos).abs().amax(-1).tolist())
        print("Joint names:", robot.joint_names)
        print("First robot joint errors:", (robot.data.joint_pos - robot.data.default_joint_pos)[0].tolist())
        print("Angular speeds:", robot.data.root_ang_vel_b.norm(dim=-1).tolist())
        assert good.float().mean() >= 0.9, "FixedLie target must be physically achievable"
        print("PASS: zero-action FixedLie target settles inside success thresholds")
        # Auto-resets use subsets of environments: verify the same inverted
        # contract there and ensure other environments retain their orientation.
        subset = torch.tensor([0, 3, 7], device=env.device)
        before = robot.data.root_quat_w.clone()
        env._reset_idx(subset)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.physics_dt)
        assert (robot.data.projected_gravity_b[subset, 2] > 0.99).all()
        untouched = torch.ones(16, dtype=torch.bool, device=env.device)
        untouched[subset] = False
        assert torch.allclose(robot.data.root_quat_w[untouched], before[untouched], atol=1e-4)
        print("PASS: partial/automatic reset writes inverted poses only to selected robots")
    finally:
        env.close()


# As in the standard Isaac entry points, only close Kit after checks pass.
# Kit shutdown in a finally block can mask an assertion's non-zero exit code.
check_success_logic()
check_simulation()
app.close()
