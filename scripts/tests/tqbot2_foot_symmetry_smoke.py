"""Check diagonal displacement against actual link poses and reward managers."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import robot_lab.tasks  # noqa: F401
from robot_lab.assets.tqbot_walk_stance import WALK_POLICY_THIGH_LIMITS, WALK_THIGH_SOFT_LIMITS
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import (
    TQBot2StandardWalkRoughEnvCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.rewards import FootDisplacementSymmetry, joint_mirror
from test_tqbot2_walk_stance import forward_kinematics


def main():
    cfg = TQBot2StandardWalkRoughEnvCfg()
    assert cfg.rewards.joint_mirror.func is FootDisplacementSymmetry
    assert cfg.rewards.joint_mirror.weight == -.05
    assert UnitreeGo2RoughEnvCfg().rewards.joint_mirror.func is joint_mirror
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
        thigh_ids = [i for i, name in enumerate(robot.joint_names) if name.endswith("_thigh_joint")]
        hard_limits = robot.root_physx_view.get_dof_limits()[:, thigh_ids]
        soft_limits = robot.data.soft_joint_pos_limits[:, thigh_ids]
        torch.testing.assert_close(hard_limits, torch.tensor(WALK_POLICY_THIGH_LIMITS, device=hard_limits.device).expand_as(hard_limits))
        torch.testing.assert_close(soft_limits, torch.tensor(WALK_THIGH_SOFT_LIMITS, device=soft_limits.device).expand_as(soft_limits))
        term_cfg = env.reward_manager.get_term_cfg("joint_mirror")
        term = term_cfg.func
        assert isinstance(term, FootDisplacementSymmetry)
        reference = term.reference_positions_body.clone()
        calibration = term.calibration.copy()

        def value():
            return term(env, **term_cfg.params)

        torch.testing.assert_close(value(), torch.zeros(4, device=env.device), atol=2e-7, rtol=0.)
        standing = robot.data.default_joint_pos.clone()
        q = standing.clone()
        q[:2, robot.joint_names.index("FR_thigh_joint")] += .04
        q[2:, robot.joint_names.index("FL_calf_joint")] -= .03
        root = robot.data.default_root_state[:, :7].clone()
        root[:, :3] += env.scene.env_origins
        root[:, 2] += 1.
        root[1, :3] += torch.tensor([.3, -.2, .1], device=env.device)
        rotations = Rotation.from_euler("xyz", [[0, 0, 0], [.3, -.2, .7], [0, 0, 1.], [np.pi, 0, 0]])
        quats = rotations.as_quat()[:, [3, 0, 1, 2]]
        root[:, 3:7] = torch.tensor(quats, dtype=q.dtype, device=env.device)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.write_root_pose_to_sim(root)
        env.sim.forward()
        env.scene.update(env.physics_dt)

        # Independent FK oracle: randomized root pose and COM must not move
        # the reference or alter the body-frame displacement cost.
        expected = []
        for index in range(4):
            angles = dict(zip(robot.joint_names, q[index].cpu().tolist()))
            _, poses = forward_kinematics(cfg.scene.robot.spawn.asset_path, "trunk", angles)
            feet = np.array([poses[n][:3, 3] for n in calibration["foot_names"]])
            delta = (feet - reference.cpu().numpy())[:, [0, 2]]
            cost = np.square(delta[::2] - delta[1::2]).sum(axis=-1).mean() * term.scale
            upright = np.clip(rotations.as_matrix()[index, 2, 2], 0., .7) / .7
            expected.append(cost * upright)
        torch.testing.assert_close(value(), torch.tensor(expected, device=env.device, dtype=q.dtype), atol=2e-6, rtol=2e-3)
        torch.testing.assert_close(value()[0], value()[1], atol=2e-6, rtol=2e-3)
        assert value()[3] == 0

        action = env.action_manager.get_term("joint_pos")
        offset = action._offset.clone()
        before = value().clone()
        action._offset += .2
        torch.testing.assert_close(value(), before)
        action._offset.copy_(offset)
        env._reset_idx(torch.tensor([0, 2], device=env.device))
        torch.testing.assert_close(reference, term.reference_positions_body)
        assert calibration == term.calibration
        env.reset()
        for step in range(40):
            actions = torch.full((4, 12), .05 * np.sin(step * .2), device=env.device)
            obs, reward, _, _, _ = env.step(actions)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            torch.testing.assert_close(env.reward_manager._step_reward[:, env.reward_manager.active_terms.index("joint_mirror")],
                                       value() * term_cfg.weight, atol=1e-6, rtol=1e-4)
        report = {"checks": "passed", "num_envs": 4, "control_steps": 40, "device": env.device,
                  "thigh_hard_limits": hard_limits[0].cpu().tolist(), "thigh_soft_limits": soft_limits[0].cpu().tolist(),
                  "calibration": calibration, "weight": term_cfg.weight, "step_dt": env.step_dt,
                  "root_frame_and_upright_gate": True, "partial_reset_and_action_offset_independent": True,
                  "go2_original_term_preserved": True, "standing_zero_cost": True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS: standing reference, body-frame invariance, gate, reset, scale and reward integration", flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
