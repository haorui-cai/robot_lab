"""Check narrowed thigh targets, recovery histories and energy terms in actual PhysX."""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.models import MLPModel

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import (
    TQBot2StandardWalkRoughEnvCfg,
)


def main():
    cfg = TQBot2StandardWalkRoughEnvCfg()
    slopes = {n: cfg.scene.terrain.terrain_generator.sub_terrains[n].slope_range
              for n in ("hf_pyramid_slope", "hf_pyramid_slope_inv")}
    assert all(abs(r[1] - math.tan(math.radians(30))) < 1e-12 for r in slopes.values())
    cfg.scene.num_envs = 16
    cfg.seed = 42
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.height_scanner = None
    cfg.scene.height_scanner_base = None
    cfg.observations.critic.height_scan = None
    cfg.curriculum.terrain_levels = None
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.randomize_apply_external_force_torque = None
    cfg.events.randomize_reset_base.params = {"pose_range": {}, "velocity_range": {}}
    cfg.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.resampling_time_range = (1000., 1000.)
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    cfg.log_dir = None
    env = gym.make("RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset(seed=42)
        robot = env.scene["robot"]
        thigh_ids = [i for i, n in enumerate(robot.joint_names) if n.endswith("_thigh_joint")]
        assert len(thigh_ids) == 4
        limits = robot.data.joint_pos_limits[:, thigh_ids]
        torch.testing.assert_close(limits[..., 0], torch.full_like(limits[..., 0], .6))
        torch.testing.assert_close(limits[..., 1], torch.full_like(limits[..., 1], 3.2))
        assert obs["policy"].shape == (16, 45)
        actor = MLPModel(obs, {"actor": ["policy"]}, "actor", 12, hidden_dims=[512, 256, 128],
                         activation="elu", obs_normalization=False,
                         distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"}).to(env.device)
        actor.load_state_dict(torch.load(args.checkpoint, map_location=env.device, weights_only=False)["actor_state_dict"])
        actor.eval()
        action = env.action_manager.get_term("joint_pos")
        for sign, target in ((-1, .6), (1, 3.2)):
            action.process_actions(torch.full((16, 12), 100. * sign, device=env.device))
            joint_ids = [i for i, n in enumerate(action._joint_names) if n.endswith("_thigh_joint")]
            torch.testing.assert_close(action.processed_actions[:, joint_ids],
                                       torch.full_like(action.processed_actions[:, joint_ids], target))
        env.reset()
        state = robot.data.default_root_state.clone()
        state[:, :3] += env.scene.env_origins
        state[8:, 2] = env.scene.env_origins[8:, 2] + .12
        state[8:, 3:7] = torch.tensor([0., 1., 0., 0.], device=env.device)
        state[:, 7:] = 0.
        ids = torch.arange(16, device=env.device)
        robot.write_root_pose_to_sim(state[:, :7], env_ids=ids)
        robot.write_root_velocity_to_sim(state[:, 7:], env_ids=ids)
        command = env.command_manager.get_term("base_velocity")
        command.vel_command_b[:] = 0.
        command.is_heading_env[:] = False
        command.is_standing_env[:] = True
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.physics_dt)
        obs = env.observation_manager.compute()
        peaks = {"torque_nm": 0., "joint_speed_rad_s": 0.}
        with torch.no_grad():
            for _ in range(300):
                obs, reward, _, _, _ = env.step(actor(obs, stochastic_output=False))
                assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
                processed = action.processed_actions[:, joint_ids]
                assert (processed >= .6 - 1e-6).all() and (processed <= 3.2 + 1e-6).all()
                assert torch.isfinite(env.reward_manager._step_reward).all()
                peaks["torque_nm"] = max(peaks["torque_nm"], robot.data.applied_torque.abs().max().item())
                peaks["joint_speed_rad_s"] = max(peaks["joint_speed_rad_s"], robot.data.joint_vel.abs().max().item())
        history = env.reward_manager.get_term_cfg("recovery_torque_change").func.history
        subset = torch.tensor([1, 7], device=env.device)
        env._reset_idx(subset)
        assert not history.valid[subset].any()
        other = torch.tensor([0, 2, 3], device=env.device)
        assert history.valid[other].all()
        report = dict(checkpoint=str(args.checkpoint.resolve()), device=env.device, num_envs=16,
                      physics_thigh_limits=[.6, 3.2], target_clipping_verified=True,
                      reward_history_partial_reset_verified=True, steps=300, nominal_slope_ranges=slopes,
                      peaks=peaks, upright_fraction_at_end=float((-robot.data.projected_gravity_b[:, 2] > .866).float().mean()),
                      note="Integration smoke test; energy reduction and 30-degree climbing performance not established")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print("TQBot2 gentle walking integration checks passed", flush=True)
    finally:
        env.close()


main()
app.close()
