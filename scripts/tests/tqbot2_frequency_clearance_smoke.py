"""Exercise frequency/terrain-clearance rewards with a frozen walking policy."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import hashlib
import json

import gymnasium as gym
import torch
from rsl_rl.models import MLPModel

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import (
    TQBot2StandardWalkRoughEnvCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.foot_ground import (
    SwingFootClearance, terrain_surface_at_feet,
)
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.normalized_gait import FootCadenceReward
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.frequency_excess import FootFrequencyExcess


def main():
    torch.set_num_threads(2)
    cfg = TQBot2StandardWalkRoughEnvCfg()
    assert cfg.rewards.feet_air_time is None and cfg.rewards.feet_height_body is None
    assert cfg.rewards.feet_frequency.params["target_frequency"] == 2.0
    assert cfg.rewards.feet_frequency.weight == .2
    assert cfg.rewards.feet_frequency_excess.weight == -.5
    assert cfg.rewards.feet_frequency_excess.params["threshold_frequency"] == 2.
    assert cfg.scene.foot_cycle_contacts.history_length == cfg.decimation == 4
    assert cfg.scene.foot_cycle_contacts.update_period == cfg.sim.dt == .005
    assert cfg.rewards.feet_ground_clearance.params["target_clearance"] == .05
    assert cfg.rewards.feet_ground_clearance.weight == -.5
    assert cfg.rewards.feet_ground_clearance.params["error_scale"] == .05
    assert cfg.rewards.joint_pos_penalty.weight == -.2
    assert cfg.rewards.joint_pos_penalty.params["swing_scale"] == 0.
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.scene.num_envs = 6
    generator = cfg.scene.terrain.terrain_generator
    generator.seed = 42
    generator.num_rows, generator.num_cols = 2, 6
    terrain_names = list(generator.sub_terrains)
    for terrain in generator.sub_terrains.values():
        terrain.proportion = 1. / len(terrain_names)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = cfg.curriculum.terrain_breakdown = None
    cfg.terminations.time_out = cfg.terminations.terrain_out_of_bounds = None
    cfg.observations.policy.enable_corruption = False
    cfg.events.randomize_push_robot = cfg.events.randomize_apply_external_force_torque = None
    cfg.events.randomize_reset_base.params = {"pose_range": {}, "velocity_range": {}}
    cfg.events.randomize_reset_joints.params["position_range"] = (1., 1.)
    cfg.events.randomize_reset_joints.params["velocity_range"] = (0., 0.)
    command_cfg = cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = command_cfg.rel_heading_envs = 0.
    command_cfg.resampling_time_range = (1000., 1000.)
    command_cfg.debug_vis = False
    command_cfg.ranges.lin_vel_x = (.5, .5)
    command_cfg.ranges.lin_vel_y = command_cfg.ranges.ang_vel_z = (0., 0.)
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    cfg.log_dir = None
    env = gym.make("RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset(seed=42)
        assert obs["policy"].shape == (6, 45) and obs["critic"].shape == (6, 235)
        reward_cfgs = {name: env.reward_manager.get_term_cfg(name)
                       for name in ("feet_frequency", "feet_frequency_excess", "feet_ground_clearance")}
        assert isinstance(reward_cfgs["feet_frequency"].func, FootCadenceReward)
        assert isinstance(reward_cfgs["feet_frequency_excess"].func, FootFrequencyExcess)
        assert isinstance(reward_cfgs["feet_ground_clearance"].func, SwingFootClearance)
        assert reward_cfgs["feet_ground_clearance"].weight == -0.5
        assert reward_cfgs["feet_ground_clearance"].params["error_scale"] == 0.05
        posture_cfg = env.reward_manager.get_term_cfg("joint_pos_penalty")
        assert posture_cfg.weight == -0.2 and posture_cfg.params["swing_scale"] == 0.0
        actor = MLPModel(obs, {"actor": ["policy"]}, "actor", 12, hidden_dims=[512, 256, 128],
                         activation="elu", obs_normalization=False,
                         distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"}).to(env.device)
        actor.load_state_dict(torch.load(args.checkpoint, map_location=env.device, weights_only=False)["actor_state_dict"], strict=True)
        actor.eval()
        raw = {name: [] for name in reward_cfgs}
        ray_valid = []
        cadence_valid_counts = []
        foot_ids = reward_cfgs["feet_ground_clearance"].params["asset_cfg"].body_ids
        with torch.no_grad():
            for step in range(250):
                obs, reward, terminated, truncated, _ = env.step(actor(obs, stochastic_output=False))
                assert not (terminated | truncated).any()
                assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
                for name, term_cfg in reward_cfgs.items():
                    value = term_cfg.func(env, **term_cfg.params)
                    assert torch.isfinite(value).all() and (value >= 0.).all()
                    index = env.reward_manager.active_terms.index(name)
                    torch.testing.assert_close(env.reward_manager._step_reward[:, index], value * term_cfg.weight)
                    raw[name].append(value.cpu().clone())
                _, _, valid = terrain_surface_at_feet(env, env.scene["robot"].data.body_link_pos_w[:, foot_ids])
                ray_valid.append(valid.cpu())
                cadence_valid_counts.append(reward_cfgs["feet_frequency"].func.valid_periods.sum(-1).cpu())
                if (step + 1) % 50 == 0:
                    print(f"PROGRESS {step + 1}/250", flush=True)
            raw = {name: torch.stack(values) for name, values in raw.items()}
            assert (raw["feet_frequency"] <= 1.).all()
            assert (raw["feet_frequency"] > 0.).any(), "No actual valid cycles observed"
            assert (raw["feet_ground_clearance"] > 0.).any(), "Clearance cost never exercised"
            assert (raw["feet_frequency_excess"] > 0.).any(), "Excess cadence cost never exercised"
            assert torch.stack(ray_valid).all()
            # Probe the actual static terrain mesh outside the spawn platforms.
            offsets = torch.linspace(-3., 3., 13, device=env.device)
            xx, yy = torch.meshgrid(offsets, offsets, indexing="ij")
            probes = torch.stack((xx.flatten(), yy.flatten(), torch.ones_like(xx).flatten()), -1)
            probes = env.scene.env_origins[:, None, :] + probes[None, :, :]
            _, normals, valid = terrain_surface_at_feet(env, probes)
            sloped = valid & (normals[..., 2] < .999) & (normals[..., 2] > .5)
            assert sloped.any(), "No non-horizontal terrain face exercised"
            term = reward_cfgs["feet_frequency"].func
            selected = torch.tensor([0, 2], device=env.device)
            unselected = torch.tensor([1, 3, 4, 5], device=env.device)
            prior_valid = term.valid_periods.clone()
            term.reset(selected)
            assert not term.valid_periods[selected].any()
            torch.testing.assert_close(term.valid_periods[unselected], prior_valid[unselected])
            excess = reward_cfgs["feet_frequency_excess"].func
            prior_measured = excess.measured.clone()
            excess.reset(selected)
            assert not excess.measured[selected].any()
            torch.testing.assert_close(excess.measured[unselected], prior_measured[unselected])
            command = env.command_manager.get_term("base_velocity")
            command.vel_command_b.zero_()
            command.is_standing_env[:] = True
            obs, _, _, _, _ = env.step(actor(obs, stochastic_output=False))
            for term_cfg in reward_cfgs.values():
                torch.testing.assert_close(term_cfg.func(env, **term_cfg.params), torch.zeros(6, device=env.device))
            assert not term.valid_periods.any()
            assert not excess.measured.any()
        report = {
            "passed": True, "device": env.device, "num_envs": 6, "control_steps": 251,
            "terrain_types": terrain_names, "actor_dimensions": 45, "critic_dimensions": 235,
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "target_frequency_hz": 2., "target_clearance_m": .05,
            "frequency_excess_weight": -.5, "frequency_excess_confirmation_seconds": .01,
            "clearance_error_scale_m": .05, "clearance_weight": -.5,
            "posture_weight": -.2, "swing_posture_scale": 0.,
            "cadence_valid_foot_count_histogram": torch.bincount(torch.stack(cadence_valid_counts).flatten(), minlength=5).tolist(),
            "active_rewards": env.reward_manager.active_terms,
            "all_foot_rays_valid": True, "non_horizontal_probe_count": int(sloped.sum()),
            "reward_manager_weighting_checked": True, "partial_reset_checked": True, "stop_gate_checked": True,
            "raw_reward_ranges": {name: [float(value.min()), float(value.max())] for name, value in raw.items()},
            "purpose": "Runtime integration only; this frozen policy has not trained with the new rewards.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("PASS: cadence and terrain clearance runtime integration", flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    # Kit shutdown can consume the Python exception; preserve failure in the log.
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
