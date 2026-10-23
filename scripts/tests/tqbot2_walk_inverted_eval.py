"""Frozen walking-policy recovery evaluation from a collision-safe inverted start."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--seed", type=int, default=680042)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
import torch
import trimesh
from rsl_rl.models import MLPModel
from scipy.spatial.transform import Rotation

from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_from_euler_xyz

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import (
    TQBot2StandardWalkRoughEnvCfg,
)


def inverted_spawn_height(path):
    """Support of all collision geometry along +Z at the default joint angles."""
    root = ET.parse(path).getroot()
    package = path.parent.parent
    poses = {"trunk": np.eye(4)}
    remaining = list(root.findall("joint"))
    while remaining:
        resolved = 0
        for joint in list(remaining):
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue
            origin = joint.find("origin")
            pose = np.eye(4)
            if origin is not None:
                pose[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                pose[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            if joint.get("type") == "revolute":
                name = joint.get("name")
                angle = 0.0 if "hip" in name else (0.8 if "thigh" in name else -1.5)
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(axis * angle).as_matrix()
            poses[joint.find("child").get("link")] = poses[parent] @ pose
            remaining.remove(joint)
            resolved += 1
        if not resolved:
            raise RuntimeError("Disconnected collision model")
    highest = -float("inf")
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            local = np.eye(4)
            if origin is not None:
                local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            pose = poses[link.get("name")] @ local
            geom = collision.find("geometry")[0]
            axes = pose[2, :3]
            center = pose[2, 3]
            if geom.tag == "mesh":
                mesh = trimesh.load(package / geom.get("filename").removeprefix("package://TQBot2_description/"), force="mesh")
                vertices = mesh.vertices * np.fromstring(geom.get("scale", "1 1 1"), sep=" ")
                bound = (vertices @ axes + center).max()
            elif geom.tag == "sphere":
                bound = center + float(geom.get("radius"))
            elif geom.tag == "cylinder":
                bound = center + abs(axes[2]) * float(geom.get("length")) / 2
                bound += float(geom.get("radius")) * math.sqrt(max(0.0, 1 - axes[2] ** 2))
            elif geom.tag == "box":
                bound = center + np.sum(abs(axes) * np.fromstring(geom.get("size"), sep=" ") / 2)
            else:
                raise RuntimeError(geom.tag)
            highest = max(highest, bound)
    return float(highest + 0.003)


def main():
    torch.set_num_threads(2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = TQBot2StandardWalkRoughEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 4.0
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.height_scanner = None
    cfg.scene.height_scanner_base = None
    cfg.observations.critic.height_scan = None
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.terrain_levels = None
    cfg.terminations.time_out = None
    cfg.terminations.terrain_out_of_bounds = None
    cfg.events.randomize_push_robot = None
    cfg.events.randomize_apply_external_force_torque = None
    cfg.events.randomize_reset_base.params["pose_range"] = {}
    cfg.events.randomize_reset_base.params["velocity_range"] = {}
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    cfg.log_dir = None
    # Small evaluation batches need smaller allocation capacities than training.
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    height = inverted_spawn_height(Path(cfg.scene.robot.spawn.asset_path))
    cfg.scene.robot.init_state.pos = (0.0, 0.0, height)
    env = gym.make("RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0", cfg=cfg).unwrapped
    try:
        dump_yaml(str(output / "evaluation_env.yaml"), cfg)
        obs, _ = env.reset(seed=args.seed)
        checkpoint = args.checkpoint.resolve()
        saved = torch.load(checkpoint, map_location=env.device, weights_only=False)
        actor = MLPModel(
            obs, {"actor": ["policy"]}, "actor", 12,
            hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ).to(env.device)
        actor.load_state_dict(saved["actor_state_dict"], strict=True)
        actor.eval()
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        foot_ids = [i for i, name in enumerate(sensor.body_names) if name.endswith("_foot")]
        assert len(foot_ids) == 4
        ids = torch.arange(env.num_envs, device=env.device)
        profiles = [("mean_zero_command", False, 0.0), ("mean_forward_0p5", False, 0.5),
                    ("gaussian_zero_command", True, 0.0), ("gaussian_forward_0p5", True, 0.5)]
        results = []
        metadata = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "seed": args.seed, "num_envs_per_profile": env.num_envs, "seconds": args.seconds,
            "device": env.device, "inverted_spawn_height_m": height,
            "criteria": {"max_tilt_deg": 30, "min_trunk_height_m": 0.20, "upright_hold_seconds": 0.5,
                         "minimum_foot_contacts_at_success": 2, "contact_force_threshold_n": 1.0},
            "physical_randomization": "training startup material/mass/COM and reset gains retained; no external pushes/forces",
            "observation_noise": False, "initial_joints": "default stance", "initial_velocity": "zero",
            "yaw": "uniform [-pi,pi], paired across profiles", "terrain": "flat",
            "action_clip": "same joint-action clip [-100,100]; no extra inference clip",
        }
        for name, stochastic, forward in profiles:
            obs, _ = env.reset(seed=args.seed)
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += env.scene.env_origins
            roll = torch.full((env.num_envs,), math.pi, device=env.device)
            yaw = (torch.rand(env.num_envs, device=env.device) * 2 - 1) * math.pi
            root_state[:, 3:7] = quat_from_euler_xyz(roll, torch.zeros_like(roll), yaw)
            root_state[:, 7:] = 0
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=ids)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.joint_vel), env_ids=ids)
            robot.set_joint_position_target(robot.data.default_joint_pos)
            command = env.command_manager.get_term("base_velocity")
            command.vel_command_b[:] = torch.tensor([forward, 0.0, 0.0], device=env.device)
            command.is_standing_env[:] = False
            command.is_heading_env[:] = False
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)
            assert (robot.data.projected_gravity_b[:, 2] > 0.999).all()
            assert torch.allclose(robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2],
                                  torch.full((env.num_envs,), height, device=env.device), atol=1e-4)
            obs = env.observation_manager.compute()
            dwell = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)
            first = torch.full((env.num_envs,), float("inf"), device=env.device)
            upright_at_end = None
            steps = round(args.seconds / env.step_dt)
            start = time.monotonic()
            with torch.no_grad():
                for step in range(steps):
                    actions = actor(obs, stochastic_output=stochastic)
                    assert torch.isfinite(actions).all()
                    obs, _, terminated, truncated, _ = env.step(actions)
                    assert not terminated.any() and not truncated.any(), "Automatic reset invalidates recovery trial"
                    good = ((robot.data.projected_gravity_b[:, 2] <= -math.cos(math.radians(30))) &
                            (robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] >= 0.20))
                    dwell = torch.where(good, dwell + 1, 0)
                    contacts = (sensor.data.net_forces_w[:, foot_ids].norm(dim=-1) > 1.0).sum(dim=-1)
                    recovered = (dwell >= math.ceil(0.5 / env.step_dt)) & (contacts >= 2)
                    newly = recovered & torch.isinf(first)
                    first[newly] = (step + 1) * env.step_dt
                    upright_at_end = good
                    if (step + 1) % 250 == 0:
                        print("PROGRESS", name, step + 1, "recovered", int(torch.isfinite(first).sum()),
                              "wall_seconds", round(time.monotonic() - start, 1), flush=True)
            times = first.cpu().tolist()
            successes = sum(math.isfinite(value) for value in times)
            entry = {
                "profile": name, "sampling": "gaussian" if stochastic else "mean", "forward_command_m_s": forward,
                "successes": successes, "attempts": env.num_envs, "success_rate": successes / env.num_envs,
                "successes_within_10s": sum(value <= 10.0 for value in times),
                "first_success_times_s": [value if math.isfinite(value) else None for value in times],
                "median_success_time_s": float(np.median([value for value in times if math.isfinite(value)])) if successes else None,
                "upright_at_end": int(upright_at_end.sum()), "wall_seconds": time.monotonic() - start,
            }
            results.append(entry)
            (output / "results.json").write_text(json.dumps({**metadata, "profiles": results}, indent=2) + "\n")
            print("RESULT", json.dumps({k: v for k, v in entry.items() if k != "first_success_times_s"}), flush=True)
        print("COMPLETE", str(output / "results.json"), flush=True)
    finally:
        env.close()


main()
app.close()
