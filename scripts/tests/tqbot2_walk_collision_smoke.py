"""Verify selective walking self-collision in USD clones and GPU physics."""

import argparse
import json
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from pxr import PhysxSchema, Usd, UsdPhysics

import robot_lab.tasks  # noqa: F401
from robot_lab.assets.tqbot import TQBOT2_CFG
from robot_lab.assets.tqbot_collision import WALK_SELF_COLLISION_PAIRS_WITH_FEET, spawn_tqbot2_walk_with_leg_collisions
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.flat_env_cfg import TQBot2FlatEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.rough_env_cfg import TQBot2RoughEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import (
    TQBot2StandardWalkRoughEnvCfg,
)


def main():
    for cls in (TQBot2RoughEnvCfg, TQBot2FlatEnvCfg, TQBot2StandardWalkRoughEnvCfg):
        cfg = cls()
        assert cfg.scene.robot.spawn.articulation_props.enabled_self_collisions
        assert cfg.scene.robot.spawn.func is spawn_tqbot2_walk_with_leg_collisions
    assert not TQBOT2_CFG.spawn.articulation_props.enabled_self_collisions

    cfg = TQBot2StandardWalkRoughEnvCfg()
    cfg.scene.num_envs = 16
    cfg.seed = 42
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 2
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.commands.base_velocity.debug_vis = False
    env = gym.make("RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0", cfg=cfg).unwrapped
    report = {"num_envs": 16, "allowed_pairs": sorted(sorted(pair) for pair in WALK_SELF_COLLISION_PAIRS_WITH_FEET)}
    try:
        env.reset()
        robot = env.scene["robot"]
        source = ET.parse(cfg.scene.robot.spawn.asset_path).getroot()
        source_mass = sum(float(mass.get("value")) for mass in source.findall("link/inertial/mass"))
        actual_mass = robot.data.default_mass.sum(dim=-1)
        assert torch.allclose(actual_mass, torch.full_like(actual_mass, source_mass), atol=1e-4, rtol=0)
        assert torch.isfinite(robot.data.default_inertia).all()
        assert (robot.data.default_inertia[..., [0, 4, 8]] > 0).all()
        report["urdf_mass_kg"] = source_mass
        report["physx_default_mass_kg"] = actual_mass.cpu().tolist()
        report["physx_default_inertia_first_env"] = robot.data.default_inertia[0].cpu().tolist()
        report["physx_body_names"] = robot.body_names
        print(f"PASS: all PhysX articulations preserve URDF mass {source_mass:.6f} kg; finite positive inertias", flush=True)
        stage = robot.stage
        for index in range(env.num_envs):
            root = stage.GetPrimAtPath(f"/World/envs/env_{index}/Robot")
            bodies = {prim.GetName(): prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
            assert len(bodies) == 17, sorted(bodies)
            articulation = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(PhysxSchema.PhysxArticulationAPI)]
            assert len(articulation) == 1
            assert PhysxSchema.PhysxArticulationAPI(articulation[0]).GetEnabledSelfCollisionsAttr().Get()
            trunk_colliders = {
                prim.GetPath() for prim in Usd.PrimRange(bodies["trunk"], Usd.TraverseInstanceProxies())
                if prim.HasAPI(UsdPhysics.CollisionAPI)
            }
            assert trunk_colliders
            for a, b in combinations(bodies, 2):
                a_targets = UsdPhysics.FilteredPairsAPI(bodies[a]).GetFilteredPairsRel().GetTargets()
                b_targets = UsdPhysics.FilteredPairsAPI(bodies[b]).GetFilteredPairsRel().GetTargets()
                filtered = bodies[b].GetPath() in a_targets or bodies[a].GetPath() in b_targets
                if "trunk" in (a, b):
                    leg_targets = b_targets if a == "trunk" else a_targets
                    filtered = trunk_colliders.issubset(leg_targets)
                assert filtered == (frozenset((a, b)) not in WALK_SELF_COLLISION_PAIRS_WITH_FEET), (index, a, b)
                assert all(path.HasPrefix(root.GetPath()) for path in a_targets + b_targets)
                assert bodies["trunk"].GetPath() not in a_targets + b_targets
        report["clone_pair_filter_checks"] = env.num_envs * 136
        print("PASS: all 16 clones enable exactly 32 opposite-leg pairs and exclude 104 other pairs", flush=True)

        for _ in range(200):
            obs, reward, _, _, _ = env.step(torch.rand(env.num_envs, 12, device=env.device) * 2 - 1)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        report["random_action_steps"] = 200
        print("PASS: 200 random-action steps with finite observations and rewards", flush=True)

        # Place legs across each other high above the terrain. Any contact force
        # in the first physics step must come from robot self-collision.
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        ids = torch.arange(env.num_envs, device=env.device)
        peaks = {"front": 0.0, "rear": 0.0}
        for hip_angle in (0.6, 0.9, 1.2, 1.52):
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] = env.scene.env_origins
            root_state[:, 2] += 5.0
            root_state[:, 3:7] = torch.tensor([1., 0., 0., 0.], device=env.device)
            root_state[:, 7:] = 0
            q = robot.data.default_joint_pos.clone()
            for j, name in enumerate(robot.joint_names):
                if "hip" in name:
                    q[:, j] = -hip_angle if name[:2] in ("FL", "RL") else hip_angle
                elif "thigh" in name:
                    q[:, j] = 0.0
                elif "calf" in name:
                    q[:, j] = -1.5
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=ids)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
            robot.set_joint_position_target(q)
            env.scene.write_data_to_sim()
            for _ in range(5):
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                sensor.update(env.physics_dt, force_recompute=True)
                forces = sensor.data.net_forces_w.norm(dim=-1)
                for group, prefixes in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
                    indices = [j for j, name in enumerate(sensor.body_names) if name[:2] in prefixes]
                    peaks[group] = max(peaks[group], forces[:, indices].max().item())
        report["airborne_self_contact_peak_n"] = peaks
        assert peaks["front"] > 1.0 and peaks["rear"] > 1.0, peaks
        print(f"PASS: physical self-contact on both opposite-leg groups: {peaks}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


main()
app.close()
