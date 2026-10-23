"""Frozen-policy contact attribution on the six training terrain types."""
import argparse
import inspect
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=120)
parser.add_argument("--seconds", type=float, default=40.)
parser.add_argument("--terrain-level", type=int, default=0)
parser.add_argument("--mean-policy", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.models import MLPModel
import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import TQBot2StandardWalkRoughEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.terrain_statistics import terrain_column_groups


def main():
    torch.set_num_threads(4)
    cfg = TQBot2StandardWalkRoughEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.terrain.max_init_terrain_level = args.terrain_level
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.terrain_breakdown = None
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    cfg.log_dir = None
    bodies = ["trunk"] + [f"{leg}_{part}" for leg in ("FR", "FL", "RR", "RL")
                           for part in ("hip", "thigh", "calf")]
    # GPU contact filters do not support the static terrain mesh in this
    # PhysX version. Attribute the original reward by body; do not label all
    # net contacts as ground contacts, since self-collision is enabled.
    env = gym.make("RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0", cfg=cfg).unwrapped
    try:
        terrain = env.scene.terrain
        terrain.terrain_levels[:] = args.terrain_level
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        env.scene.env_origins[:] = terrain.env_origins
        obs, _ = env.reset(seed=42)
        robot, sensor = env.scene["robot"], env.scene["contact_forces"]
        ids = [sensor.body_names.index(name) for name in bodies]
        actor = MLPModel(obs, {"actor": ["policy"]}, "actor", 12, hidden_dims=[512, 256, 128],
                         activation="elu", obs_normalization=False,
                         distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "scalar"}).to(env.device)
        actor.load_state_dict(torch.load(args.checkpoint, map_location=env.device, weights_only=False)["actor_state_dict"])
        actor.eval()
        names, columns = terrain_column_groups(terrain.cfg.terrain_generator)
        groups = torch.as_tensor(columns, device=env.device)[terrain.terrain_types]
        stats = {name: {window: {"samples": 0, "contacts": torch.zeros(len(bodies), device=env.device),
                                 "weighted_cost": torch.zeros(len(bodies), device=env.device)}
                        for window in ("all", "upright_after_5s")} for name in ["all_terrains", *names]}
        steps = round(args.seconds / env.step_dt)
        with torch.no_grad():
            for step in range(steps):
                action = actor(obs, stochastic_output=not args.mean_policy)
                obs, reward, _, _, _ = env.step(action)
                assert torch.isfinite(reward).all()
                contact = sensor.data.net_forces_w_history[:, :, ids].norm(dim=-1).amax(dim=1) > 1.
                gate = (-robot.data.projected_gravity_b[:, 2]).clamp(0., .7) / .7
                upright = (robot.data.projected_gravity_b[:, 2] < -.866) & (env.episode_length_buf * env.step_dt >= 5.)
                for group, name in enumerate(["all_terrains", *names]):
                    selected = torch.ones(env.num_envs, device=env.device, dtype=torch.bool) if group == 0 else groups == group - 1
                    for window, mask in (("all", selected), ("upright_after_5s", selected & upright)):
                        entry = stats[name][window]
                        entry["samples"] += int(mask.sum())
                        entry["contacts"] += contact[mask].float().sum(0)
                        entry["weighted_cost"] += (contact[mask].float() * gate[mask, None]).sum(0)
                if (step + 1) % 250 == 0:
                    print(f"CONTACT_DIAGNOSTIC {step + 1}/{steps}", flush=True)
        result = {"checkpoint": str(args.checkpoint.resolve()), "config_source": inspect.getfile(TQBot2StandardWalkRoughEnvCfg),
                  "num_envs": env.num_envs, "device": env.device, "simulated_seconds": args.seconds,
                  "terrain_level": args.terrain_level, "policy": "mean" if args.mean_policy else "sampled",
                  "body_names": bodies, "statistics": {},
                  "notes": "Frozen checkpoint replay; not the live training population. Random resets and domain randomization retained. Contact threshold 1 N; 3-physics-step force history. Net contacts include terrain and self-collision; static terrain GPU filtering is unsupported."}
        for name, windows in stats.items():
            result["statistics"][name] = {}
            for window, entry in windows.items():
                n = entry["samples"]
                result["statistics"][name][window] = {"samples": n, "bodies": {
                    body: {"any_contact_fraction": float(entry["contacts"][i] / max(1, n)),
                           "weighted_penalty_contribution": -float(entry["weighted_cost"][i] / max(1, n))}
                    for i, body in enumerate(bodies)}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("CONTACT_DIAGNOSTIC_COMPLETE", args.output, flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
