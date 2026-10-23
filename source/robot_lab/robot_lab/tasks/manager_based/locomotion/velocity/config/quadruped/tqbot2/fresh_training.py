"""Portable fresh-start assertions and provenance; no checkpoint is loaded here."""
import hashlib
import json
from pathlib import Path
import subprocess

from .free_gait_policy import REMOVED, V4_ACTIVE, active_reward_names
from .free_gait_terrain import validate_free_gait_terrain


def fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit_fresh_start(runner, env_cfg, agent_cfg, args_cli, log_dir):
    import torch

    if args_cli.resume or args_cli.warm_start or agent_cfg.resume:
        raise ValueError("Fresh-control entry refuses resume/warm_start; use documented recovery workflow separately")
    if agent_cfg.load_run != "__NO_RESUME__" or agent_cfg.load_checkpoint != "__NO_CHECKPOINT__":
        raise ValueError("Fresh controls require disabled load selectors")
    if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
        raise ValueError("Expected iteration0 and emptyAdam")
    if runner.alg.actor.obs_dim != 45 or runner.alg.critic.obs_dim != 235:
        raise ValueError("Expected original45 actor /235 critic interface")
    if runner.alg.actor.obs_normalization or runner.alg.critic.obs_normalization:
        raise ValueError("V4 controls use fresh identity normalizers")
    variant = "free" if "FreeGaitScratch" in args_cli.task else "v4"
    expected = V4_ACTIVE - REMOVED.keys() if variant == "free" else V4_ACTIVE
    if active_reward_names(env_cfg.rewards) != expected:
        raise ValueError("Runtime reward set differs from reviewed variant")
    generator = env_cfg.scene.terrain.terrain_generator
    if variant == 'free':
        validate_free_gait_terrain(generator)
    # Locate the checkout by its package metadata rather than a machine-specific path.
    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    model = Path(env_cfg.scene.robot.spawn.asset_path)
    output = Path(log_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "passed": True, "variant": variant, "seed": agent_cfg.seed,
        "initialization": "random actor/critic, emptyAdam, identity normalizers",
        "resume": False, "load_run": agent_cfg.load_run, "load_checkpoint": agent_cfg.load_checkpoint,
        "optimizer_state_entries": 0, "first_iteration": 0, "code_commit": revision,
        "actor_fingerprint": fingerprint(runner.alg.actor.state_dict()),
        "critic_fingerprint": fingerprint(runner.alg.critic.state_dict()),
        "asset_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "active_rewards": sorted(expected), "num_envs": env_cfg.scene.num_envs,
        "terrain_proportions": {name: cfg.proportion for name, cfg in generator.sub_terrains.items()},
        "terrain_rows_columns": [generator.num_rows, generator.num_cols],
        "iterations": agent_cfg.max_iterations,
    }
    torch.save({**runner.alg.save(), "iter": -1, "infos": audit}, output / "initial_random_state.pt")
    (output / "fresh_initialization.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("[TQBot2 FRESH CONTROL VERIFIED] " + json.dumps(audit), flush=True)
