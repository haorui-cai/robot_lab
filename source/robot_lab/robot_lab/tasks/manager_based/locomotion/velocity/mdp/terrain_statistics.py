"""Read-only terrain breakdown around Isaac Lab's unchanged distance curriculum."""

import numpy as np
import torch


def terrain_column_groups(generator):
    """Match TerrainGenerator's deterministic column selection exactly."""
    names = list(generator.sub_terrains)
    weights = np.array([cfg.proportion for cfg in generator.sub_terrains.values()])
    cumulative = np.cumsum(weights / weights.sum())
    groups = np.searchsorted(cumulative, np.arange(generator.num_cols) / generator.num_cols + 0.001,
                             side="right")
    return names, groups


def terrain_levels_with_statistics(env, env_ids):
    """Update once with the upstream curriculum and retain per-reset decisions."""
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import terrain_levels_vel

    if env_ids is None:
        env_ids = slice(None)
    terrain = env.scene.terrain
    ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None:
        ids = ids[env_ids]
    distance = torch.linalg.vector_norm(
        env.scene["robot"].data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1
    )
    command = env.command_manager.get_command("base_velocity")
    up = distance > terrain.cfg.terrain_generator.size[0] / 2
    down = (distance < torch.linalg.vector_norm(command[ids, :2], dim=1)
            * env.max_episode_length_s * 0.5) & ~up
    completed = up & (terrain.terrain_levels[ids] == terrain.max_terrain_level - 1)
    value = terrain_levels_vel(env, env_ids)
    env._terrain_curriculum_decisions = (ids, up, down, completed)
    return value


def summarize_terrain_levels(levels, types, generator, decisions=None):
    """All-population distribution; transition rates cover only this reset batch.

    ``move_down_rate`` includes requests at level zero, exposing failure even
    when clamping means the numerical level cannot decrease any further.
    """
    names, groups = terrain_column_groups(generator)
    column_groups = torch.as_tensor(groups, device=levels.device)
    population_groups = column_groups[types]
    result = {}
    for group, name in enumerate(names):
        selected = levels[population_groups == group]
        if not selected.numel():
            continue
        result[f"{name}/mean"] = selected.float().mean()
        result[f"{name}/max"] = selected.max()
        result[f"{name}/env_count"] = selected.numel()
        fractions = torch.bincount(selected, minlength=generator.num_rows).float() / selected.numel()
        for level in range(generator.num_rows):
            result[f"{name}/level_{level}_fraction"] = fractions[level]
        if decisions is not None:
            ids, up, down, completed = decisions
            batch = population_groups[ids] == group
            if batch.any():
                for key, values in (("move_up_rate", up), ("move_down_rate", down),
                                    ("top_completed_rate", completed)):
                    result[f"{name}/{key}"] = values[batch].float().mean()
    return result


def terrain_level_statistics(env, env_ids):
    """Additional logging term; never modifies terrain levels or origins."""
    terrain = env.scene.terrain
    return summarize_terrain_levels(
        terrain.terrain_levels, terrain.terrain_types, terrain.cfg.terrain_generator,
        getattr(env, "_terrain_curriculum_decisions", None),
    )
