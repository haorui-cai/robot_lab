"""Reuse the parent's two rough-terrain slope generators for specialization."""
import math
import torch
from isaaclab.managers import CurriculumTermCfg, TerminationTermCfg
from isaaclab.terrains import MeshPlaneTerrainCfg
from isaaclab.terrains.height_field.hf_terrains import pyramid_sloped_terrain
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import TQBot2StandardWalkRoughEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity import mdp


def pyramid_with_target_row(difficulty, cfg):
    # Retain the original geometry; ensure the last row includes the requested ceiling.
    return pyramid_sloped_terrain(1. if difficulty >= .9 else difficulty, cfg)


def statistics(env):
    terrain = env.scene.terrain
    result = {}
    for name, mask in [('convex', terrain.terrain_types < 4),
                       ('concave', (terrain.terrain_types >= 4) & (terrain.terrain_types < 8)),
                       ('flat', terrain.terrain_types >= 8)]:
        levels = terrain.terrain_levels[mask].float()
        result[f'{name}_count'] = int(mask.sum())
        if len(levels):
            result[f'{name}_mean_level'] = float(levels.mean())
            result[f'{name}_target_row_fraction'] = float((levels == 9).float().mean())
    return result


def curriculum(env, env_ids):
    terrain = env.scene.terrain
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    # Exactly one inherited distance curriculum update per reset.
    finished = ids[env.episode_length_buf[ids] > 0]
    if len(finished):
        mdp.terrain_levels_vel(env, finished)
    # Keep one fifth of each slope group on the nominal 30-degree row.
    challenge = ids[(ids % 5 == 0) & (terrain.terrain_types[ids] < 8)]
    terrain.terrain_levels[challenge] = 9
    terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
    return statistics(env)


def fallen(env):
    return -env.scene['robot'].data.projected_gravity_b[:, 2] < math.cos(math.radians(70))


def configuration(num_envs, device, seed):
    cfg = TQBot2StandardWalkRoughEnvCfg()
    cfg.seed, cfg.sim.device, cfg.scene.num_envs = seed, device, num_envs
    generator = cfg.scene.terrain.terrain_generator
    generator.seed, generator.num_cols, generator.num_rows = seed, 10, 10
    generator.use_cache = False
    selected = {}
    for name in ('hf_pyramid_slope', 'hf_pyramid_slope_inv'):
        terrain = generator.sub_terrains[name].copy()
        terrain.proportion = .4
        terrain.function = pyramid_with_target_row
        selected[name] = terrain
    selected['flat_rehearsal'] = MeshPlaneTerrainCfg(proportion=.2)
    generator.sub_terrains = selected
    cfg.scene.terrain.max_init_terrain_level = 3
    cfg.events.high_friction_flat_material = None
    cfg.terminations.high_friction_flat_exit = None
    cfg.events.randomize_foot_material.params.update(
        static_friction_range=(.7, 1.), dynamic_friction_range=(.7, 1.),
        restitution_range=(0., 0.), make_consistent=True)
    # Start walking on the existing center platforms; retain arbitrary headings.
    cfg.events.randomize_reset_base.params = {
        'pose_range': {'x': (-.25, .25), 'y': (-.25, .25), 'roll': (-.03, .03),
                       'pitch': (-.03, .03), 'yaw': (-math.pi, math.pi)},
        'velocity_range': {},
    }
    cfg.events.randomize_apply_external_force_torque = None
    cfg.events.randomize_push_robot = None
    cfg.terminations.slope_fall = TerminationTermCfg(func=fallen)
    cfg.curriculum.terrain_levels = CurriculumTermCfg(func=curriculum)
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = command.rel_heading_envs = 0.
    command.resampling_time_range = (25., 25.)
    command.ranges.lin_vel_x = (.3, .6)
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (0., 0.)
    command.debug_vis = False
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**20
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**20
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**22
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
    cfg.log_dir = None
    return cfg


def course(env):
    return {'terrain_levels': env.scene.terrain.terrain_levels.cpu().tolist(), 'statistics': statistics(env)}


def restore_course(env, data):
    if 'terrain_levels' in data:
        terrain = env.scene.terrain
        levels = torch.tensor(data['terrain_levels'], device=env.device)
        if levels.shape == terrain.terrain_levels.shape:
            terrain.terrain_levels[:] = levels
            terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
