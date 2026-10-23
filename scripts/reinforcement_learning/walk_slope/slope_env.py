"""Independent planar-slope evaluation over the model_4200 training source."""
import math
import numpy as np
import torch
import trimesh

from isaaclab.managers import CurriculumTermCfg, EventTermCfg, TerminationTermCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_apply, quat_apply_inverse
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import TQBot2StandardWalkRoughEnvCfg
from geometry import ramp_vertices


def ramp_mesh(difficulty, cfg):
    level = min(6, int(difficulty * 7))
    vertices, faces = ramp_vertices(cfg.sign * level * 5., cfg.size)
    return [trimesh.Trimesh(vertices=vertices, faces=faces, process=False)], np.array([cfg.size[0]/2, cfg.size[1]/2, 0.])


@configclass
class RampCfg(SubTerrainBaseCfg):
    function = ramp_mesh
    sign: float = 1.


def state(env):
    if hasattr(env, '_walk_slope'):
        return env._walk_slope
    n, dev = env.num_envs, env.device
    terrain = env.scene.terrain
    signs = torch.where(terrain.terrain_types < 2, 0., torch.where(terrain.terrain_types < 6, 1., -1.))
    s = {'sign': signs, 'angle': torch.zeros(n, device=dev), 'completed': {}, 'fixed_cases': None,
         'friction': torch.full((n,), float('nan'), device=dev)}
    for name in ('steps', 'count', 'error2', 'forward', 'expected', 'lateral', 'aligned', 'heading', 'bad_contact'):
        s[name] = torch.zeros(n, device=dev)
    s['start'] = torch.zeros((n, 3), device=dev)
    s['last'] = torch.zeros((n, 3), device=dev)
    s['failed'] = torch.zeros(n, device=dev, dtype=torch.bool)
    env._walk_slope = s
    return s


def axes(env):
    a = torch.deg2rad(state(env)['angle'])
    normal = torch.stack((-a.sin(), torch.zeros_like(a), a.cos()), -1)
    tangent = torch.stack((a.cos(), torch.zeros_like(a), a.sin()), -1)
    return normal, tangent


def reset_on_ramp(env, env_ids):
    s = state(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    terrain, robot = env.scene.terrain, env.scene['robot']
    if s['fixed_cases'] is None:
        levels = terrain.terrain_levels[ids]
        angle = levels * 5 * s['sign'][ids]
    else:
        cases = s['fixed_cases']
        angle = cases[ids, 0]
        levels = (angle.abs()/5).long()
        terrain.terrain_types[ids] = torch.where(angle == 0, 0, torch.where(angle > 0, 2, 6))
    s['angle'][ids] = angle
    terrain.terrain_levels[ids] = levels
    terrain.env_origins[ids] = terrain.terrain_origins[levels, terrain.terrain_types[ids]]
    env.scene.env_origins[ids] = terrain.env_origins[ids]
    normal, tangent = axes(env)
    # Spread starts along each ramp without letting reset height depend on world z.
    dx = (torch.rand(len(ids), device=env.device) - .5) * 2.
    dy = (torch.rand(len(ids), device=env.device) - .5) * 4.
    pos = terrain.env_origins[ids] + tangent[ids] * dx[:, None] + normal[ids] * .35
    pos[:, 1] += dy
    zeros = torch.zeros(len(ids), device=env.device)
    slope_q = quat_from_euler_xyz(zeros, -torch.deg2rad(angle), zeros)
    jitter = (torch.rand((len(ids), 3), device=env.device) - .5) * torch.tensor([.06, .06, .16], device=env.device)
    q = quat_mul(slope_q, quat_from_euler_xyz(*jitter.unbind(-1)))
    robot.write_root_pose_to_sim(torch.cat((pos, q), -1), env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros((len(ids), 6), device=env.device), env_ids=ids)
    for name in ('steps', 'count', 'error2', 'forward', 'expected', 'lateral', 'aligned', 'heading', 'bad_contact'):
        s[name][ids] = 0
    s['failed'][ids] = False
    s['start'][ids] = pos
    s['last'][ids] = pos


def slope_failure_and_metrics(env):
    s, robot = state(env), env.scene['robot']
    normal, tangent = axes(env)
    pos, velocity = robot.data.root_link_pos_w, robot.data.root_link_lin_vel_w
    up = quat_apply_inverse(robot.data.root_link_quat_w, normal)[:, 2]
    forward_axis = torch.zeros_like(normal)
    forward_axis[:, 0] = 1.
    facing = quat_apply(robot.data.root_link_quat_w, forward_axis)
    heading_good = (facing * tangent).sum(-1) / torch.sqrt(((facing*tangent).sum(-1))**2 + facing[:, 1]**2).clamp_min(1e-6) > math.cos(math.radians(15))
    sensor = env.scene['contact_forces']
    trunk = sensor.body_names.index('trunk')
    trunk_force = sensor.data.net_forces_w[:, trunk].norm(dim=-1)
    s['bad_contact'] = torch.where(trunk_force > 20., s['bad_contact'] + 1, 0.)
    height = ((pos-env.scene.env_origins)*normal).sum(-1)
    outside = ((pos-env.scene.env_origins)[:, 0].abs() > 10.5) | ((pos-env.scene.env_origins)[:, 1].abs() > 4.5)
    failure = (up < math.cos(math.radians(65))) | (height < .07) | (s['bad_contact'] * env.step_dt > .2) | outside
    s['failed'] |= failure
    s['steps'] += 1
    active = s['steps'] * env.step_dt > 1.
    cmd = env.command_manager.get_command('base_velocity')[:, 0]
    speed = (velocity*tangent).sum(-1)
    s['count'] += active
    s['error2'] += ((speed-cmd).square() + velocity[:, 1].square()) * active
    s['forward'] += ((pos-s['last'])*tangent).sum(-1) * active
    s['expected'] += cmd * env.step_dt * active
    s['lateral'] = torch.maximum(s['lateral'], (pos[:, 1]-s['start'][:, 1]).abs())
    s['aligned'] += (up > math.cos(math.radians(45))) * active
    s['heading'] += heading_good * active
    s['last'].copy_(pos)
    return failure


def finish_episodes(env, env_ids):
    s = state(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[s['steps'][ids] > 0]
    if len(ids):
        values = torch.stack([s[k][ids] for k in ('angle', 'steps', 'count', 'error2', 'forward', 'expected', 'lateral', 'aligned', 'heading', 'failed', 'friction')], -1).cpu().tolist()
        for i, value in zip(ids.cpu().tolist(), values):
            angle, steps, count, e2, forward, expected, lateral, aligned, heading, failed, friction = value
            measured = count * env.step_dt
            command = expected/max(measured, 1e-9)
            row = dict(environment=i, slope_deg=angle, duration_s=steps*env.step_dt, measured_s=measured,
                       command_m_s=command, tracking_rmse=math.sqrt(e2/max(count, 1)), forward_m=forward,
                       lateral_max_m=lateral, aligned_fraction=aligned/max(count, 1),
                       heading_good_fraction=heading/max(count, 1), failed=bool(failed), friction=friction)
            s['completed'].setdefault(i, row)
    return {'target30_fraction': (s['angle'].abs() == 30).float().mean()}


def configuration(num_envs, device, seed):
    cfg = TQBot2StandardWalkRoughEnvCfg()
    cfg.seed, cfg.sim.device, cfg.scene.num_envs = seed, device, num_envs
    cfg.episode_length_s = 12.
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=seed, size=(24., 10.), border_width=2., num_rows=7, num_cols=10,
        curriculum=True, use_cache=False, sub_terrains={
            'flat': RampCfg(proportion=.2, sign=0.),
            'uphill': RampCfg(proportion=.4, sign=1.),
            'downhill': RampCfg(proportion=.4, sign=-1.),
        })
    cfg.scene.terrain.max_init_terrain_level = 2
    cfg.events.high_friction_flat_material = None
    material = cfg.events.randomize_foot_material.params
    material.update(static_friction_range=(.7, 1.), dynamic_friction_range=(.7, 1.),
                    restitution_range=(0., 0.), make_consistent=True)
    cfg.events.randomize_reset_base = EventTermCfg(func=reset_on_ramp, mode='reset')
    cfg.events.randomize_apply_external_force_torque = None
    cfg.events.randomize_push_robot = None
    # Retain the parent's mass/COM/gain randomization and observation interface.
    cfg.curriculum.terrain_levels = CurriculumTermCfg(func=finish_episodes)
    cfg.terminations.terrain_out_of_bounds = None
    cfg.terminations.high_friction_flat_exit = None
    cfg.terminations.slope_failure = TerminationTermCfg(func=slope_failure_and_metrics)
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = command.rel_heading_envs = 0.
    command.resampling_time_range = (20., 20.)
    command.ranges.lin_vel_x = (.3, .6)
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (0., 0.)
    command.debug_vis = False
    cfg.observations.policy.enable_corruption = True
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**20
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**20
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**22
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**20
    cfg.log_dir = None
    return cfg
