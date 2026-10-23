"""Bounded simulation-only physics-rate recorder. GPU integration is not yet validated.

Uses deterministic actor mean and a controlled evaluation protocol, not training overrides.
Run only with checkpoints you trust (PyTorch deserialization).
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant', choices=('free', 'v4'), default='free')
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seconds', type=float, default=8.)
parser.add_argument('--speed', type=float, default=.5)
parser.add_argument('--num_envs', type=int, default=24)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--terrain', choices=('mixed', 'slope'), default='mixed')
parser.add_argument('--slope-deg', type=float, default=5.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.seconds <= 2 or args.num_envs < 1 or not 0 <= args.slope_deg <= 30:
    parser.error('Use seconds>2, environments>0, slope angle within0..30 degrees')
if args.output.exists():
    parser.error('Output directory must not exist; preserve prior evaluations')
app = AppLauncher(args).app

import hashlib
import json
import math
import numpy as np
import torch
import gymnasium as gym
from rsl_rl.models import MLPModel
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.io import dump_yaml
import robot_lab.tasks
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.free_gait_env_cfg import TQBot2FreeGaitEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.reward_v4_env_cfg import TQBot2RewardV4EnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.foot_ground import terrain_surface_at_feet


def main():
    torch.set_num_threads(2)
    cfg = (TQBot2FreeGaitEnvCfg if args.variant == 'free' else TQBot2RewardV4EnvCfg)()
    cfg.seed, cfg.sim.device, cfg.scene.num_envs = args.seed, args.device, args.num_envs
    generator = cfg.scene.terrain.terrain_generator
    generator.seed = args.seed
    generator.curriculum = False
    cfg.scene.terrain.max_init_terrain_level = 0
    if args.terrain == 'slope':
        terrain = generator.sub_terrains['hf_pyramid_slope'].copy()
        terrain.slope_range = (math.tan(math.radians(args.slope_deg)),)*2
        terrain.proportion = 1.
        generator.sub_terrains = {'hf_pyramid_slope': terrain}
        generator.num_rows, generator.num_cols = 1, 1
    else:
        generator.num_rows, generator.num_cols = 5, 6
    cfg.curriculum.terrain_levels = cfg.curriculum.terrain_breakdown = None
    for name in ('command_levels_lin_vel', 'command_levels_ang_vel'):
        if hasattr(cfg.curriculum, name):
            setattr(cfg.curriculum, name, None)
    # Bounded replay counts failures explicitly and avoids automatic reset discontinuities.
    cfg.terminations.time_out = cfg.terminations.terrain_out_of_bounds = None
    cfg.observations.policy.enable_corruption = False
    for name in ('randomize_push_robot', 'push_robot', 'randomize_apply_external_force_torque'):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.events.randomize_reset_base.params = {'pose_range': {}, 'velocity_range': {}}
    cfg.events.randomize_reset_joints.params['position_range'] = (1.,1.)
    cfg.events.randomize_reset_joints.params['velocity_range'] = (0.,0.)
    command_cfg = cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = command_cfg.rel_heading_envs = 0.
    command_cfg.resampling_time_range = (1000.,1000.)
    command_cfg.ranges.lin_vel_x = (args.speed,args.speed)
    command_cfg.ranges.lin_vel_y = command_cfg.ranges.ang_vel_z = (0.,0.)
    command_cfg.debug_vis = False
    cfg.log_dir = None
    args.output.mkdir(parents=True)
    task = f'RobotLab-Isaac-Velocity-Rough-TQBot2-{"FreeGait" if args.variant == "free" else "RewardV4"}Scratch-v0'
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset(seed=args.seed)
        actor = MLPModel(obs, {'actor':['policy']}, 'actor', 12, hidden_dims=[512,256,128],
                         activation='elu', obs_normalization=False,
                         distribution_cfg={'class_name':'GaussianDistribution','init_std':1.,'std_type':'scalar'}).to(env.device)
        actor.load_state_dict(torch.load(args.checkpoint, map_location=env.device, weights_only=False)['actor_state_dict'], strict=True)
        actor.eval()
        robot, sensor = env.scene['robot'], env.scene['contact_forces']
        names = ['FL_foot','FR_foot','RL_foot','RR_foot']
        feet = robot.find_bodies(names, preserve_order=True)[0]
        contacts = sensor.find_bodies(names, preserve_order=True)[0]
        term = env.command_manager.get_term('base_velocity')
        term.vel_command_b[:] = 0.
        term.is_standing_env[:] = True
        term.is_heading_env[:] = False
        obs = env.observation_manager.compute()
        with torch.inference_mode():
            for _ in range(round(1./env.step_dt)):
                obs, _, _, _, _ = env.step(actor(obs, stochastic_output=False))
        term.vel_command_b[:] = torch.tensor([args.speed,0.,0.], device=env.device)
        term.is_standing_env[:] = False
        obs = env.observation_manager.compute()
        keys = ('foot_force','foot_z_body','foot_clearance','foot_tangent_speed','upright','velocity_b','root_pos','command','local_slope_deg')
        trace = {key:[] for key in keys}
        original_update = env.scene.update
        def recorded_update(dt, *pos, **kw):
            result = original_update(dt, *pos, **kw)
            positions = robot.data.body_link_pos_w[:, feet]
            hits, normals, valid = terrain_surface_at_feet(env, positions)
            body = quat_apply_inverse(robot.data.root_link_quat_w[:,None,:].expand(-1,4,-1), positions-robot.data.root_link_pos_w[:,None,:])
            velocity = robot.data.body_link_lin_vel_w[:,feet]
            tangent = velocity-(velocity*normals).sum(-1,keepdim=True)*normals
            values = {'foot_force':sensor.data.net_forces_w[:,contacts].norm(dim=-1),
                      'foot_z_body':body[...,2],
                      'foot_clearance':torch.where(valid,((positions-hits)*normals).sum(-1)-.012,float('nan')),
                      'foot_tangent_speed':torch.where(valid,tangent.norm(dim=-1),float('nan')),
                      'upright':-robot.data.projected_gravity_b[:,2], 'velocity_b':robot.data.root_lin_vel_b,
                      'root_pos':robot.data.root_link_pos_w, 'command':term.vel_command_b,
                      'local_slope_deg':torch.where(valid,torch.rad2deg(torch.acos(normals[...,2].clamp(-1,1))),float('nan'))}
            for key,value in values.items():
                trace[key].append(value.detach().cpu().numpy().copy())
            return result
        env.scene.update = recorded_update
        with torch.inference_mode():
            for _ in range(round(args.seconds/env.step_dt)):
                obs, _, _, _, _ = env.step(actor(obs, stochastic_output=False))
        env.scene.update = original_update
        arrays = {key:np.stack(value) for key,value in trace.items()}
        expected = round(args.seconds/env.step_dt)*cfg.decimation
        if len(arrays['foot_force']) != expected:
            raise RuntimeError('Recorder did not capture every physics step')
        np.savez_compressed(args.output/'trace.npz', **arrays, t=(np.arange(expected)+1)*env.physics_dt,
                            episode=np.zeros(arrays['upright'].shape, dtype=int))
        dump_yaml(str(args.output/'evaluation_env.yaml'), cfg)
        metadata = {**vars(args), 'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    'foot_names':names, 'physics_dt':env.physics_dt, 'control_dt':env.step_dt,
                    'protocol':'1s zero-command settle; bounded actor-mean replay. Startup mass/COM/material and PD randomization retained; fixed seed. No pushes/no observation noise/no automatic resets. Failure samples remain in trace. Pyramid includes a flat platform: nominal angle alone does not prove slope traversal.',
                    'slip':'World foot velocity projected onto local terrain tangent; ground stationary',
                    'validation':'CPU syntax only for this portable recorder; validate in Isaac before quantitative use'}
        (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2, default=str)+'\n')
    finally:
        env.close()


try:
    main()
finally:
    app.close()
