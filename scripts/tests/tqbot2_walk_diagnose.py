"""Finite, frozen-policy diagnostics for TQBot2 and the reference Go2 walker."""
import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, action='append', default=[])
parser.add_argument('--robot', choices=('tqbot2', 'go2'), default='tqbot2')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=32)
parser.add_argument('--seconds', type=float, default=10.)
parser.add_argument('--baseline', action='store_true', help='Also hold the default joint targets without a policy')
parser.add_argument('--baseline-only', action='store_true')
parser.add_argument('--nominal-gains', action='store_true')
parser.add_argument('--physics-dt', type=float, default=.005)
parser.add_argument('--upright-only', action='store_true')
parser.add_argument('--torque-diagnostics', action='store_true', help='Sample explicit actuator effort at every physics substep')
parser.add_argument('--kp', type=float, help='Override nominal stiffness for a reproducible comparison')
parser.add_argument('--kd', type=float, help='Override nominal damping for a reproducible comparison')
parser.add_argument('--calf-kd', type=float, help='Override only the calf nominal damping in this diagnostic')
parser.add_argument('--verify-default-gains', action='store_true', help='Check the walking, rolling and recovery defaults before simulation')
parser.add_argument('--actuator-ablation', action='store_true', help='Paired fixed-gain probes: friction, calf inertia, armature and calf damping')
parser.add_argument('--ablation-probes', nargs='+', choices=('baseline', 'friction_zero', 'friction_all_zero', 'dynamic_friction_equal', 'calf_inertia_x3', 'calf_armature_0p003', 'calf_kd_0p1', 'calf_kd_0p2'),
                    help='friction_zero clears the new friction properties and retains the initial legacy coefficient (now zero for TQBot2); friction_all_zero explicitly clears both')
parser.add_argument('--disable-self-collisions', action='store_true')
parser.add_argument('--solver-velocity-iterations', type=int)
parser.add_argument('--free-air', action='store_true', help='Fix the trunk at 2m, disable gravity and perturb calf angles by 0.05 rad')
parser.add_argument('--merged-feet', action='store_true', help='Use the original merged-foot model for a fixed-target physics diagnostic')
parser.add_argument('--joint-friction', type=float, help='Override joint friction only in this diagnostic')
parser.add_argument('--video-dir', type=Path, help='Record actual renderer frames at every physics substep, encoded at 25fps for 8x slow motion with 5ms physics')
parser.add_argument('--video-start', type=float, default=3.)
parser.add_argument('--video-seconds', type=float, default=1.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint and not args.baseline_only:
    parser.error('--checkpoint is required unless --baseline-only is selected')
if args.actuator_ablation and not (args.baseline_only and args.nominal_gains):
    parser.error('--actuator-ablation requires --baseline-only --nominal-gains')
if args.video_dir:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.models import MLPModel
from isaaclab.utils.math import quat_from_euler_xyz
import robot_lab.tasks
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import TQBot2StandardWalkRoughEnvCfg


def main():
    torch.set_num_threads(2)
    checked_defaults = {}
    if args.verify_default_gains:
        from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.rough_env_cfg import TQBot2RoughEnvCfg
        from robot_lab.tasks.manager_based.locomotion.roll.config.quadruped.tqbot2.rough_env_cfg import TQBot2RollRoughEnvCfg
        from robot_lab.tasks.manager_based.locomotion.recover.config.tqbot2.recover_env_cfg import TQBot2RecoverEnvCfg
        for config_class in (TQBot2StandardWalkRoughEnvCfg, TQBot2RoughEnvCfg, TQBot2RollRoughEnvCfg, TQBot2RecoverEnvCfg):
            resolved = config_class()
            checked_defaults[config_class.__name__] = {}
            for group, actuator in resolved.scene.robot.actuators.items():
                assert actuator.stiffness == 14. and actuator.damping == .6, (config_class.__name__, group)
                checked_defaults[config_class.__name__][group] = {'kp': actuator.stiffness, 'kd': actuator.damping}
        print('DEFAULT_GAINS_VERIFIED', json.dumps(checked_defaults), flush=True)
    if args.robot == 'go2':
        from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
        cfg = UnitreeGo2RoughEnvCfg()
        task = 'RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0'
    else:
        cfg = TQBot2StandardWalkRoughEnvCfg()
        task = 'RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0'
    for actuator in cfg.scene.robot.actuators.values():
        if args.kp is not None:
            actuator.stiffness = args.kp
        if args.kd is not None:
            actuator.damping = args.kd
        if args.joint_friction is not None:
            actuator.friction = args.joint_friction
    if args.calf_kd is not None:
        if 'calf' not in cfg.scene.robot.actuators:
            parser.error('--calf-kd requires a separate calf actuator group')
        cfg.scene.robot.actuators['calf'].damping = args.calf_kd
    if args.disable_self_collisions:
        import isaaclab.sim as sim_utils
        if args.robot == 'tqbot2':
            from robot_lab.assets.tqbot_spawn import spawn_tqbot2_from_urdf
            cfg.scene.robot.spawn.func = spawn_tqbot2_from_urdf
        else:
            cfg.scene.robot.spawn.func = sim_utils.spawn_from_urdf
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    if args.solver_velocity_iterations is not None:
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = args.solver_velocity_iterations
    if args.free_air:
        cfg.scene.robot.spawn.fix_base = True
        cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        cfg.scene.robot.init_state.pos = (0., 0., 2.)
    if args.merged_feet:
        if args.robot != 'tqbot2' or not args.baseline_only:
            parser.error('--merged-feet is only for the TQBot2 fixed-target baseline')
        from isaaclab.managers import SceneEntityCfg
        cfg.scene.robot.spawn.asset_path = str(Path(cfg.scene.robot.spawn.asset_path).with_name('TQBot2_isaac.urdf'))
        cfg.foot_link_name = '.*_calf'
        def remap_feet(value):
            if isinstance(value, SceneEntityCfg):
                value.body_names = remap_feet(value.body_names)
            elif isinstance(value, str):
                return value.replace('_foot', '_calf')
            elif isinstance(value, dict):
                return {k: remap_feet(v) for k, v in value.items()}
            elif isinstance(value, (tuple, list)):
                return type(value)(remap_feet(v) for v in value)
            return value
        for terms in (cfg.rewards, cfg.events):
            for term in vars(terms).values():
                if term is not None and hasattr(term, 'params'):
                    term.params = remap_feet(term.params)
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.sim.dt = args.physics_dt
    cfg.decimation = round(.02 / args.physics_dt)
    assert abs(cfg.sim.dt * cfg.decimation - .02) < 1e-8
    if args.nominal_gains:
        cfg.events.randomize_actuator_gains = None
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 5.
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.height_scanner = cfg.scene.height_scanner_base = None
    cfg.observations.critic.height_scan = None
    cfg.observations.policy.enable_corruption = False
    cfg.curriculum.terrain_levels = None
    if hasattr(cfg.curriculum, 'terrain_breakdown'):
        cfg.curriculum.terrain_breakdown = None
    cfg.terminations.time_out = cfg.terminations.terrain_out_of_bounds = None
    cfg.events.randomize_push_robot = cfg.events.randomize_apply_external_force_torque = None
    cfg.events.randomize_reset_base.params = {'pose_range': {}, 'velocity_range': {}}
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.
    cfg.commands.base_velocity.resampling_time_range = (1000., 1000.)
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    cfg.log_dir = None
    if args.video_dir:
        cfg.viewer.resolution = (640, 480)
        cfg.viewer.origin_type = 'env'
        cfg.viewer.env_index = 0
        target_height = 2. if args.free_air else .22
        cfg.viewer.eye = (.95, .8, target_height + .38)
        cfg.viewer.lookat = (0., 0., target_height)
    env = gym.make(task, cfg=cfg, render_mode='rgb_array' if args.video_dir else None).unwrapped
    try:
        obs, _ = env.reset(seed=42)
        robot = env.scene['robot']
        sensor = env.scene['contact_forces']
        import re
        feet = [i for i, n in enumerate(sensor.body_names) if re.fullmatch(cfg.foot_link_name, n)]
        trunk = sensor.body_names.index(cfg.base_link_name)
        actor = MLPModel(obs, {'actor': ['policy']}, 'actor', 12, hidden_dims=[512, 256, 128],
                         activation='elu', obs_normalization=False,
                         distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar'}).to(env.device)
        actor.eval()
        action = env.action_manager.get_term('joint_pos')
        original_inertias = robot.root_physx_view.get_inertias().clone()
        original_armature = robot.data.default_joint_armature.clone()
        original_friction = robot.data.default_joint_friction_coeff.clone()
        original_viscous_friction = robot.data.default_joint_viscous_friction_coeff.clone()
        imported_friction_properties = robot.root_physx_view.get_dof_friction_properties()[0].tolist()
        original_legacy_friction = robot.root_physx_view.get_dof_friction_coefficients().clone()
        original_dynamic_friction = robot.data.default_joint_dynamic_friction_coeff.clone()
        calf_body_ids, _ = robot.find_bodies('.*_calf')
        calf_joint_ids, _ = robot.find_joints('.*_calf_joint')
        video_state = {'active': False, 'substeps': 0, 'writer': None, 'frames': 0}
        if args.video_dir:
            import imageio.v2 as imageio
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            args.video_dir.mkdir(parents=True, exist_ok=True)
            video_font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 17)
            original_scene_update = env.scene.update
            front_right_calf = robot.joint_names.index('FR_calf_joint')
            def recorded_scene_update(dt, *update_args, **update_kwargs):
                result = original_scene_update(dt, *update_args, **update_kwargs)
                if not video_state['active']:
                    return result
                video_state['substeps'] += 1
                elapsed = video_state['substeps'] * env.physics_dt
                if args.video_start <= elapsed < args.video_start + args.video_seconds:
                    frame = env.render()
                    canvas = Image.fromarray(frame)
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((0, 0, 640, 92), fill=(18, 22, 27))
                    posture_label = '机身正常朝向，固定悬空2m，无重力' if args.free_air else '地面测试，重力开启，机身自由'
                    dq = robot.data.joint_vel[0, front_right_calf].item()
                    lines = [posture_label, video_state['label'],
                             f't={elapsed:.3f}s  FR小腿速度={dq:+.2f}rad/s  物理{env.physics_dt*1000:g}ms',
                             f'实际仿真逐物理步采样 / {1/env.physics_dt:g}fps采集 / 25fps播放（慢放{1/env.physics_dt/25:g}倍）']
                    for i, line in enumerate(lines):
                        draw.text((10, 4 + i * 22), line, font=video_font, fill='white')
                    video_state['writer'].append_data(np.asarray(canvas))
                    if video_state['frames'] == 0:
                        canvas.save(video_state['path'].with_suffix('.png'))
                    video_state['frames'] += 1
                    if video_state['frames'] % 50 == 0:
                        print('VIDEO_FRAMES', video_state['path'].name, video_state['frames'], flush=True)
                return result
            env.scene.update = recorded_scene_update
            # Create and warm the render product without stepping physics.
            for _ in range(8):
                env.render()
        def mass_probe():
            matrix = robot.root_physx_view.get_generalized_mass_matrices()[0].double()
            inverse = torch.linalg.inv(matrix)
            return {'matrix_shape': list(matrix.shape), 'joint_names': robot.joint_names,
                    'mass_matrix': matrix.tolist(), 'effective_inertia_inverse_diagonal': (1 / inverse.diagonal()).tolist(),
                    'joint_positions_rad': robot.data.joint_pos[0].tolist(),
                    'calf_inertia_tensors': robot.root_physx_view.get_inertias()[0, calf_body_ids].tolist(),
                    'dof_armature': robot.root_physx_view.get_dof_armatures()[0].tolist()}
        torque_state = {'active': False, 'steady': False, 'windows': {}, 'previous_velocity': {}, 'traces': {}}
        if args.torque_diagnostics:
            for group, actuator in robot.actuators.items():
                original_compute = actuator.compute

                def measured_compute(*compute_args, _original=original_compute, _actuator=actuator, _group=group, **compute_kwargs):
                    result = _original(*compute_args, **compute_kwargs)
                    if torque_state['active']:
                        computed = _actuator.computed_effort
                        applied = _actuator.applied_effort
                        velocity = compute_kwargs.get('joint_vel')
                        if velocity is None:
                            velocity = compute_args[2]
                        previous = torque_state['previous_velocity'].get(_group)
                        acceleration = (velocity - previous) / env.physics_dt if previous is not None else None
                        if torque_state['steady'] and len(torque_state['traces'][_group]) < 64:
                            position = compute_kwargs.get('joint_pos')
                            if position is None:
                                position = compute_args[1]
                            torque_state['traces'][_group].append({
                                'q_rad_env0': position[0].tolist(), 'dq_rad_s_env0': velocity[0].tolist(),
                                'computed_nm_env0': computed[0].tolist(), 'applied_nm_env0': applied[0].tolist()})
                        for window in (('all', 'steady') if torque_state['steady'] else ('all',)):
                            stats = torque_state['windows'][window][_group]
                            stats['samples'] += applied.shape[0]
                            stats['peak_applied'] = torch.maximum(stats['peak_applied'], applied.abs().amax(0))
                            stats['peak_computed'] = torch.maximum(stats['peak_computed'], computed.abs().amax(0))
                            stats['sum_abs_applied'] += applied.abs().sum(0)
                            stats['at_absolute_cap'] += (applied.abs() >= _actuator.effort_limit - 1e-4).sum(0)
                            stats['computed_above_absolute_cap'] += (computed.abs() > _actuator.effort_limit + 1e-4).sum(0)
                            stats['clipped_by_dc_envelope'] += ((computed - applied).abs() > 1e-4).sum(0)
                            stats['sum_velocity_sq'] += velocity.square().sum(0)
                            if acceleration is not None:
                                stats['acceleration_samples'] += applied.shape[0]
                                stats['sum_acceleration_sq'] += acceleration.square().sum(0)
                                stats['velocity_sign_flip'] += ((velocity * previous < 0) &
                                                               (velocity.abs() > .05) & (previous.abs() > .05)).sum(0)
                        torque_state['previous_velocity'][_group] = velocity.clone()
                    return result

                actuator.compute = measured_compute
        steps = round(args.seconds / env.step_dt)
        output = {'robot': args.robot, 'task': task, 'terrain': 'plane', 'command_m_s': .5, 'seconds': args.seconds,
                  'num_envs': env.num_envs, 'seed': 42,
                  'physics_dt': cfg.sim.dt, 'nominal_gains': args.nominal_gains,
                  'diagnostic_changes': {'free_air': args.free_air, 'merged_feet': args.merged_feet,
                                         'self_collisions': cfg.scene.robot.spawn.articulation_props.enabled_self_collisions,
                                         'solver_velocity_iterations': cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count},
                  'nominal_actuator_gains': {group: {'kp': actuator.stiffness, 'kd': actuator.damping}
                                            for group, actuator in cfg.scene.robot.actuators.items()},
                  'checked_default_gain_configs': checked_defaults,
                  'robot_body_names': robot.body_names,
                  'default_mass_kg': robot.data.default_mass[0].sum().item(),
                  'initial_physx_friction_properties': imported_friction_properties,
                  'initial_legacy_joint_friction': original_legacy_friction[0].tolist(),
                  'actuator_parameters': {group: {
                      'model': type(actuator).__name__,
                      'effort_limit_nm': actuator.effort_limit[0].tolist(),
                      'velocity_limit_rad_s': actuator.velocity_limit[0].tolist(),
                      'effort_limit_sim_nm': actuator.effort_limit_sim[0].tolist(),
                      'velocity_limit_sim_rad_s': actuator.velocity_limit_sim[0].tolist(),
                      'friction': actuator.friction[0].tolist(),
                      'dynamic_friction': actuator.dynamic_friction[0].tolist(),
                      'viscous_friction': actuator.viscous_friction[0].tolist(),
                  } for group, actuator in robot.actuators.items()},
                  'randomization': 'training material/mass/COM and reset gains; no pushes or external force; no observation noise',
                  'reward_units': 'weighted reward per second, averaged over second half', 'profiles': []}
        profiles = [] if args.baseline_only else [(checkpoint, 'actor_mean') for checkpoint in args.checkpoint]
        if args.baseline or args.baseline_only:
            profiles.append((None, 'zero_action'))
        probes = ('baseline', 'friction_zero', 'calf_inertia_x3', 'calf_armature_0p003', 'calf_kd_0p1') if args.actuator_ablation else ('baseline',)
        if args.ablation_probes is not None:
            probes = args.ablation_probes
        profiles = [(checkpoint, control, probe) for checkpoint, control in profiles for probe in probes]
        for checkpoint, control, probe in profiles:
            if checkpoint is not None:
                actor.load_state_dict(torch.load(checkpoint, map_location=env.device, weights_only=False)['actor_state_dict'])
            for posture in (('upright',) if args.upright_only else ('upright', 'random_orientation')):
                obs, _ = env.reset(seed=42)
                if args.actuator_ablation:
                    robot.root_physx_view.set_dof_friction_coefficients(original_legacy_friction, torch.arange(env.num_envs))
                    robot.root_physx_view.set_inertias(original_inertias, torch.arange(env.num_envs))
                    armature = original_armature.clone()
                    if probe == 'calf_armature_0p003':
                        armature[:, calf_joint_ids] = .003
                    robot.write_joint_armature_to_sim(armature)
                    if probe == 'friction_all_zero':
                        robot.root_physx_view.set_dof_friction_coefficients(torch.zeros_like(original_legacy_friction), torch.arange(env.num_envs))
                    zero_new_friction = probe in ('friction_zero', 'friction_all_zero')
                    # Write the three new properties together, after any legacy
                    # setter, and verify the real engine values for zero probes.
                    robot.write_joint_friction_coefficient_to_sim(
                        0. if zero_new_friction else original_friction,
                        0. if zero_new_friction else
                        (original_friction if probe == 'dynamic_friction_equal' else original_dynamic_friction),
                        0. if zero_new_friction else original_viscous_friction,
                    )
                    if zero_new_friction:
                        assert robot.root_physx_view.get_dof_friction_properties().abs().max().item() == 0.
                    if probe == 'calf_inertia_x3':
                        inertias = original_inertias.clone()
                        inertias[:, calf_body_ids] *= 3.
                        robot.root_physx_view.set_inertias(inertias, torch.arange(env.num_envs))
                    for group, actuator in robot.actuators.items():
                        actuator.stiffness[:] = cfg.scene.robot.actuators[group].stiffness
                        actuator.damping[:] = cfg.scene.robot.actuators[group].damping
                        if probe == 'calf_kd_0p1' and group == 'calf':
                            actuator.damping[:] = .1
                        if probe == 'calf_kd_0p2' and group == 'calf':
                            actuator.damping[:] = .2
                state = robot.data.default_root_state.clone()
                state[:, :3] += env.scene.env_origins
                if posture == 'random_orientation':
                    euler = (torch.rand((env.num_envs, 3), device=env.device) * 2 - 1) * 3.14
                    state[:, 3:7] = quat_from_euler_xyz(*euler.unbind(-1))
                    state[:, 2] += .1
                state[:, 7:] = 0.
                robot.write_root_pose_to_sim(state[:, :7])
                robot.write_root_velocity_to_sim(state[:, 7:])
                if args.free_air:
                    positions = robot.data.default_joint_pos.clone()
                    positions[:, calf_joint_ids] += .05
                    robot.write_joint_state_to_sim(positions, torch.zeros_like(positions))
                command = env.command_manager.get_term('base_velocity')
                command.vel_command_b[:] = torch.tensor([.5, 0., 0.], device=env.device)
                command.is_standing_env[:] = command.is_heading_env[:] = False
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(env.physics_dt)
                obs = env.observation_manager.compute()
                starting_mass_probe = None
                if args.video_dir:
                    video_state.update(active=False, substeps=0, frames=0)
                    for _ in range(4):
                        env.render()
                    path = args.video_dir / f'{args.robot}_{"air" if args.free_air else "ground"}_{probe}_slow8x.mp4'
                    video_state['path'] = path
                    video_state['label'] = {'baseline': 'Kp14 / Kd0.6，原配置', 'calf_kd_0p1': '诊断对照：仅小腿Kd改为0.1',
                                            'friction_zero': '诊断对照：新摩擦=0，旧系数0.2保留',
                                            'friction_all_zero': '诊断对照：新旧摩擦全部=0'}.get(probe, probe)
                    video_state['writer'] = imageio.get_writer(path, fps=25, codec='libx264', quality=8, macro_block_size=16)
                    video_state['active'] = True
                totals = torch.zeros(8, device=env.device)
                rewards = torch.zeros(len(env.reward_manager.active_terms), device=env.device)
                joint_targets = torch.zeros(12, device=env.device)
                outside_targets = torch.zeros(12, device=env.device)
                joint_pos = torch.zeros(12, device=env.device)
                per_env_velocity_sq = torch.zeros(env.num_envs, device=env.device)
                per_env_acceleration_sq = torch.zeros(env.num_envs, device=env.device)
                per_env_upright = torch.zeros(env.num_envs, device=env.device)
                peak_speed = 0.
                torque_state['windows'] = {
                    window: {group: {'samples': 0, 'acceleration_samples': 0, **{
                        field: torch.zeros(len(actuator.joint_names), device=env.device)
                        for field in ('peak_applied', 'peak_computed', 'sum_abs_applied', 'at_absolute_cap',
                                      'computed_above_absolute_cap', 'clipped_by_dc_envelope', 'sum_velocity_sq',
                                      'sum_acceleration_sq', 'velocity_sign_flip')}}
                        for group, actuator in robot.actuators.items()}
                    for window in ('all', 'steady')}
                torque_state['active'] = args.torque_diagnostics
                torque_state['previous_velocity'] = {}
                torque_state['traces'] = {group: [] for group in robot.actuators}
                with torch.no_grad():
                    for step in range(steps):
                        torque_state['steady'] = step >= steps // 2
                        raw_action = actor(obs, stochastic_output=False) if control == 'actor_mean' else torch.zeros((env.num_envs, 12), device=env.device)
                        obs, _, terminated, truncated, _ = env.step(raw_action)
                        assert not terminated.any() and not truncated.any()
                        if step == 0 and args.actuator_ablation:
                            # PhysX refreshes the generalized matrix after an actual physics step.
                            starting_mass_probe = mass_probe()
                        if step >= steps // 2:
                            q = robot.data.projected_gravity_b[:, 2]
                            h = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
                            vel = robot.data.root_lin_vel_b
                            contacts = sensor.data.net_forces_w.norm(dim=-1)
                            totals += torch.stack([(q < -.866).float().mean(), h.mean(), vel[:, 0].mean(),
                                                   (vel[:, :2] - torch.tensor([.5, 0.], device=env.device)).norm(dim=-1).mean(),
                                                   (contacts[:, feet] > 1.).sum(-1).float().mean(),
                                                   (contacts[:, trunk] > 1.).float().mean(),
                                                   robot.data.joint_vel.square().mean().sqrt(),
                                                   robot.data.joint_acc.square().mean().sqrt()])
                            rewards += env.reward_manager._step_reward.mean(0)
                            targets = action.processed_actions
                            ids = action._joint_ids
                            limits = robot.data.joint_pos_limits[:, ids]
                            joint_targets += targets.mean(0)
                            outside_targets += ((targets < limits[..., 0]) | (targets > limits[..., 1])).float().mean(0)
                            joint_pos += robot.data.joint_pos[:, ids].mean(0)
                            per_env_velocity_sq += robot.data.joint_vel.square().mean(1)
                            per_env_acceleration_sq += robot.data.joint_acc.square().mean(1)
                            per_env_upright += (q < -.866).float()
                            peak_speed = max(peak_speed, robot.data.joint_vel.abs().max().item())
                count = steps - steps // 2
                if args.video_dir:
                    video_state['active'] = False
                    video_state['writer'].close()
                    video_state['writer'] = None
                stats = dict(zip(['upright_fraction', 'root_height_m', 'forward_velocity_m_s', 'velocity_error_m_s',
                                  'foot_contacts', 'trunk_contact_fraction', 'joint_velocity_rms_rad_s', 'joint_acceleration_rms_rad_s2'],
                                 (totals / count).tolist()))
                entry = {'checkpoint': str(checkpoint.resolve()) if checkpoint else None, 'control': control, 'probe': probe, 'posture': posture, **stats,
                         'starting_mass_probe': starting_mass_probe,
                         'peak_joint_speed_rad_s': peak_speed,
                         'weighted_rewards': dict(zip(env.reward_manager.active_terms, (rewards / count).tolist())),
                         'joint_names': action._joint_names, 'mean_targets_rad': (joint_targets / count).tolist(),
                         'target_outside_physical_limits_fraction': (outside_targets / count).tolist(),
                         'mean_joint_positions_rad': (joint_pos / count).tolist(),
                         'per_env_velocity_rms_rad_s': (per_env_velocity_sq / count).sqrt().tolist(),
                         'per_env_acceleration_rms_rad_s2': (per_env_acceleration_sq / count).sqrt().tolist(),
                         'per_env_upright_fraction': (per_env_upright / count).tolist(),
                         'actual_actuator_gain_ranges': {group: {
                             'kp': [actuator.stiffness.min().item(), actuator.stiffness.max().item()],
                             'kd': [actuator.damping.min().item(), actuator.damping.max().item()]}
                             for group, actuator in robot.actuators.items()},
                         'physx_stiffness_max': robot.root_physx_view.get_dof_stiffnesses().abs().max().item()}
                entry['physx_damping_max'] = robot.root_physx_view.get_dof_dampings().abs().max().item()
                entry['has_implicit_actuators'] = robot._has_implicit_actuators
                entry['legacy_joint_friction'] = robot.root_physx_view.get_dof_friction_coefficients()[0].tolist()
                entry['physx_friction_properties'] = robot.root_physx_view.get_dof_friction_properties()[0].tolist()
                if args.video_dir:
                    entry['video'] = {'path': str(video_state['path'].resolve()), 'frames': video_state['frames'],
                                      'capture_fps': 1/env.physics_dt, 'playback_fps': 25,
                                      'physical_time_window_s': [args.video_start, args.video_start + args.video_seconds],
                                      'environment_index': 0, 're_recorded': True, 'simulation_device': env.device}
                torque_state['active'] = False
                if args.torque_diagnostics:
                    entry['physics_step_trace_env0'] = torque_state['traces']
                    entry['torque_physics_substeps'] = {
                        window: {group: {
                            'joint_names': robot.actuators[group].joint_names,
                            'samples_per_joint': stats['samples'],
                            'absolute_cap_nm': robot.actuators[group].effort_limit[0].tolist(),
                            'peak_applied_nm': stats['peak_applied'].tolist(),
                            'peak_computed_nm': stats['peak_computed'].tolist(),
                            'mean_abs_applied_nm': (stats['sum_abs_applied'] / stats['samples']).tolist(),
                            'velocity_rms_rad_s': (stats['sum_velocity_sq'] / stats['samples']).sqrt().tolist(),
                            'acceleration_rms_rad_s2': (stats['sum_acceleration_sq'] / max(1, stats['acceleration_samples'])).sqrt().tolist(),
                            'velocity_sign_flip_fraction': (stats['velocity_sign_flip'] / max(1, stats['acceleration_samples'])).tolist(),
                            **{field + '_fraction': (stats[field] / stats['samples']).tolist()
                               for field in ('at_absolute_cap', 'computed_above_absolute_cap', 'clipped_by_dc_envelope')},
                        } for group, stats in groups.items()}
                        for window, groups in torque_state['windows'].items()}
                output['profiles'].append(entry)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(output, indent=2) + '\n')
                print('RESULT', checkpoint.name if checkpoint else control, probe, posture, json.dumps(stats), flush=True)
    finally:
        if args.video_dir and video_state['writer'] is not None:
            video_state['writer'].close()
        env.close()


main()
app.close()
