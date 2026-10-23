"""Select representative walking successes/failures and render their exact recorded states.

Run evaluate on CUDA without cameras, then render on CPU with cameras. Rendering
does not integrate physics or rerun the policy, so it preserves the GPU rollout.
Use PYTHONPATH to select the training's frozen source.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=['evaluate', 'render', 'replay-unloaded'], required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seconds', type=float, default=8.)
parser.add_argument('--motion-audit', action='store_true', help='Record targets, foot clearance/contact timing and reward components')
parser.add_argument('--replay-source', type=Path, help='Recorded rollout directory for the unloaded actuator probe')
parser.add_argument('--kp-scale', type=float, default=1., help='Diagnostic stiffness multiplier; does not change training')
parser.add_argument('--effort-scale', type=float, default=1., help='Diagnostic torque-limit multiplier; does not change training')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.mode == 'render':
    args.enable_cameras = True
app = AppLauncher(args).app

import hashlib
import inspect
import json
import math

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.models import MLPModel
from isaaclab.utils.io import dump_yaml
import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.standard_walk_env_cfg import TQBot2StandardWalkRoughEnvCfg

TERRAINS = ['pyramid_stairs', 'pyramid_stairs_inv', 'boxes', 'random_rough',
            'hf_pyramid_slope', 'hf_pyramid_slope_inv']
LABELS = ['下台阶（凸形）', '上台阶（凹形）', '随机方块', '随机起伏',
          '下坡（凸形）', '上坡（凹形）']


def measure_terrain_surface(env, position):
    """Diagnostic raycast independent of the checkpoint's frozen reward code."""
    from isaaclab.sensors import RayCaster
    from isaaclab.utils.warp import raycast_mesh

    if env.scene.terrain.cfg.terrain_type == 'plane':
        hits = position.clone()
        hits[..., 2] = 0.
        normals = torch.zeros_like(position)
        normals[..., 2] = 1.
        return hits, normals, torch.ones_like(position[..., 0], dtype=torch.bool)
    path = env.scene['height_scanner'].cfg.mesh_prim_paths[0]
    starts = position.clone()
    starts[..., 2] += 1.
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1.
    hits, _, normals, _ = raycast_mesh(starts, directions, RayCaster.meshes[path],
                                      max_dist=5., return_normal=True)
    valid = torch.isfinite(hits).all(-1) & torch.isfinite(normals).all(-1)
    normals *= torch.where(normals[..., 2:3] < 0., -1., 1.)
    return hits, normals, valid


def configuration():
    cfg = TQBot2StandardWalkRoughEnvCfg()
    for actuator in cfg.scene.robot.actuators.values():
        actuator.stiffness *= args.kp_scale
        actuator.effort_limit_sim *= args.effort_scale
    if args.mode == 'replay-unloaded':
        assert args.replay_source is not None
        cfg.scene.robot.spawn.fix_base = True
        cfg.scene.robot.spawn.rigid_props.disable_gravity = True
        cfg.scene.robot.init_state.pos = (0., 0., 2.)
    cfg.seed = 42
    cfg.sim.device = args.device
    cfg.scene.num_envs = 24
    cfg.sim.render_interval = 8
    cfg.viewer.resolution = (960, 640)
    cfg.viewer.origin_type = 'world'
    generator = cfg.scene.terrain.terrain_generator
    generator.seed = 42
    generator.sub_terrains = {name: generator.sub_terrains[name].copy() for name in TERRAINS}
    for terrain in generator.sub_terrains.values():
        terrain.proportion = 1. / len(TERRAINS)
    generator.num_rows = 10  # Preserve training's difficulty-to-level mapping.
    generator.num_cols = 6
    generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = cfg.curriculum.terrain_breakdown = None
    cfg.terminations.time_out = cfg.terminations.terrain_out_of_bounds = None
    cfg.observations.policy.enable_corruption = False
    cfg.events.randomize_push_robot = cfg.events.randomize_apply_external_force_torque = None
    cfg.events.randomize_reset_base.params = {'pose_range': {}, 'velocity_range': {}}
    cfg.events.randomize_reset_joints.params['position_range'] = (1., 1.)
    cfg.events.randomize_reset_joints.params['velocity_range'] = (0., 0.)
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0.
    command.rel_heading_envs = 0.
    command.resampling_time_range = (1000., 1000.)
    command.debug_vis = False
    command.ranges.lin_vel_x = (.5, .5)
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (0., 0.)
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
    cfg.log_dir = None
    return cfg


def evaluate(env, cfg):
    terrain = env.scene.terrain
    levels = torch.tensor([0, 0, 3, 3] * 6, device=env.device)
    assert terrain.terrain_types.tolist() == [i for i in range(6) for _ in range(4)]
    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
    env.scene.env_origins[:] = terrain.env_origins
    obs, _ = env.reset(seed=42)
    actor = MLPModel(obs, {'actor': ['policy']}, 'actor', 12, hidden_dims=[512, 256, 128],
                     activation='elu', obs_normalization=False,
                     distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar'}).to(env.device)
    saved = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    actor.eval()
    command = env.command_manager.get_term('base_velocity')
    command.vel_command_b[:] = 0.
    command.is_heading_env[:] = False
    command.is_standing_env[:] = True
    obs = env.observation_manager.compute()
    robot = env.scene['robot']
    sensor = env.scene['contact_forces']
    calf_ids = [i for i, name in enumerate(sensor.body_names) if name.endswith('_calf')]
    # Remove the initial drop onto the ground from the walking comparison.
    with torch.no_grad():
        for _ in range(50):
            obs, _, _, _, _ = env.step(actor(obs, stochastic_output=False))
    command.vel_command_b[:] = torch.tensor([.5, 0., 0.], device=env.device)
    command.is_standing_env[:] = False
    obs = env.observation_manager.compute()
    trace = {key: [] for key in ('root_pose', 'joint_pos', 'joint_vel', 'velocity_b', 'yaw_rate', 'upright', 'calf_contacts')}
    if args.motion_audit:
        from isaaclab.utils.math import quat_apply_inverse
        foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        foot_ids = robot.find_bodies(foot_names, preserve_order=True)[0]
        contact_ids = sensor.find_bodies(foot_names, preserve_order=True)[0]
        trace.update({key: [] for key in ('joint_target', 'computed_torque', 'applied_torque', 'foot_clearance',
                                          'foot_z_body', 'air_time', 'contact_time', 'last_air_time',
                                          'first_contact', 'raw_action', 'weighted_rewards')})
        (args.output / 'motion_metadata.json').write_text(json.dumps({
            'joint_names': robot.joint_names, 'foot_names': foot_names,
            'action_joint_names': env.action_manager.get_term('joint_pos')._joint_names,
            'reward_names': env.reward_manager.active_terms,
            'hard_limits': robot.data.joint_pos_limits[0].cpu().tolist(),
            'soft_limits': robot.data.soft_joint_pos_limits[0].cpu().tolist(),
            'effort_limits': robot.data.joint_effort_limits.cpu().tolist(),
            'stiffness': robot.data.joint_stiffness.cpu().tolist(),
            'damping': robot.data.joint_damping.cpu().tolist(),
            'torque_note': 'Implicit-actuator computed/applied torques are PD estimates, not direct solver torque measurements.',
        }, indent=2) + '\n')

    def capture():
        values = {'root_pose': robot.data.root_link_pose_w, 'joint_pos': robot.data.joint_pos,
                  'joint_vel': robot.data.joint_vel, 'velocity_b': robot.data.root_lin_vel_b,
                  'yaw_rate': robot.data.root_ang_vel_b[:, 2], 'upright': -robot.data.projected_gravity_b[:, 2],
                  'calf_contacts': (sensor.data.net_forces_w_history[:, :, calf_ids].norm(dim=-1).amax(1) > 1.).sum(1)}
        if args.motion_audit:
            position = robot.data.body_link_pos_w[:, foot_ids]
            hits, normals, valid = measure_terrain_surface(env, position)
            assert valid.all()
            body = quat_apply_inverse(robot.data.root_link_quat_w[:, None, :].expand(-1, 4, -1),
                                      position - robot.data.root_link_pos_w[:, None, :])
            values.update(joint_target=robot.data.joint_pos_target, computed_torque=robot.data.computed_torque,
                          applied_torque=robot.data.applied_torque,
                          foot_clearance=((position - hits) * normals).sum(-1) - .012,
                          foot_z_body=body[..., 2], air_time=sensor.data.current_air_time[:, contact_ids],
                          contact_time=sensor.data.current_contact_time[:, contact_ids],
                          last_air_time=sensor.data.last_air_time[:, contact_ids],
                          first_contact=sensor.compute_first_contact(env.step_dt)[:, contact_ids],
                          raw_action=env.action_manager.action,
                          weighted_rewards=env.reward_manager._step_reward)
        for key, value in values.items():
            trace[key].append(value.detach().cpu().numpy().copy())

    capture()
    with torch.no_grad():
        for step in range(round(args.seconds / env.step_dt)):
            obs, reward, terminated, truncated, _ = env.step(actor(obs, stochastic_output=False))
            assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
            assert not (terminated | truncated).any()
            capture()
            if (step + 1) % 100 == 0:
                print('EVALUATE_STEPS', step + 1, flush=True)
    trace = {key: np.stack(values) for key, values in trace.items()}
    origins = terrain.env_origins.cpu().numpy()
    np.savez_compressed(args.output / 'rollout.npz', **trace, origins=origins,
                        terrain_origins=terrain.terrain_origins.cpu().numpy())
    start = round(2. / env.step_dt)
    candidates = []
    for index in range(env.num_envs):
        velocity = trace['velocity_b'][start:, index]
        rmse = float(np.sqrt(np.mean((velocity[:, 0] - .5)**2 + velocity[:, 1]**2)))
        upright = float(np.mean(trace['upright'][start:, index] > math.cos(math.radians(30))))
        yaw_rmse = float(np.sqrt(np.mean(trace['yaw_rate'][start:, index]**2)))
        delta = trace['root_pose'][-1, index, :3] - trace['root_pose'][0, index, :3]
        # Exclude trajectories leaving their assigned 8 m terrain tile.
        inside = bool((np.abs(trace['root_pose'][:, index, :2] - origins[index, :2]) < 3.95).all())
        candidates.append(dict(environment=index, terrain=TERRAINS[index // 4], label=LABELS[index // 4],
                               level=int(levels[index]), velocity_rmse_m_s=rmse, upright_fraction=upright,
                               yaw_rate_rmse_rad_s=yaw_rmse, mean_vx_m_s=float(velocity[:, 0].mean()),
                               final_dx_m=float(delta[0]), final_dy_m=float(delta[1]), inside_tile=inside,
                               score=rmse + 2. * (1. - upright) + .15 * yaw_rmse))
    eligible = [x for x in candidates if x['inside_tile']]
    assert len(eligible) >= 2
    # Select relatively good/bad examples, without calling a weak best sample successful.
    best = min(eligible, key=lambda x: x['score'])
    worst = max(eligible, key=lambda x: x['score'])
    assert best['environment'] != worst['environment']
    manifest = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    model_iteration=saved['iter'], config_source=inspect.getfile(TQBot2StandardWalkRoughEnvCfg),
                    evaluation_device=env.device, control_dt=env.step_dt, seconds=args.seconds, command=[.5, 0., 0.],
                    seed=42, candidates=candidates, selected={'better': best, 'worse': worst},
                    diagnostic_kp_scale=args.kp_scale, diagnostic_effort_scale=args.effort_scale,
                    criteria='Among 24 controlled rollouts, minimize/maximize XY velocity RMSE + 2*(1-upright fraction) + 0.15*yaw-rate RMSE after 2 seconds; exclude tile exits.',
                    evaluation='Deterministic actor mean, upright nominal reset, 1 second zero-command settling, fixed walking command, no observation noise or external pushes; training mass/COM/material/gain randomization retained.',
                    recording='Render exact saved GPU root/joint trajectories; 25 fps, real-time playback. CPU renderer does not rerun or integrate physics.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    dump_yaml(str(args.output / 'evaluation_env.yaml'), cfg)
    print('SELECTED', json.dumps(manifest['selected'], ensure_ascii=False), flush=True)


def replay_unloaded(env):
    """Replay recorded actions with a fixed, elevated trunk and gravity disabled.

    The policy is not queried outside its training distribution. A separate
    bounded 1 Hz calf target checks that the actuator can flex and extend freely.
    """
    source = np.load(args.replay_source / 'rollout.npz')
    metadata = json.loads((args.replay_source / 'motion_metadata.json').read_text())
    terrain = env.scene.terrain
    levels = torch.tensor([0, 0, 3, 3] * 6, device=env.device)
    terrain.terrain_levels[:] = levels
    terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
    env.scene.env_origins[:] = terrain.env_origins
    env.reset(seed=42)
    robot, sensor = env.scene['robot'], env.scene['contact_forces']
    assert robot.joint_names == metadata['joint_names']
    action_term = env.action_manager.get_term('joint_pos')
    assert action_term._joint_names == metadata['action_joint_names']
    action_joint_ids = [robot.joint_names.index(name) for name in action_term._joint_names]
    feet = sensor.find_bodies(metadata['foot_names'], preserve_order=True)[0]
    audit = {'joint_names': robot.joint_names, 'hard_limits': robot.data.joint_pos_limits[0].cpu().tolist(),
             'stiffness': robot.data.joint_stiffness.cpu().tolist(),
             'damping': robot.data.joint_damping.cpu().tolist(),
             'effort_limits': robot.data.joint_effort_limits.cpu().tolist(),
             'source': str(args.replay_source), 'start_source_step': 100,
             'physics_dt': env.physics_dt, 'control_dt': env.step_dt,
             'intervention': 'Fix trunk at spawn height 2m, disable gravity, replay identical recorded policy actions; retain self-collisions and joint limits. No policy inference in free air.'}
    q = torch.as_tensor(source['joint_pos'][100], device=env.device)
    dq = torch.as_tensor(source['joint_vel'][100], device=env.device)
    robot.write_joint_state_to_sim(q, dq)
    env.sim.forward()
    env.scene.update(env.physics_dt)
    trace = {key: [] for key in ('joint_pos', 'joint_target', 'joint_vel', 'foot_force', 'all_body_force', 'root_pos', 'estimated_torque')}
    def capture():
        values = {'joint_pos': robot.data.joint_pos, 'joint_target': robot.data.joint_pos_target,
                  'joint_vel': robot.data.joint_vel, 'root_pos': robot.data.root_link_pos_w,
                  'estimated_torque': robot.data.computed_torque,
                  'foot_force': sensor.data.net_forces_w[:, feet].norm(dim=-1),
                  'all_body_force': sensor.data.net_forces_w.norm(dim=-1).amax(-1)}
        for name, value in values.items():
            trace[name].append(value.detach().cpu().numpy().copy())
    for index in range(101, len(source['raw_action'])):
        env.step(torch.as_tensor(source['raw_action'][index], device=env.device))
        capture()
        if index % 100 == 0:
            print('UNLOADED_REPLAY_STEP', index, flush=True)
    np.savez_compressed(args.output / 'unloaded_replay.npz', **{key: np.stack(value) for key, value in trace.items()})
    # Stay away from all joint limits for the separate, deliberately feasible probe.
    for key in trace:
        trace[key] = []
    nominal = robot.data.default_joint_pos.clone()
    calf_ids = robot.find_joints('.*_calf_joint')[0]
    nominal[:, calf_ids] = -1.9
    robot.write_joint_state_to_sim(nominal, torch.zeros_like(nominal))
    env.sim.forward()
    env.scene.update(env.physics_dt)
    for step in range(200):
        target = nominal.clone()
        target[:, calf_ids] = -1.9 + .3 * math.sin(2 * math.pi * step * env.step_dt)
        raw = (target[:, action_joint_ids] - action_term._offset) / action_term._scale
        env.step(raw)
        capture()
    np.savez_compressed(args.output / 'unloaded_sine.npz', **{key: np.stack(value) for key, value in trace.items()})
    (args.output / 'probe_metadata.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('UNLOADED_PROBES_COMPLETE', flush=True)


def render(env):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from pxr import UsdGeom

    manifest = json.loads((args.output / 'manifest.json').read_text())
    trace = np.load(args.output / 'rollout.npz')
    np.testing.assert_allclose(env.scene.terrain.terrain_origins.cpu().numpy(), trace['terrain_origins'], atol=1e-6)
    robot = env.scene['robot']
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title_font = ImageFont.truetype(font_path, 25)
    font = ImageFont.truetype(font_path, 21)
    for kind, selected in manifest['selected'].items():
        index = selected['environment']
        ids = torch.tensor([index], device=env.device)
        for i in range(env.num_envs):
            visible = UsdGeom.Imageable(robot.stage.GetPrimAtPath(f'/World/envs/env_{i}'))
            visible.MakeVisible() if i == index else visible.MakeInvisible()
        filename = f'tqbot2_{kind}_model{manifest["model_iteration"]}.mp4'
        path = args.output / filename
        print('RENDER_START', filename, flush=True)
        with imageio.get_writer(str(path), fps=25, codec='libx264', quality=8, macro_block_size=16,
                               ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart']) as writer:
            for frame_index, step in enumerate(range(2, len(trace['root_pose']), 2)):
                pose = torch.as_tensor(trace['root_pose'][step, index:index+1], device=env.device)
                joint = torch.as_tensor(trace['joint_pos'][step, index:index+1], device=env.device)
                velocity = torch.as_tensor(trace['joint_vel'][step, index:index+1], device=env.device)
                robot.write_root_pose_to_sim(pose, env_ids=ids)
                robot.write_joint_state_to_sim(joint, velocity, env_ids=ids)
                env.sim.forward()
                env.scene.update(env.physics_dt)
                position = trace['root_pose'][step, index, :3]
                env.sim.set_camera_view(eye=position + np.array([1.25, -1.75, .95]),
                                        target=position + np.array([.12, 0., -.06]))
                if frame_index == 0:
                    for _ in range(10):
                        env.render()
                frame = env.render()
                assert frame is not None and frame.any(), 'Empty renderer frame'
                canvas = Image.fromarray(frame)
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, 960, 112), fill=(17, 25, 36))
                color = (115, 224, 187) if kind == 'better' else (255, 175, 120)
                label = '相对较好' if kind == 'better' else '相对较差'
                draw.text((18, 6), f'TQBot2 · model_{manifest["model_iteration"]} · {label}', font=title_font, fill=color)
                draw.text((18, 43), f'{selected["label"]} / {selected["level"]} 级   指令 0.50 m/s   原速 · GPU仿真轨迹回放', font=font, fill='white')
                vx = trace['velocity_b'][step, index, 0]
                dx = position[0] - trace['root_pose'][0, index, 0]
                lean = math.degrees(math.acos(float(np.clip(trace['upright'][step, index], -1., 1.))))
                draw.text((18, 77), f't={step * manifest["control_dt"]:.2f}s   实际 vx={vx:+.2f} m/s   前进 {dx:+.2f} m   机身倾角 {lean:.0f}°', font=font, fill=(199, 211, 224))
                writer.append_data(np.asarray(canvas))
                if frame_index in (0, 49, 99, 149, 199):
                    canvas.save(args.output / f'{kind}_frame_{frame_index:03d}.png')
                if (frame_index + 1) % 50 == 0:
                    print('RENDER_FRAMES', kind, frame_index + 1, flush=True)
        selected['video'] = filename
        print('RENDER_DONE', filename, flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def main():
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    env = gym.make('RobotLab-Isaac-Velocity-Rough-TQBot2-StandardWalk-v0', cfg=cfg,
                   render_mode='rgb_array' if args.mode == 'render' else None).unwrapped
    try:
        if args.mode == 'evaluate':
            evaluate(env, cfg)
        elif args.mode == 'replay-unloaded':
            replay_unloaded(env)
        else:
            render(env)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
