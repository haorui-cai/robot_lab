"""Train/evaluate a frozen historical TQBot2 policy on physical planar ramps."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--num-envs', type=int, default=1024)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--replicates', type=int, default=8)
parser.add_argument('--angles', type=int, nargs='+', default=[-30, 0, 30])
parser.add_argument('--resume', action='store_true')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.source / 'source/robot_lab'))
app = AppLauncher(args).app

import importlib.metadata
import itertools
import math
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.warp import raycast_mesh
from isaaclab.sensors import RayCaster
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.models import MLPModel
import robot_lab.tasks
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.tqbot2.agents.rsl_rl_ppo_cfg import TQBot2RoughPPORunnerCfg
from slope_env import configuration, state, axes
from geometry import summarize
import pyramid_train


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify_geometry(env):
    """Ray-check collision surfaces, not just the requested terrain parameter."""
    terrain = env.scene.terrain
    centers = terrain.terrain_origins.flatten(0, 1)
    points = centers[:, None, :] + torch.tensor([[-2., 0., 3.], [0., 0., 3.], [2., 0., 3.]], device=env.device)
    direction = torch.zeros_like(points)
    direction[..., 2] = -1.
    path = env.scene['height_scanner'].cfg.mesh_prim_paths[0]
    hits, _, normals, _ = raycast_mesh(points, direction, RayCaster.meshes[path], max_dist=6., return_normal=True)
    assert torch.isfinite(hits).all()
    measured = torch.rad2deg(torch.atan2(hits[:, 2, 2]-hits[:, 0, 2], hits[:, 2, 0]-hits[:, 0, 0]))
    expected = torch.tensor([level*5*sign for level in range(7) for sign in [0, 0, 1, 1, 1, 1, -1, -1, -1, -1]], device=env.device)
    assert torch.max(torch.abs(measured-expected)) < .02, (measured, expected)
    return {'max_angle_error_deg': float((measured-expected).abs().max()),
            'angles_deg': measured.reshape(7, 10).cpu().tolist(), 'ray_samples': int(hits.numel()/3)}


def set_eval_material(env, cases):
    robot = env.scene['robot']
    view = robot.root_physx_view
    materials = view.get_material_properties().clone()
    # Same coefficient on all robot shapes makes the ground-contact condition unambiguous.
    for i, case in enumerate(cases):
        materials[i, :, 0:2] = case[2]
        materials[i, :, 2] = 0.
    view.set_material_properties(materials, torch.arange(env.num_envs, dtype=torch.int32))
    actual = view.get_material_properties()
    expected = torch.tensor([c[2] for c in cases], device=actual.device)
    assert torch.allclose(actual[:, :, 0], expected[:, None].expand_as(actual[:, :, 0]), atol=1e-6)
    state(env)['friction'] = expected.to(env.device)


def evaluate():
    cases = [case for case in itertools.product(args.angles, [.3, .45, .6], [.7, .85, 1.]) for _ in range(args.replicates)]
    cfg = configuration(len(cases), args.device, args.seed)
    env = ManagerBasedRLEnv(cfg)
    s = state(env)
    s['fixed_cases'] = torch.tensor(cases, device=env.device)
    obs, _ = env.reset(seed=args.seed)
    s['completed'].clear()
    set_eval_material(env, cases)
    geometry = verify_geometry(env)
    assert obs['policy'].shape[-1] == 45
    actor = MLPModel(obs, {'actor': ['policy']}, 'actor', 12, hidden_dims=[512, 256, 128],
                     activation='elu', obs_normalization=False,
                     distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar'}).to(env.device)
    saved = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    actor.eval()
    command = env.command_manager.get_term('base_velocity')
    speeds = torch.tensor([c[1] for c in cases], device=env.device)
    trace = {k: [] for k in ['root_pose', 'joint_pos', 'joint_vel']}
    trace_ids = torch.tensor([i*args.replicates for i in range(len(cases)//args.replicates)], device=env.device)
    with torch.inference_mode():
        for step in range(605):
            command.vel_command_b[:] = 0.
            command.vel_command_b[:, 0] = speeds * (s['steps']*env.step_dt >= 1.)
            command.is_heading_env[:] = command.is_standing_env[:] = False
            obs = env.observation_manager.compute()
            obs, reward, terminated, timeout, extras = env.step(actor(obs, stochastic_output=False))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            if step % 2 == 0:
                robot = env.scene['robot']
                for key, tensor in [('root_pose', robot.data.root_link_pose_w), ('joint_pos', robot.data.joint_pos), ('joint_vel', robot.data.joint_vel)]:
                    trace[key].append(tensor[trace_ids].cpu().numpy().copy())
            if step % 100 == 0:
                print('EVAL_PROGRESS', step, len(s['completed']), '/', len(cases), flush=True)
            if len(s['completed']) == len(cases):
                break
    if len(s['completed']) != len(cases):
        raise RuntimeError('Incomplete evaluation; missing episodes cannot be silently excluded')
    rows = [s['completed'][i] for i in range(len(cases))]
    for i, row in enumerate(rows):
        row.update(command_m_s=cases[i][1], seed=args.seed)
    summary = summarize(rows)
    summary.update(checkpoint=str(args.checkpoint.resolve()), seed=args.seed, geometry=geometry,
                   checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   sampling='deterministic actor mean; training observation noise and mass/COM/gain randomization retained',
                   source=str(args.source.resolve()), protocol='12 s trial; first 1 s zero-command settling; all failures counted')
    write_json(args.output/'episodes.json', rows)
    write_json(args.output/'summary.json', summary)
    np.savez_compressed(args.output/'rollout.npz', **{k: np.stack(v) for k, v in trace.items()},
                        environment_ids=trace_ids.cpu().numpy(), cases=np.array(cases))
    dump_yaml(str(args.output/'env.yaml'), cfg)
    print('EVAL_RESULT', json.dumps(summary), flush=True)
    env.close()


def train():
    cfg = pyramid_train.configuration(args.num_envs, args.device, args.seed)
    cfg.log_dir = str(args.output)
    env = ManagerBasedRLEnv(cfg)
    env.reset(seed=args.seed)
    geometry = {'training_terrain_types': list(cfg.scene.terrain.terrain_generator.sub_terrains),
                'source_generators': 'original HfPyramidSlopedTerrainCfg, normal and inverted',
                'nominal_target_slope_deg': 30., 'terrain_statistics': pyramid_train.statistics(env)}
    agent = TQBot2RoughPPORunnerCfg()
    agent.seed, agent.device = args.seed, args.device
    agent.algorithm.learning_rate = .0001
    agent.algorithm.schedule = 'fixed'
    agent.algorithm.entropy_coef = .005
    agent.save_interval = 100
    agent.max_iterations = args.iterations
    agent.check_for_nan = True
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(wrapper, agent.to_dict(), log_dir=str(args.output), device=args.device)
    if args.resume:
        info = runner.load(str(args.checkpoint)) or {}
        runner.current_learning_iteration += 1  # RSL-RL stores the last completed zero-based iteration.
        if 'course' in info:
            pyramid_train.restore_course(env, info['course'])
        env.reset()
    else:
        runner.load(str(args.checkpoint), load_cfg={'actor': True, 'critic': True})
        assert runner.current_learning_iteration == 0
    start = runner.current_learning_iteration
    # Preserve curriculum state at every checkpoint, not only at clean exit.
    original_save = runner.save
    def save(path, infos=None):
        original_save(path, {'course': pyramid_train.course(env),
                             'parent': str(args.checkpoint), 'specialization': 'walk_slope_30'})
    runner.save = save
    dump_yaml(str(args.output/'env.yaml'), cfg)
    dump_yaml(str(args.output/'agent.yaml'), agent)
    write_json(args.output/'started.json', {'source': str(args.source.resolve()),
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'start_iteration': start, 'requested_iterations': args.iterations,
        'resume_optimizer': args.resume, 'geometry': geometry})
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    checkpoint = args.output / f'model_{runner.current_learning_iteration}.pt'
    assert checkpoint.is_file() and runner.current_learning_iteration == start+args.iterations-1
    write_json(args.output/'completed.json', {'status': 'completed', 'checkpoint': str(checkpoint.resolve()),
        'start_iteration': start, 'last_iteration': runner.current_learning_iteration,
        'iterations_completed': args.iterations, 'course': pyramid_train.statistics(env)})
    env.close()


try:
    evaluate() if args.mode == 'evaluate' else train()
finally:
    app.close()
