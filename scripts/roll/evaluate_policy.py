"""Frozen-checkpoint Isaac evaluation with full-cycle and equal-episode accounting.

The JSON plan specifies jobs (checkpoint, label, sampling, profile, seed). No
optimizer or training runner is constructed. Profiles freeze curricula. The
reference environment is the current narrow-DR left-thigh task, with B rewards
and yaw weight 10 for every policy; task-success metrics are reward-independent.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--num-envs', type=int, default=128)
parser.add_argument('--episodes-per-env', type=int, default=2)
parser.add_argument('--max-steps', type=int, default=1100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import hashlib
import json
import time
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.models import MLPModel
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg
import robot_lab.tasks
from robot_lab.tasks.manager_based.locomotion.roll.mdp import hip_only_cycle as cycle
from robot_lab.tasks.manager_based.locomotion.roll.mdp import slopes
from robot_lab.tasks.manager_based.locomotion.roll.mdp.base_kinematics import TQBOT2_BASE_OFFSET, base_point_velocity
from evaluation_metrics import summarize, stratify

torch.set_num_threads(4)
OUT = Path(args.output).resolve()
OUT.mkdir(parents=True,exist_ok=True)
jobs = json.loads(Path(args.plan).read_text())
TASK = 'RobotLab-Isaac-Roll-LeftThigh-Slope-TQBot2-v0'
original_update = cycle.update_cycle
original_reset = slopes.reset_slope_rolling
original_resample = cycle.RollYawCommand._resample_command
current = {}


def reset_profile(env, env_ids, height_table_name='left_thigh_roll_spawn_heights.json'):
    level = cycle.course(env)
    profile = current['job']['profile']
    level.initial_speed_scales = (0.,) if profile == 'unassisted' else (1.,)
    level.assist_level = level.phase_level = level.yaw_level = 0
    level.phase_widths = (np.pi,) if profile == 'unassisted' else (.1,)
    level.yaw_widths = (0.,)
    terrain = env.scene.terrain
    terrain.slope_course.levels = {'uphill':1,'downhill':1} if profile == 'current' else {'uphill':0,'downhill':0}
    if profile != 'slope_probe':
        original_reset(env,env_ids,height_table_name)
    else:
        values = torch.tensor([-30.,-20.,-10.,0.,5.,10.],device=env.device)
        angles = values[env_ids % len(values)]
        terrain.assign_slopes(env_ids,angles)
        cycle.reset_rolling_phase(env,env_ids,height_table_name,terrain_roll=torch.deg2rad(angles))


def resample(self, env_ids):
    original_resample(self,env_ids)
    if current['job']['profile'] == 'yaw_probe':
        values = torch.tensor([-.5,-.25,-.1,0.,.1,.25,.5],device=self.device)
        self._command[env_ids,0] = -10.
        self._command[env_ids,1] = values[env_ids % len(values)]
    if current['job']['profile'] == 'slope_probe':
        self._command[env_ids,0] = -10.


def record_reset(env, env_ids):
    if 'counts' not in current or not hasattr(env,'_hip_episode_time'):
        return 0.
    st = cycle.tracker(env)
    ids = env_ids[env._hip_episode_time[env_ids] > 0]
    cmd = env.command_manager.get_command('base_velocity')
    for i in ids.cpu().tolist():
        if current['counts'][i] < args.episodes_per_env:
            n = current['counts'][i]+1
            values = dict(seed=current['job']['seed'],env=i,episode=int(n),seconds=float(env._hip_episode_time[i]),
                          cycles=int(st.cycles[i]),roll_good=int(st.roll_good[i]),yaw_good=int(st.yaw_good[i]),
                          roll_command=float(cmd[i,0]),yaw_command=float(cmd[i,1]),slope=float(env.scene.terrain.slope_degrees[i]),
                          yaw_error_sum=float(current['yaw_error_sum'][i]),
                          terminations=[k for k in env.termination_manager.active_terms if bool(env.termination_manager.get_term(k)[i])],
                          reward_terms={k:float(v[i]) for k,v in env.reward_manager._episode_sums.items()})
            current['episodes'].append(values)
        current['counts'][i] += 1
        current['yaw_error_sum'][i] = 0
    return 0.


def record_update(env):
    st = original_update(env)
    if 'last_clock' not in current or current['last_clock'] == env.common_step_counter:
        return st
    current['last_clock'] = env.common_step_counter
    active = torch.as_tensor(current['counts'] < args.episodes_per_env,device=env.device)
    finished = (st.cycles > current['last_cycles']) & active
    cmd = env.command_manager.get_command('base_velocity')
    for i in finished.nonzero().flatten().cpu().tolist():
        error = abs(float(st.last_yaw[i]-cmd[i,1]))
        current['yaw_error_sum'][i] += error
        current['cycles'].append(dict(seed=current['job']['seed'],env=i,episode=int(current['counts'][i]+1),
                                     time=env.common_step_counter*env.step_dt,roll_rate=float(st.last_roll[i]),
                                     yaw_rate=float(st.last_yaw[i]),roll_command=float(cmd[i,0]),yaw_command=float(cmd[i,1]),
                                     slope=float(env.scene.terrain.slope_degrees[i])))
    current['last_cycles'].copy_(st.cycles)
    data = env.scene['robot'].data
    offset = torch.tensor(TQBOT2_BASE_OFFSET,device=env.device).expand(env.num_envs,-1)
    vel = base_point_velocity(data.root_link_lin_vel_w,data.root_ang_vel_w,data.root_link_quat_w,offset)
    valid = current['prev_valid'] & active
    az = ((vel-current['prev_vel'])/env.step_dt)[:,2]
    for key, value in [('base_az2',az[valid].square()),('power', (data.applied_torque*data.joint_vel).abs().sum(-1)[active])]:
        current[key+'_sum'] += float(value.sum())
        current[key+'_count'] += int(value.numel())
    current['prev_vel'].copy_(vel)
    current['prev_valid'][:] = True
    if env.common_step_counter % 5 == 0:
        pos = data.root_link_pos_w + quat_apply(data.root_link_quat_w,offset)
        for i in range(min(8,env.num_envs)):
            if active[i]:
                current['traces'].append([env.common_step_counter*env.step_dt,i,int(current['counts'][i]+1),
                                          float(cmd[i,0]),float(data.root_ang_vel_b[i,0]),float(cmd[i,1]),
                                          float(data.root_ang_vel_w[i,2]),float(pos[i,2]-env.scene.env_origins[i,2])])
    return st


cfg = parse_env_cfg(TASK,device=args.device,num_envs=args.num_envs)
# Allocation capacities only: sufficient for a small evaluation batch, without
# reserving the original 4096-environment GPU contact buffers. Overflow invalidates a run.
if args.num_envs <= 256:
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
cfg.commands.base_velocity.debug_vis = False
cfg.commands.base_velocity.legacy_reward_fraction = 0.
cfg.rewards.yaw_cycle_tracking.weight = 10.
cfg.curriculum.rolling = CurriculumTermCfg(func=record_reset)
cfg.curriculum.slopes = None
cfg.events.randomize_reset_base.func = reset_profile
cfg.log_dir = None
cfg.observations.policy.enable_corruption = True
cfg.seed = jobs[0]['seed']
original_com_randomizer = cfg.events.randomize_com_positions.func


def randomize_com_from_nominal(env, env_ids, com_range, asset_cfg):
    asset = env.scene[asset_cfg.name]
    if not hasattr(env,'_audit_nominal_com'):
        env._audit_nominal_com = asset.root_physx_view.get_coms().clone()
    indices = torch.arange(env.num_envs,device='cpu') if env_ids is None else env_ids.cpu()
    asset.root_physx_view.set_coms(env._audit_nominal_com,indices)
    original_com_randomizer(env,env_ids,com_range=com_range,asset_cfg=asset_cfg)


cfg.events.randomize_com_positions.func = randomize_com_from_nominal
current['job'] = jobs[0]
cycle.RollYawCommand._resample_command = resample
cycle.update_cycle = record_update
env = gym.make(TASK,cfg=cfg)
raw = env.unwrapped
dump_yaml(str(OUT/'evaluation_env.yaml'),cfg)
try:
    for job in jobs:
        destination = OUT/job['name']
        destination.mkdir(exist_ok=True)
        if (destination/'summary.json').exists():
            print('SKIP_COMPLETED',job['name'],flush=True)
            continue
        current.clear()
        current.update(job=job, counts=np.zeros(args.num_envs,dtype=int),episodes=[],cycles=[],traces=[],
                       yaw_error_sum=np.zeros(args.num_envs),last_cycles=torch.zeros(args.num_envs,device=raw.device),
                       last_clock=-1, prev_valid=torch.zeros(args.num_envs,dtype=torch.bool,device=raw.device),
                       prev_vel=torch.zeros(args.num_envs,3,device=raw.device),base_az2_sum=0.,base_az2_count=0,power_sum=0.,power_count=0)
        raw.seed(job['seed'])
        # Re-sample startup physical properties for each seed/policy comparison.
        raw.event_manager.apply(mode='startup')
        # Avoid counting the previous job's partial episodes in the reset hook.
        if hasattr(raw,'_hip_episode_time'):
            raw._hip_episode_time[:] = 0
        obs,_ = env.reset(seed=job['seed'])
        checkpoint = Path(job['checkpoint']).resolve()
        saved = torch.load(checkpoint,map_location=raw.device,weights_only=False)
        state = saved['actor_state_dict']
        # Explicitly reject incompatible normalizers/architectures instead of silently approximating inference.
        actor = MLPModel(obs,{'actor':['policy']},'actor',6,hidden_dims=[512,256,128],activation='elu',
                         obs_normalization=False,distribution_cfg={'class_name':'GaussianDistribution','init_std':.8,'std_type':'scalar'}).to(raw.device)
        actor.load_state_dict(state,strict=True)
        actor.eval()
        sat = torch.zeros(6,device=raw.device)
        mean_sat = torch.zeros(6,device=raw.device)
        positive = torch.zeros(6,device=raw.device)
        raw_square = torch.zeros(6,device=raw.device)
        action_count = 0
        start = time.monotonic()
        print('EVALUATING',job,flush=True)
        with torch.inference_mode():
            for step in range(args.max_steps):
                active = torch.as_tensor(current['counts'] < args.episodes_per_env,device=raw.device)
                mean = actor(obs)
                action = actor(obs,stochastic_output=True) if job['sampling'] == 'gaussian' else mean
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite policy actions')
                sat += (action[active].abs() >= 1).sum(0)
                mean_sat += (mean[active].abs() >= 1).sum(0)
                positive += (action[active] >= 1).sum(0)
                raw_square += mean[active].square().sum(0)
                action_count += int(active.sum())
                obs,_,done,timeout,_ = env.step(action.clamp(-1,1))
                reset = done | timeout
                current['last_cycles'][reset] = 0
                current['prev_valid'][reset] = False
                if (step+1) % 250 == 0:
                    print('PROGRESS',job['name'],step+1,len(current['episodes']),f'{time.monotonic()-start:.1f}s',flush=True)
                if np.all(current['counts'] >= args.episodes_per_env):
                    break
        # Keep exactly the requested completed episodes per env. Unfinished episodes never count as passes.
        epkeys = {(r['env'],r['episode']) for r in current['episodes']}
        circles = [r for r in current['cycles'] if (r['env'],r['episode']) in epkeys]
        summary = summarize(current['episodes'],circles)
        summary.update(job=job,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       policy_std=state['distribution.std_param'].cpu().tolist(),
                       executed_action_saturation=(sat/max(action_count,1)).cpu().tolist(),
                       policy_mean_saturation=(mean_sat/max(action_count,1)).cpu().tolist(),
                       positive_limit_fraction=(positive/max(action_count,1)).cpu().tolist(),
                       policy_mean_rms=torch.sqrt(raw_square/max(action_count,1)).cpu().tolist(),
                       base_world_z_acceleration_rms=(current['base_az2_sum']/max(current['base_az2_count'],1))**.5,
                       mean_absolute_joint_power_W=current['power_sum']/max(current['power_count'],1),
                       wall_seconds=time.monotonic()-start,steps=step+1,device=str(raw.device),
                       expected_episodes=args.num_envs*args.episodes_per_env,
                       complete=bool(np.all(current['counts'] >= args.episodes_per_env)),
                       by_roll_command=stratify(current['episodes'],circles,'roll_command',[6,8,10,12,14]),
                       by_yaw_command=stratify(current['episodes'],circles,'yaw_command'),
                       by_slope=stratify(current['episodes'],circles,'slope'))
        for filename,content in [('summary.json',summary),('episodes.json',current['episodes']),('cycles.json',circles)]:
            (destination/filename).write_text(json.dumps(content,indent=2,allow_nan=False))
        np.savetxt(destination/'traces.csv',np.asarray(current['traces']).reshape(-1,8),delimiter=',',
                   header='time,env,episode,roll_command,roll_instant,yaw_command,yaw_instant,base_z_relative_origin',comments='')
        robot = raw.scene['robot']
        (destination/'physics.json').write_text(json.dumps({'body_names':robot.body_names,
          'mass':robot.root_physx_view.get_masses().cpu().tolist(),'com':robot.root_physx_view.get_coms().cpu().tolist()},indent=2))
        print('EVALUATION_DONE',job['name'],{k:summary[k] for k in ['episodes','roll_success','yaw_success','joint_success','cycle_yaw_mae','wall_seconds']},flush=True)
finally:
    cycle.update_cycle = original_update
    cycle.RollYawCommand._resample_command = original_resample
    env.close()
    app.close()
