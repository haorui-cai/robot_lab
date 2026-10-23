"""Sample consistent root-link and whole-robot CoM vertical kinematics at physics rate."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--commands", type=float, nargs="+", default=[-6.0, -10.0, -14.0])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import csv
import importlib.metadata
import json
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robot_lab.tasks  # noqa: F401

task = "RobotLab-Isaac-Roll-LeftThigh-Slope-TQBot2-v0"
cfg = parse_env_cfg(task, device=args.device, num_envs=len(args.commands))
cfg.seed = 42
cfg.commands.base_velocity.debug_vis = False
cfg.commands.base_velocity.roll_range = (-10.0, -10.0)
cfg.observations.policy.enable_corruption = False
agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
raw = env.unwrapped
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
try:
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=raw.device)
    robot = raw.scene["robot"]
    sensor = raw.scene["contact_forces"]
    assert robot.body_names[0] == "trunk"
    # The Isaac adapter removes the virtual ROS base, leaving trunk as root.
    # Recover the actual xacro base point from the source fixed transform.
    ros_urdf = Path(robot.cfg.spawn.asset_path).with_name("TQBot2.urdf")
    joint = ET.parse(ros_urdf).getroot().find("joint[@name='floating_base']")
    assert joint.find("parent").get("link") == "base" and joint.find("child").get("link") == "trunk"
    origin = joint.find("origin")
    translation = torch.tensor([float(x) for x in origin.get("xyz").split()], device=raw.device).reshape(1, 3)
    rpy = torch.tensor([float(x) for x in origin.get("rpy", "0 0 0").split()], device=raw.device)
    rotation = quat_from_euler_xyz(*rpy.unbind()).reshape(1, 4)
    base_offset = -quat_apply_inverse(rotation, translation)
    masses = robot.root_physx_view.get_masses().to(raw.device)
    weights = masses / masses.sum(dim=-1, keepdim=True)
    left_ids = sensor.find_bodies("(FL|RL)_(hip|thigh)")[0]
    right_ids = sensor.find_bodies("(FR|RR)_hip")[0]
    targets = torch.tensor(args.commands, device=raw.device)
    command = raw.command_manager.get_command("base_velocity")
    command[:, 0] = targets
    command[:, 1] = 0
    obs = env.get_observations()
    episode = np.zeros(raw.num_envs, dtype=int)
    physics_step = 0
    rows = []
    columns = ["t_s", "env", "episode", "cmd_roll_rad_s", "slope_deg", "ground_origin_z_m",
               "base_link_z_m", "root_com_z_m", "whole_com_z_m", "base_link_vz_m_s",
               "root_com_vz_m_s", "whole_com_vz_m_s", "roll_rate_rad_s", "yaw_rate_rad_s",
               "left_Fz_N", "right_Fz_N", "total_Fz_N", "max_contact_force_N",
               "xacro_base_z_m", "xacro_base_vz_m_s"]

    def sample():
        data = robot.data
        force = sensor.data.net_forces_w
        offset_w = quat_apply(data.root_link_quat_w, base_offset.expand(raw.num_envs, -1))
        base_pos_w = data.root_link_pos_w + offset_w
        base_vel_w = data.root_link_lin_vel_w + torch.linalg.cross(data.root_ang_vel_w, offset_w, dim=-1)
        values = torch.stack((command[:, 0], raw.scene.terrain.slope_degrees, raw.scene.env_origins[:, 2],
                              data.root_link_pos_w[:, 2], data.root_com_pos_w[:, 2],
                              (data.body_com_pos_w[:, :, 2] * weights).sum(-1),
                              data.root_link_lin_vel_w[:, 2], data.root_com_lin_vel_w[:, 2],
                              (data.body_com_lin_vel_w[:, :, 2] * weights).sum(-1),
                              data.root_ang_vel_b[:, 0], data.root_ang_vel_w[:, 2],
                              force[:, left_ids, 2].sum(-1), force[:, right_ids, 2].sum(-1),
                              force[:, :, 2].sum(-1), force.norm(dim=-1).amax(-1),
                              base_pos_w[:, 2], base_vel_w[:, 2]), -1).cpu().numpy()
        for index, value in enumerate(values):
            rows.append([physics_step * raw.physics_dt, index, int(episode[index]), *value.tolist()])

    sample()
    original_update = raw.scene.update

    def update_and_sample(dt):
        # Sample after each physics substep, before episode reset.
        global physics_step
        original_update(dt)
        if dt > 0:
            physics_step += 1
            sample()

    raw.scene.update = update_and_sample
    with torch.inference_mode():
        for _ in range(round(args.seconds / raw.step_dt)):
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            episode += dones.cpu().numpy().astype(int)
            if dones.any():
                command[:, 0] = targets
                command[:, 1] = 0
                obs = env.get_observations()
    raw.scene.update = original_update
    array = np.asarray(rows, dtype=np.float64)
    derived = np.full((len(array), 8), np.nan)
    # Differentiate matching point velocities; never bridge reset discontinuities.
    for index in range(raw.num_envs):
        for ep in np.unique(array[array[:, 1] == index, 2]):
            ids = np.flatnonzero((array[:, 1] == index) & (array[:, 2] == ep))
            for col, (velocity_col, stride) in enumerate((
                (9, 1), (10, 1), (11, 1), (9, cfg.decimation), (10, cfg.decimation),
                (11, cfg.decimation), (19, 1), (19, cfg.decimation)
            )):
                derived[ids[stride:], col] = (
                    array[ids[stride:], velocity_col] - array[ids[:-stride], velocity_col]
                ) / (array[ids[stride:], 0] - array[ids[:-stride], 0])
    columns += ["base_link_az_m_s2", "root_com_az_m_s2", "whole_com_az_m_s2",
                "base_link_az_20ms_m_s2", "root_com_az_20ms_m_s2", "whole_com_az_20ms_m_s2",
                "xacro_base_az_m_s2", "xacro_base_az_20ms_m_s2"]
    with (output / "vertical_motion.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(np.concatenate((array, derived), -1))
    metadata = dict(checkpoint=str(Path(args.checkpoint).resolve()), task=task, seed=42,
                    commands=args.commands, physics_dt=raw.physics_dt, control_dt=raw.step_dt,
                    duration_s=args.seconds, sampling="after each physics substep, before reset",
                    contact_threshold_N=5.0, observation_noise=False, domain_randomization="same as training",
                    initial_phase_half_width=0.1, initial_speed_scale=1.0, slope_degrees=0.0,
                    active_rewards={name:raw.reward_manager.get_term_cfg(name).weight for name in raw.reward_manager.active_terms},
                    body_names=robot.body_names, body_masses_kg=masses.cpu().tolist(),
                    root_com_offset_in_trunk_m=robot.data.body_com_pos_b[:, 0].cpu().tolist(),
                    xacro_base_offset_in_trunk_m=base_offset[0].cpu().tolist(), source_ros_urdf=str(ros_urdf),
                    z_definition="world vertical; legacy base_link_* columns are Isaac root=trunk; xacro_base_* is the ROS rolling-shell-center frame; whole_com is articulated robot CoM",
                    acceleration_definition="backward difference of matching world-point velocity; first sample per episode NaN")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"VERTICAL_DIAGNOSTIC_OK rows={len(rows)} physics_steps={physics_step} output={output}", flush=True)
finally:
    env.close()
    app.close()
