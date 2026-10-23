"""Blind actor inputs, physical box terrain, resets, and episode accounting."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import torch

from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from .blind_box_geometry import (
    BOX_FRONT, BOX_HEIGHTS, BOX_LENGTHS, BoxCourse, build_box_mesh, feet_on_top, stable_landing, stable_top,
)


class BlindBoxTerrain(TerrainImporter):
    def import_ground_plane(self, name, size=None):
        mesh, origins = build_box_mesh()
        self.box_origins = torch.tensor(origins, device=self.device, dtype=torch.float32)
        self.import_mesh(name, mesh)

    def configure_env_origins(self, origins=None):
        self.terrain_origins = None
        self.course = BoxCourse()
        self.levels = torch.zeros(self.cfg.num_envs, device=self.device, dtype=torch.long)
        self.kinds = torch.arange(self.cfg.num_envs, device=self.device) % len(BOX_LENGTHS)
        # Spread robots within each shared, collision-filtered patch.
        self.lateral_offsets = torch.linspace(-6.0, 6.0, self.cfg.num_envs, device=self.device)
        self.env_origins = self.box_origins[self.kinds].clone()
        self.env_origins[:, 1] += self.lateral_offsets
        self.heights = torch.full((self.cfg.num_envs,), BOX_HEIGHTS[0], device=self.device)
        self.lengths = torch.tensor(BOX_LENGTHS, device=self.device)[self.kinds]

    def assign(self, env_ids, levels, kinds):
        self.levels[env_ids], self.kinds[env_ids] = levels, kinds
        self.env_origins[env_ids] = self.box_origins[levels * len(BOX_LENGTHS) + kinds]
        self.env_origins[env_ids, 1] += self.lateral_offsets[env_ids]
        self.heights[env_ids] = torch.tensor(BOX_HEIGHTS, device=self.device)[levels]
        self.lengths[env_ids] = torch.tensor(BOX_LENGTHS, device=self.device)[kinds]


def state(env):
    if not hasattr(env, "_blind_box"):
        robot = env.scene["robot"]
        names = [f"{leg}_calf" for leg in ("FR", "FL", "RR", "RL")]
        body_ids = robot.find_bodies(names, preserve_order=True)[0]
        sensor_ids = env.scene["contact_forces"].find_bodies(names, preserve_order=True)[0]
        xml = ET.parse(robot.cfg.spawn.asset_path).getroot()
        offsets, radii = [], []
        for leg in ("FR", "FL", "RR", "RL"):
            joint = xml.find(f"joint[@name='{leg}_foot_fixed']")
            if joint.find("origin").get("rpy", "0 0 0") != "0 0 0":
                raise ValueError("Blind-box foot geometry expects untranslated foot orientation")
            offsets.append([float(x) for x in joint.find("origin").get("xyz").split()])
            radii.append(float(xml.find(f"link[@name='{leg}_foot']/collision/geometry/sphere").get("radius")))
        env._blind_box = SimpleNamespace(
            body_ids=body_ids, sensor_ids=sensor_ids,
            offsets=torch.tensor(offsets, device=env.device), radii=torch.tensor(radii, device=env.device),
            dwell=torch.zeros(env.num_envs, device=env.device),
            stop_dwell=torch.zeros(env.num_envs, device=env.device),
            top_contact_age=torch.full((env.num_envs, 4), 1e6, device=env.device),
            stopped=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
            last_stopped=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
            succeeded=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
            progress=torch.zeros(env.num_envs, device=env.device),
            previous_distance=torch.zeros(env.num_envs, device=env.device),
            last_step=-1,
        )
        resume = env.cfg.box_curriculum_resume_path
        if resume:
            data = json.loads(Path(resume).read_text())
            for key in vars(env.scene.terrain.course):
                setattr(env.scene.terrain.course, key, data["course"][key])
    return env._blind_box


def foot_positions(env):
    s, robot = state(env), env.scene["robot"]
    offsets = s.offsets.expand(env.num_envs, -1, -1)
    points = robot.data.body_pos_w[:, s.body_ids] + quat_apply(
        robot.data.body_quat_w[:, s.body_ids].reshape(-1, 4), offsets.reshape(-1, 3)
    ).reshape(env.num_envs, 4, 3)
    return points - env.scene.env_origins[:, None, :]


def goal_distance(env):
    terrain, robot = env.scene.terrain, env.scene["robot"]
    relative = robot.data.root_pos_w - env.scene.env_origins
    goal = torch.stack((BOX_FRONT + terrain.lengths / 2, torch.zeros_like(terrain.heights)), dim=-1)
    return torch.linalg.vector_norm(relative[:, :2] - goal, dim=-1)


def reset_box(env, env_ids):
    terrain, robot, s = env.scene.terrain, env.scene["robot"], state(env)
    n = len(env_ids)
    height = env.cfg.box_evaluation_height
    if height is None:
        levels = torch.full((n,), terrain.course.level, device=env.device, dtype=torch.long)
        replay = torch.rand(n, device=env.device) < 0.25
        levels[replay] = torch.randint(terrain.course.level + 1, (int(replay.sum()),), device=env.device)
    else:
        if not any(abs(height - h) < 1e-6 for h in BOX_HEIGHTS):
            raise ValueError(f"Evaluation height must be one of {BOX_HEIGHTS}")
        levels = torch.full((n,), min(range(len(BOX_HEIGHTS)), key=lambda i: abs(BOX_HEIGHTS[i] - height)),
                            device=env.device, dtype=torch.long)
    kinds = torch.randint(len(BOX_LENGTHS), (n,), device=env.device)
    if env.cfg.box_evaluation_length is not None:
        kinds[:] = BOX_LENGTHS.index(env.cfg.box_evaluation_length)
    terrain.assign(env_ids, levels, kinds)
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += terrain.env_origins[env_ids]
    root[:, 0] += torch.empty(n, device=env.device).uniform_(-0.15, 0.35)
    root[:, 1] += torch.empty(n, device=env.device).uniform_(-0.08, 0.08)
    r = torch.empty(n, device=env.device).uniform_(-0.04, 0.04)
    p = torch.empty(n, device=env.device).uniform_(-0.04, 0.04)
    y = torch.empty(n, device=env.device).uniform_(-0.08, 0.08)
    root[:, 3:7] = quat_from_euler_xyz(r, p, y)
    root[:, 7:] = 0
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    robot.set_joint_position_target(robot.data.joint_pos[env_ids], env_ids=env_ids)
    s.dwell[env_ids] = 0
    s.stop_dwell[env_ids] = 0
    s.top_contact_age[env_ids] = 1e6
    s.stopped[env_ids] = False
    s.succeeded[env_ids] = False
    s.progress[env_ids] = 0
    relative = root[:, :2] - terrain.env_origins[env_ids, :2]
    goal = torch.stack((BOX_FRONT + terrain.lengths[env_ids] / 2, torch.zeros(n, device=env.device)), dim=-1)
    s.previous_distance[env_ids] = torch.linalg.vector_norm(relative - goal, dim=-1)


def joint_tracking_error(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    return robot.data.joint_pos_target[:, asset_cfg.joint_ids] - robot.data.joint_pos[:, asset_cfg.joint_ids]


def box_geometry_observation(env):
    """Privileged critic-only geometry; never exposed to the actor."""
    terrain = env.scene.terrain
    relative = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    return torch.stack((BOX_FRONT - relative[:, 0], BOX_FRONT + terrain.lengths - relative[:, 0],
                        terrain.heights, relative[:, 1], relative[:, 2] - terrain.heights), dim=-1)


def box_success(env):
    s = state(env)
    if s.last_step != env.common_step_counter:
        robot, terrain = env.scene["robot"], env.scene.terrain
        feet = foot_positions(env)
        forces = env.scene["contact_forces"].data.net_forces_w[:, s.sensor_ids]
        top_contacts = feet_on_top(feet, terrain.heights, terrain.lengths, s.radii) & (forces[..., 2] > 2.0)
        s.top_contact_age.copy_(torch.where(top_contacts, 0.0, s.top_contact_age + env.step_dt))
        relative = robot.data.root_pos_w - env.scene.env_origins
        stable = stable_landing(
            feet, terrain.heights, terrain.lengths,
            robot.data.projected_gravity_b, robot.data.root_lin_vel_w, robot.data.root_ang_vel_w,
            relative, s.top_contact_age,
        )
        stable &= ~box_failure(env)
        s.dwell.copy_(torch.where(stable, s.dwell + env.step_dt, 0.0))
        s.succeeded.copy_((s.dwell >= 0.5) & ((forces[..., 2] > 2.0).sum(dim=1) >= 2))
        stopped = stable_top(feet, terrain.heights, terrain.lengths, s.radii,
                             robot.data.projected_gravity_b, robot.data.root_lin_vel_w, robot.data.root_ang_vel_w,
                             relative, forces) & ~box_failure(env)
        s.stop_dwell.copy_(torch.where(stopped, s.stop_dwell + env.step_dt, 0.0))
        s.stopped.copy_(s.stop_dwell >= 0.5)
        distance = goal_distance(env)
        s.progress.copy_((s.previous_distance - distance) / env.step_dt)
        s.previous_distance.copy_(distance)
        s.last_step = env.common_step_counter
    return s.succeeded


def box_failure(env):
    robot, terrain = env.scene["robot"], env.scene.terrain
    relative = robot.data.root_pos_w - env.scene.env_origins
    return (
        (robot.data.projected_gravity_b[:, 2] > -0.1)
        | (relative[:, 2] < 0.12)
        | (relative[:, 1].abs() > 0.65)
        | (relative[:, 0] < -0.7)
        | (relative[:, 0] > BOX_FRONT + terrain.lengths + 0.3)
    )


def progress_reward(env):
    return state(env).progress


def top_support_reward(env):
    s, terrain = state(env), env.scene.terrain
    on_top = feet_on_top(foot_positions(env), terrain.heights, terrain.lengths, s.radii)
    contact = env.scene["contact_forces"].data.net_forces_w[:, s.sensor_ids, 2] > 2.0
    upright = (-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0, 1)
    return (on_top & contact).float().mean(dim=1) * upright


def goal_reward(env):
    upright = (-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0, 1)
    return torch.exp(-goal_distance(env).square() / 0.25) * upright


def success_reward(env):
    return state(env).succeeded.float()


def failure_reward(env):
    return box_failure(env).float()


def box_curriculum(env, env_ids):
    terrain, s = env.scene.terrain, state(env)
    ids = env_ids[(env.episode_length_buf[env_ids] > 0) & env.termination_manager.dones[env_ids]]
    s.last_stopped[ids] = s.stopped[ids]
    frontier = ids[terrain.levels[ids] == terrain.course.level]
    if env.cfg.box_evaluation_height is None and len(frontier):
        terrain.course.observe(len(frontier), int(s.succeeded[frontier].sum()))
    if len(ids) and env.cfg.log_dir and env.cfg.box_evaluation_height is None:
        path = Path(env.cfg.log_dir) / "box_curriculum.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"course": vars(terrain.course), "heights_m": BOX_HEIGHTS}, indent=2) + "\n")
    return {"max_height_m": BOX_HEIGHTS[terrain.course.level], "frontier_success_rate": terrain.course.last_rate}
