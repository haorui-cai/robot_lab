"""Physical ramp terrain, slope-aligned resets and independent slope curricula."""

import json
import math
from pathlib import Path

import torch

from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply_inverse

from . import hip_only_cycle as cycle
from .slope_tracking import SlopeCourse, build_slope_mesh


class RollingSlopeTerrain(TerrainImporter):
    """Use the plane import hook to create a fixed bank of sloped collision meshes."""

    width = 40.0
    length = 60.0

    def import_ground_plane(self, name, size=None):
        mesh, origins, angles = build_slope_mesh(self.width, self.length)
        self.slope_origins = torch.tensor(origins, device=self.device, dtype=torch.float32)
        self.slope_angles_deg = torch.tensor(angles, device=self.device)
        self.import_mesh(name, mesh)

    def configure_env_origins(self, origins=None):
        self.terrain_origins = None
        self.env_origins = self.slope_origins[30].repeat(self.cfg.num_envs, 1)
        self.slope_centers = self.env_origins.clone()
        # Spread thousands of independently filtered robots across each patch,
        # avoiding a pathological broadphase pile-up at one identical point.
        side = math.ceil(math.sqrt(self.cfg.num_envs))
        ids = torch.arange(self.cfg.num_envs, device=self.device)
        self.spawn_offsets = torch.zeros(self.cfg.num_envs, 3, device=self.device)
        if side > 1:
            self.spawn_offsets[:, 0] = ((ids % side) / (side - 1) - 0.5) * 24
            self.spawn_offsets[:, 1] = ((ids // side) / (side - 1) - 0.5) * 16
        self.slope_degrees = torch.zeros(self.cfg.num_envs, device=self.device)
        self.slope_kinds = torch.zeros(self.cfg.num_envs, device=self.device, dtype=torch.long)
        self.slope_levels = torch.zeros_like(self.slope_kinds)
        self.slope_course = SlopeCourse()
        self.assign_slopes(ids, self.slope_degrees.clone())

    def assign_slopes(self, env_ids, angles):
        self.slope_degrees[env_ids] = angles
        self.slope_centers[env_ids] = self.slope_origins[(angles + 30).long()]
        offset = self.spawn_offsets[env_ids].clone()
        offset[:, 2] = offset[:, 1] * torch.deg2rad(angles).tan()
        self.env_origins[env_ids] = self.slope_centers[env_ids] + offset


def reset_slope_rolling(env, env_ids, height_table_name="left_thigh_roll_spawn_heights.json"):
    terrain = env.scene.terrain
    course = terrain.slope_course
    draw = torch.rand(len(env_ids), device=env.device)
    # 20% flat rehearsal, 40% uphill, 40% downhill. At level zero all are flat.
    kinds = torch.where(draw < 0.2, 0, torch.where(draw < 0.6, 1, 2))
    angles = torch.zeros(len(env_ids), device=env.device)
    levels = torch.zeros_like(kinds)
    for kind, name, sign in ((1, "uphill", 1), (2, "downhill", -1)):
        selected = kinds == kind
        # Uniform integer-degree sampling includes zero and the active maximum.
        samples = torch.randint(course.maximum(name) + 1, (len(env_ids),), device=env.device)
        angles[selected] = sign * samples[selected].float()
        levels[selected] = course.levels[name]
    terrain.slope_kinds[env_ids] = kinds
    terrain.slope_levels[env_ids] = levels
    terrain.assign_slopes(env_ids, angles)
    cycle.reset_rolling_phase(env, env_ids, height_table_name, terrain_roll=torch.deg2rad(angles))


def slope_normals(env):
    angle = torch.deg2rad(env.scene.terrain.slope_degrees)
    return torch.stack((torch.zeros_like(angle), -angle.sin(), angle.cos()), dim=-1)


def invalid_slope_pose(env):
    robot = env.scene["robot"]
    normal = slope_normals(env)
    height = ((robot.data.root_pos_w - env.scene.env_origins) * normal).sum(-1)
    axis_tilt = quat_apply_inverse(robot.data.root_quat_w, normal)[:, 0].abs()
    return (height > 0.5) | (height < -0.05) | (axis_tilt > 0.7)


def slope_out_of_bounds(env):
    terrain = env.scene.terrain
    relative = env.scene["robot"].data.root_pos_w - terrain.slope_centers
    return (relative[:, 0].abs() > terrain.width / 2 - 1) | (relative[:, 1].abs() > terrain.length / 2 - 1)


def slope_curriculum(env, env_ids):
    terrain = env.scene.terrain
    course = terrain.slope_course
    state = cycle.tracker(env)
    ids = env_ids[env._hip_episode_time[env_ids] > 0]
    changed = False
    if len(ids):
        enough = (env._hip_episode_time[ids] >= 6) & (state.cycles[ids] >= 4)
        roll_ok = enough & (state.roll_good[ids] / state.cycles[ids].clamp_min(1) >= 0.8)
        yaw_ok = state.yaw_good[ids] / state.cycles[ids].clamp_min(1) >= 0.8
        angle = torch.deg2rad(terrain.slope_degrees[ids])
        relative = env.scene["robot"].data.root_pos_w[ids] - terrain.env_origins[ids]
        travel = relative[:, 1] * angle.cos() + relative[:, 2] * angle.sin()
        target = env.command_manager.get_command("base_velocity")[ids, 0].abs()
        # Spinning while sliding backwards must not qualify as successful climbing.
        forward = travel >= 0.25 * cycle.ROLL_RADIUS * target * env._hip_episode_time[ids]
        for kind, name in ((1, "uphill"), (2, "downhill")):
            frontier = (terrain.slope_kinds[ids] == kind) & (
                terrain.slope_degrees[ids].abs() >= math.ceil(0.75 * course.maximum(name))
            ) & (terrain.slope_levels[ids] == course.levels[name])
            changed |= course.observe(name, frontier.sum().item(), (frontier & roll_ok & forward).sum().item(),
                                      (frontier & roll_ok & yaw_ok & forward).sum().item(), env.common_step_counter)
    if changed and env.cfg.log_dir:
        path = Path(env.cfg.log_dir) / "slope_curriculum.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vars(course), indent=2) + "\n")
    return {"uphill_max_deg": course.maximum("uphill"), "downhill_max_deg": course.maximum("downhill"),
            "uphill_level": course.levels["uphill"], "downhill_level": course.levels["downhill"]}
