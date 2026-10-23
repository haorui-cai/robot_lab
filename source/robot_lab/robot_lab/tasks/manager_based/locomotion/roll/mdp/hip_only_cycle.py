"""Commands, reset curriculum and cycle-averaged tracking for hip-only rolling."""

import hashlib
import json
import math
from pathlib import Path

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

from .cycle_tracking import CycleTracker, RollingCourse

ROLL_RADIUS = 0.112
INITIAL_ROLL_RATE = -0.8 / ROLL_RADIUS


def course(env):
    if not hasattr(env, "_hip_roll_course"):
        env._hip_roll_course = RollingCourse(env.cfg.commands.base_velocity.initial_speed_scales)
    return env._hip_roll_course


def tracker(env):
    if not hasattr(env, "_hip_cycle_tracker"):
        cfg = env.cfg.commands.base_velocity
        env._hip_cycle_tracker = CycleTracker(
            env.num_envs, env.device, actual_cycle_rewards=cfg.actual_cycle_rewards,
            history_capacity=math.ceil(env.cfg.episode_length_s / env.step_dt) + 2,
            yaw_min_speed=cfg.yaw_reward_min_speed,
            yaw_tracking_std=cfg.yaw_tracking_std,
            yaw_success_tolerance=cfg.yaw_success_tolerance,
            legacy_reward_fraction=cfg.legacy_reward_fraction,
        )
        env._hip_cycle_step = env.common_step_counter
        env._hip_episode_time = torch.zeros(env.num_envs, device=env.device)
    return env._hip_cycle_tracker


def update_cycle(env):
    state = tracker(env)
    if env._hip_cycle_step != env.common_step_counter:
        robot = env.scene["robot"]
        state.update(robot.data.root_ang_vel_b[:, 0], robot.data.root_ang_vel_w[:, 2],
                     env.command_manager.get_command("base_velocity"), env.step_dt)
        env._hip_episode_time += env.step_dt
        env._hip_cycle_step = env.common_step_counter
    return state


class RollYawCommand(CommandTerm):
    """Two angular commands: signed body-x roll rate and world-z yaw rate."""

    def __init__(self, cfg, env):
        self.robot = env.scene["robot"]
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["cycle_roll_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cycle_yaw_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        return self._command

    def _resample_command(self, env_ids):
        self._command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.roll_range)
        state = course(self._env)
        width = state.yaw_widths[state.yaw_level]
        self._command[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(-width, width)

    def _update_command(self):
        pass

    def _update_metrics(self):
        state = tracker(self._env)
        valid = state.cycles > 0
        self.metrics["cycle_roll_error"][:] = torch.where(valid, (state.last_roll - self.command[:, 0]).abs(), 0.0)
        self.metrics["cycle_yaw_error"][:] = torch.where(valid, (state.last_yaw - self.command[:, 1]).abs(), 0.0)

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis and not hasattr(self, "arrows"):
            self.arrows = {}
            for name, cfg in (("goal", GREEN_ARROW_X_MARKER_CFG), ("actual", BLUE_ARROW_X_MARKER_CFG)):
                for axis in ("roll", "yaw"):
                    self.arrows[name, axis] = VisualizationMarkers(cfg.replace(prim_path=f"/Visuals/RollCommand/{name}_{axis}"))
        if hasattr(self, "arrows"):
            for marker in self.arrows.values():
                marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized or not hasattr(self, "_command"):
            return
        zero = torch.zeros(self.num_envs, device=self.device)
        for name in ("goal", "actual"):
            rates = self.command if name == "goal" else torch.stack(
                (self.robot.data.root_ang_vel_b[:, 0], self.robot.data.root_ang_vel_w[:, 2]), dim=-1)
            for column, axis in enumerate(("roll", "yaw")):
                position = self.robot.data.root_pos_w.clone()
                position[:, 2] += 0.45 if axis == "roll" else 0.75
                value = rates[:, column]
                if axis == "roll":
                    direction = quat_from_euler_xyz(zero, zero, torch.where(value < 0, math.pi, 0.0))
                    direction = quat_mul(self.robot.data.root_quat_w, direction)
                else:
                    direction = quat_from_euler_xyz(zero, torch.where(value < 0, math.pi / 2, -math.pi / 2), zero)
                scale = torch.full((self.num_envs, 3), 0.10, device=self.device)
                scale[:, 0] = value.abs() * (0.06 if axis == "roll" else 0.6)
                self.arrows[name, axis].visualize(position, direction, scale)


@configclass
class RollYawCommandCfg(CommandTermCfg):
    class_type = RollYawCommand
    resampling_time_range = (1000.0, 1000.0)  # Hold each target for the entire episode.
    debug_vis = True
    roll_range = (-1.0 / ROLL_RADIUS, -0.7 / ROLL_RADIUS)
    initial_speed_scales: tuple[float, ...] = (1.0,)
    actual_cycle_rewards: bool = False
    yaw_reward_min_speed: float | None = None
    yaw_tracking_std: float = 0.25
    yaw_success_tolerance: float = 0.15
    legacy_reward_fraction: float = 0.0


def reset_rolling_phase(env, env_ids, height_table_name: str = "roll_spawn_heights.json", terrain_roll=None):
    """Sample the current phase curriculum, with matched initial roll/translation."""
    robot = env.scene["robot"]
    level = course(env)
    width = level.phase_widths[level.phase_level]
    roll = torch.empty(len(env_ids), device=env.device).uniform_(-2.0 - width, -2.0 + width)
    roll = (roll + math.pi) % (2 * math.pi) - math.pi
    if not hasattr(env, "_roll_height_table"):
        model = Path(ISAACLAB_ASSETS_DATA_DIR) / "Robots/TQBot/TQBot2_description"
        data = json.loads((model / height_table_name).read_text())
        actual_hash = hashlib.sha256(Path(robot.cfg.spawn.asset_path).read_bytes()).hexdigest()
        if actual_hash != data["urdf_sha256"]:
            raise ValueError("Roll spawn height table is stale; run build_tqbot2_roll_spawn_heights.py")
        defaults = data.get("joint_angles_rad", {
            name: (1.52 if "R_hip" in name else -1.52) for name in robot.joint_names
        })
        expected = torch.tensor([defaults[name] for name in robot.joint_names], device=env.device)
        if not torch.allclose(robot.data.default_joint_pos[0], expected):
            raise ValueError("Roll spawn height table does not match the default joint pose")
        env._roll_height_table = torch.tensor(data["heights_m"], device=env.device)
    table = env._roll_height_table
    index = (roll + math.pi) / (2 * math.pi) * (len(table) - 1)
    lower = index.long().clamp(0, len(table) - 2)
    height = torch.lerp(table[lower], table[lower + 1], index - lower)
    zero = torch.zeros_like(roll)
    slope = zero if terrain_roll is None else terrain_roll
    positions = env.scene.env_origins[env_ids].clone()
    positions[:, 1] -= height * slope.sin()
    positions[:, 2] += height * slope.cos()
    orientation = quat_from_euler_xyz(roll + slope, zero, zero)
    velocity = torch.zeros(len(env_ids), 6, device=env.device)
    # Negative x rotation rolls the approximate cylinder in positive world y.
    velocity[:, 1] = 0.8 * level.initial_speed_scale * slope.cos()
    velocity[:, 2] = 0.8 * level.initial_speed_scale * slope.sin()
    velocity[:, 3] = INITIAL_ROLL_RATE * level.initial_speed_scale
    robot.write_root_pose_to_sim(torch.cat((positions, orientation), dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)
    tracker(env).reset(env_ids, initial_roll_rate=velocity[:, 3], initial_yaw_rate=velocity[:, 5])
    env._hip_episode_time[env_ids] = 0


def cycle_roll_reward(env):
    return update_cycle(env).roll_payment


def cycle_yaw_reward(env):
    return update_cycle(env).yaw_payment


def roll_angular_acceleration_l2(env):
    """Control-step body-x rate derivative, excluding reset discontinuities."""
    return update_cycle(env).roll_acceleration.square()


def yaw_angular_acceleration_l2(env):
    """Control-step world-z rate derivative, excluding reset discontinuities."""
    return update_cycle(env).yaw_acceleration.square()


def cycle_observation(env):
    state = tracker(env)
    return torch.stack((state.window.mean[:, 0] / 10, state.window.mean[:, 1],
                        state.progress / (2 * math.pi), state.elapsed / 3), dim=-1).clamp(-10, 10)


def support_observation(env, left_bodies: str = "(FL|RL)_hip", right_bodies: str = "(FR|RR)_hip"):
    sensor = env.scene["contact_forces"]
    if not hasattr(env, "_roll_support_ids"):
        env._roll_support_ids = [sensor.find_bodies(names)[0] for names in (left_bodies, right_bodies)]
    return torch.stack([(sensor.data.net_forces_w[:, ids].norm(dim=-1).amax(dim=-1) > 5).float()
                        for ids in env._roll_support_ids], dim=-1)


def cycle_stalled(env):
    state = update_cycle(env)
    return state.elapsed > 3.0  # No completed revolution for 3 seconds.


def invalid_rolling_pose(env):
    robot = env.scene["robot"]
    # Prevent counting longitudinal spins while airborne or with a near-vertical axis.
    return (robot.data.root_pos_w[:, 2] > 0.5) | (robot.data.projected_gravity_b[:, 0].abs() > 0.7)


def rolling_curriculum(env, env_ids):
    state = tracker(env)
    level = course(env)
    valid = env._hip_episode_time[env_ids] > 0
    ids = env_ids[valid]
    if len(ids):
        enough = (env._hip_episode_time[ids] >= 6.0) & (state.cycles[ids] >= 4)
        good_roll = enough & (state.roll_good[ids] / state.cycles[ids].clamp_min(1) >= 0.8)
        good_yaw = state.yaw_good[ids] / state.cycles[ids].clamp_min(1) >= 0.8
        changed = level.observe(len(ids), good_roll.sum().item(), (good_roll & good_yaw).sum().item(),
                                env.common_step_counter)
        if changed and env.cfg.log_dir:
            path = Path(env.cfg.log_dir) / "rolling_curriculum.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(vars(level), indent=2) + "\n")
    return {"phase_half_width": level.phase_widths[level.phase_level],
            "yaw_max": level.yaw_widths[level.yaw_level],
            "phase_level": level.phase_level, "yaw_level": level.yaw_level,
            "assist_level": level.assist_level, "initial_speed_scale": level.initial_speed_scale,
            "initial_roll_rate": INITIAL_ROLL_RATE * level.initial_speed_scale,
            "initial_linear_speed": 0.8 * level.initial_speed_scale}
