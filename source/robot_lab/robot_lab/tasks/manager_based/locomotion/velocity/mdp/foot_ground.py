"""TQBot2 foot-sphere motion against actual terrain, including slopes."""

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.warp import raycast_mesh

from .foot_motion import foot_motion_costs, foot_point_state, support_foot_slip


def terrain_surface_at_feet(env, position):
    """Sample the static training terrain beneath each spherical foot."""
    if env.scene.terrain.cfg.terrain_type == "plane":
        hits = position.clone()
        hits[..., 2] = 0.0
        normal = torch.zeros_like(position)
        normal[..., 2] = 1.0
        valid = torch.ones_like(position[..., 0], dtype=torch.bool)
    else:
        path = env.scene["height_scanner"].cfg.mesh_prim_paths[0]
        mesh = RayCaster.meshes[path]
        starts = position.clone()
        starts[..., 2] += 1.0
        directions = torch.zeros_like(starts)
        directions[..., 2] = -1.0
        hits, _, normal, _ = raycast_mesh(starts, directions, mesh, max_dist=5., return_normal=True)
        valid = torch.isfinite(hits).all(dim=-1) & torch.isfinite(normal).all(dim=-1)
        hits = torch.where(valid[..., None], hits, position)
        normal = torch.where(valid[..., None], normal, directions.neg())
        normal *= torch.where(normal[..., 2:3] < 0., -1., 1.)
    return hits, normal, valid


class SupportFootSlip(ManagerTermBase):
    """Contact-gated foot surface slip in the tangent plane of static terrain."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg, sensor_cfg = cfg.params["asset_cfg"], cfg.params["sensor_cfg"]
        self.robot = env.scene[asset_cfg.name]
        sensor = env.scene[sensor_cfg.name]
        names = [self.robot.body_names[i] for i in asset_cfg.body_ids]
        if names != [sensor.body_names[i] for i in sensor_cfg.body_ids]:
            raise ValueError("Foot/contact ordering mismatch")
        if not sensor.cfg.track_air_time:
            raise ValueError("Support foot slip requires contact phase tracking")

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                 foot_radius: float = 0.012, contact_clearance: float = 0.02):
        data, ids = self.robot.data, asset_cfg.body_ids
        position = data.body_link_pos_w[:, ids]
        hits, normal, valid = terrain_surface_at_feet(env, position)
        clearance = ((position - hits) * normal).sum(dim=-1) - foot_radius
        sensor = env.scene[sensor_cfg.name]
        support = (sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0)
        support &= valid & (clearance < contact_clearance)
        slip = support_foot_slip(data.body_link_lin_vel_w[:, ids], data.body_link_ang_vel_w[:, ids],
                                 normal, support, foot_radius)
        # Keep the original reward's upright gate and sum-over-feet scaling.
        return slip * torch.clamp(-data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


class SwingFootClearance(ManagerTermBase):
    """Dimensionless swing-foot clearance shortfall against the local terrain.

    A foot link is the centre of its spherical collision shape. Its clearance is
    the normal distance to the surface under that centre, minus the sphere radius.
    Penalize only clearance below ``target_clearance``. Divide that shortfall
    by ``error_scale`` in metres before squaring; at or above the target the
    cost is zero.
    This is local clearance, not a look-ahead collision check at obstacle edges.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg, sensor_cfg = cfg.params["asset_cfg"], cfg.params["sensor_cfg"]
        self.robot = env.scene[asset_cfg.name]
        sensor = env.scene[sensor_cfg.name]

        def selected_names(names, ids):
            return names[ids] if isinstance(ids, slice) else [names[i] for i in ids]

        names = selected_names(self.robot.body_names, asset_cfg.body_ids)
        sensor_names = selected_names(sensor.body_names, sensor_cfg.body_ids)
        if not names or names != sensor_names:
            raise ValueError(f"Foot/contact ordering mismatch: {names} vs {sensor_names}")
        if not sensor.cfg.track_air_time:
            raise ValueError("Swing foot clearance requires contact phase tracking")
        for name, default, allow_zero in (("target_clearance", 0.05, False),
                                          ("error_scale", 0.05, False),
                                          ("tanh_mult", 2.0, False),
                                          ("foot_radius", 0.012, True)):
            value = cfg.params.get(name, default)
            if not math.isfinite(value) or value < 0.0 or (value == 0.0 and not allow_zero):
                raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")

    def __call__(self, env, command_name: str, asset_cfg: SceneEntityCfg,
                 sensor_cfg: SceneEntityCfg, target_clearance: float = 0.05,
                 tanh_mult: float = 2.0, foot_radius: float = 0.012,
                 error_scale: float = 0.05):
        data, ids = self.robot.data, asset_cfg.body_ids
        position = data.body_link_pos_w[:, ids]
        hits, normal, valid = terrain_surface_at_feet(env, position)
        clearance = ((position - hits) * normal).sum(dim=-1) - foot_radius
        relative_velocity = data.body_link_lin_vel_w[:, ids] - data.root_link_lin_vel_w[:, None, :]
        tangent_velocity = relative_velocity - (relative_velocity * normal).sum(dim=-1, keepdim=True) * normal
        speed_weight = torch.tanh(tanh_mult * tangent_velocity.norm(dim=-1))
        sensor = env.scene[sensor_cfg.name]
        swinging = sensor.data.current_air_time[:, sensor_cfg.body_ids] > 0.0
        shortfall = (target_clearance - clearance).clamp_min(0.0) / error_scale
        cost = torch.where(valid & swinging, shortfall.square() * speed_weight, 0.0)
        active = env.command_manager.get_command(command_name).norm(dim=-1) > 0.1
        upright = torch.clamp(-data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return cost.sum(dim=-1) * active * upright


class FootGroundMotion(ManagerTermBase):
    """Use the fixed URDF foot offset, rather than the calf origin or COM velocity."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        sensor_cfg = cfg.params["sensor_cfg"]
        self.robot = env.scene[asset_cfg.name]
        sensor = env.scene[sensor_cfg.name]
        robot_names = [self.robot.body_names[i] for i in asset_cfg.body_ids]
        sensor_names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
        if robot_names != sensor_names:
            raise ValueError(f"Foot/contact ordering mismatch: {robot_names} vs {sensor_names}")

    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                 mode: str, command_name: str, foot_offset: tuple, foot_radius: float,
                 target_clearance: float = 0.03):
        data = self.robot.data
        ids = asset_cfg.body_ids
        position, velocity = foot_point_state(
            data.body_link_pos_w[:, ids], data.body_link_quat_w[:, ids],
            data.body_link_lin_vel_w[:, ids], data.body_link_ang_vel_w[:, ids], foot_offset,
        )
        hits, normal, valid = terrain_surface_at_feet(env, position)
        clearance = ((position - hits) * normal).sum(dim=-1) - foot_radius
        forces = env.scene[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        contact = forces.norm(dim=-1).amax(dim=1) > 1.0
        slip, swing = foot_motion_costs(clearance, velocity, data.root_link_lin_vel_w, normal,
                                        contact, valid, target_clearance=target_clearance)
        # Fade out while recovering; allow ordinary walking on a 30-degree slope.
        upright = ((-data.projected_gravity_b[:, 2] - 0.5) / 0.35).clamp(0., 1.)
        if mode == "slip":
            return slip * upright
        if mode == "clearance":
            command = env.command_manager.get_command(command_name)
            active = command[:, :2].norm(dim=1) + 0.25 * command[:, 2].abs() > 0.1
            return swing * active * upright
        raise ValueError(f"Unknown foot motion mode: {mode}")
