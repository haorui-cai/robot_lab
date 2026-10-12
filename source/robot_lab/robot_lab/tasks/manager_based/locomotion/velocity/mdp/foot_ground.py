"""TQBot2 foot-sphere motion against actual terrain, including slopes."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.warp import raycast_mesh

from .foot_motion import foot_motion_costs, foot_point_state


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
        terrain = env.scene.terrain
        if terrain.cfg.terrain_type == "plane":
            hits = position.clone()
            hits[..., 2] = 0.0  # TerrainImporter planes are created at world z=0.
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
