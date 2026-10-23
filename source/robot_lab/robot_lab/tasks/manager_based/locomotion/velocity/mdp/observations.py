# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class JointPositionRelativeToReference(ManagerTermBase):
    """Joint positions relative to an explicit, named policy reference.

    Resolve names once in the observation's own joint order. This does not
    modify default_joint_pos, which remains available for resets and rewards.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        ids = asset_cfg.joint_ids
        names = asset.joint_names[ids] if isinstance(ids, slice) else [asset.joint_names[i] for i in ids]
        indices, _, values = resolve_matching_names_values(cfg.params["reference_joint_pos"], names)
        if len(indices) != len(names):
            raise ValueError("The policy reference must cover every observed joint.")
        self._reference = torch.empty(len(names), device=env.device, dtype=asset.data.joint_pos.dtype)
        self._reference[indices] = torch.tensor(values, device=env.device, dtype=self._reference.dtype)

    def __call__(
        self, env: ManagerBasedEnv, reference_joint_pos: dict[str, float], asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids] - self._reference


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor
