"""Custom action terms for locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RecoveryJointPositionAction(JointPositionAction):
    """Blend walking joint targets with soft-limit targets when the robot falls.

    Walking targets use the regular affine action mapping. Recovery targets map
    normalized actions in ``[-1, 1]`` to the articulation's soft joint limits.
    A posture-dependent factor transitions continuously between both mappings.
    """

    cfg: RecoveryJointPositionActionCfg

    def __init__(self, cfg: RecoveryJointPositionActionCfg, env: ManagerBasedEnv):
        if cfg.fallen_gravity_z <= cfg.upright_gravity_z:
            raise ValueError(
                "fallen_gravity_z must be greater than upright_gravity_z, "
                f"got {cfg.fallen_gravity_z} <= {cfg.upright_gravity_z}."
            )
        if cfg.activation_rate <= 0.0 or cfg.deactivation_rate <= 0.0:
            raise ValueError("Recovery alpha activation/deactivation rates must be positive.")

        super().__init__(cfg, env)

        self._recovery_alpha = torch.zeros(self.num_envs, device=self.device)
        self._walking_zero_targets = torch.zeros_like(self._raw_actions) + self._offset
        self._previous_joint_targets = self._walking_zero_targets.clone()
        self._processed_actions[:] = self._walking_zero_targets

    @property
    def recovery_alpha(self) -> torch.Tensor:
        """Per-environment blend factor: zero is walking and one is recovery."""
        return self._recovery_alpha

    @property
    def previous_joint_targets(self) -> torch.Tensor:
        """Joint targets produced during the previous environment step."""
        return self._previous_joint_targets

    def _compute_target_alpha(self) -> torch.Tensor:
        """Compute a smooth posture score from projected-gravity z."""
        gravity_z = self._asset.data.projected_gravity_b[:, 2]
        posture_score = (gravity_z - self.cfg.upright_gravity_z) / (
            self.cfg.fallen_gravity_z - self.cfg.upright_gravity_z
        )
        posture_score = posture_score.clamp(0.0, 1.0)
        return posture_score.square() * (3.0 - 2.0 * posture_score)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._previous_joint_targets[:] = self._processed_actions

        # An upright robot has projected-gravity z close to -1. Smoothstep
        # avoids a derivative discontinuity at the two posture thresholds.
        target_alpha = self._compute_target_alpha()

        # Rate limiting prevents the action mapping from changing instantly.
        alpha_error = target_alpha - self._recovery_alpha
        activation_step = self.cfg.activation_rate * self._env.step_dt
        deactivation_step = self.cfg.deactivation_rate * self._env.step_dt
        alpha_step = torch.where(
            alpha_error >= 0.0,
            alpha_error.clamp(max=activation_step),
            alpha_error.clamp(min=-deactivation_step),
        )
        self._recovery_alpha.add_(alpha_step).clamp_(0.0, 1.0)

        walking_targets = self._raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            walking_targets = torch.clamp(
                walking_targets, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        lower_limits = soft_limits[..., 0]
        upper_limits = soft_limits[..., 1]
        normalized_actions = self._raw_actions.clamp(-1.0, 1.0)
        recovery_targets = lower_limits + 0.5 * (normalized_actions + 1.0) * (
            upper_limits - lower_limits
        )

        alpha = self._recovery_alpha.unsqueeze(-1)
        self._processed_actions[:] = torch.lerp(walking_targets, recovery_targets, alpha)
        self._processed_actions.clamp_(min=lower_limits, max=upper_limits)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        # Reset events have already written the new root pose at this point.
        # Initialize fallen environments in recovery mode immediately instead
        # of spending the first 0.2 seconds widening their action range.
        self._recovery_alpha[env_ids] = self._compute_target_alpha()[env_ids]
        self._processed_actions[env_ids] = self._walking_zero_targets[env_ids]
        self._previous_joint_targets[env_ids] = self._processed_actions[env_ids]


@configclass
class RecoveryJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`RecoveryJointPositionAction`."""

    class_type: type[ActionTerm] = RecoveryJointPositionAction

    upright_gravity_z: float = -0.9
    """At or below this gravity-z value, use the walking mapping."""

    fallen_gravity_z: float = -0.4
    """At or above this gravity-z value, request the recovery mapping."""

    activation_rate: float = 5.0
    """Maximum alpha increase per second (zero to one in 0.2 seconds)."""

    deactivation_rate: float = 2.0
    """Maximum alpha decrease per second (one to zero in 0.5 seconds)."""
