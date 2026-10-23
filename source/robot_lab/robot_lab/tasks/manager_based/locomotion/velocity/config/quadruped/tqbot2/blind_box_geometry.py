"""Simulator-independent geometry and success gates for blind box climbing."""

import numpy as np
import torch
import trimesh

BOX_HEIGHTS = tuple(round(0.04 + 0.02 * i, 2) for i in range(14))
BOX_LENGTHS = (1.2, 1.6, 2.0)
BOX_FRONT = 1.0
PATCH_WIDTH = 16.0


def build_box_mesh():
    """A bank of physical boxes; each origin is on the approach ground."""
    meshes, origins = [], []
    for level, height in enumerate(BOX_HEIGHTS):
        for kind, length in enumerate(BOX_LENGTHS):
            origin = np.array([level * 8.0, kind * (PATCH_WIDTH + 2.0), 0.0])
            ground = trimesh.creation.box(extents=(6.0, PATCH_WIDTH, 0.1))
            ground.apply_translation(origin + [1.0, 0.0, -0.05])
            box = trimesh.creation.box(extents=(length, PATCH_WIDTH, height))
            box.apply_translation(origin + [BOX_FRONT + length / 2, 0.0, height / 2])
            meshes.extend((ground, box))
            origins.append(origin)
    return trimesh.util.concatenate(meshes), np.asarray(origins)


def feet_on_top(feet, height, length, radius, margin=0.04, tolerance=0.025):
    """Check actual foot sphere centers, not the merged calf/knee origins."""
    return (
        (feet[..., 0] >= BOX_FRONT + margin)
        & (feet[..., 0] <= BOX_FRONT + length[:, None] - margin)
        & (feet[..., 1].abs() < PATCH_WIDTH / 2 - margin)
        & ((feet[..., 2] - radius[None, :] - height[:, None]).abs() < tolerance)
    )


def stable_top(feet, height, length, radius, gravity, linear_velocity, angular_velocity,
               base_position, contact_forces):
    on_top = feet_on_top(feet, height, length, radius)
    base_height = base_position[:, 2] - height
    return (
        on_top.all(dim=1)
        & ((contact_forces[..., 2] > 2.0).sum(dim=1) >= 3)
        & (gravity[:, 2] < -0.85)
        & (torch.linalg.vector_norm(linear_velocity, dim=1) < 0.25)
        & (torch.linalg.vector_norm(angular_velocity, dim=1) < 0.6)
        & (base_height > 0.20) & (base_height < 0.55)
        & (base_position[:, 0] > BOX_FRONT + 0.15)
        & (base_position[:, 0] < BOX_FRONT + length - 0.15)
    )


def stable_landing(feet, height, length, gravity, linear_velocity, angular_velocity,
                   base_position, top_contact_age):
    """Allow a normal walking gait after every foot has actually landed on top.

    Current swinging feet may be above the surface. Recent top contacts and a
    bounded body motion distinguish walking on top from jumping over the box.
    """
    inside = (
        (feet[..., 0] >= BOX_FRONT + 0.04)
        & (feet[..., 0] <= BOX_FRONT + length[:, None] - 0.04)
        & (feet[..., 1].abs() < PATCH_WIDTH / 2 - 0.04)
    ).all(dim=1)
    base_height = base_position[:, 2] - height
    return (
        inside & (top_contact_age <= 0.6).all(dim=1)
        & (gravity[:, 2] < -0.85)
        & (torch.linalg.vector_norm(linear_velocity[:, :2], dim=1) < 1.0)
        & (linear_velocity[:, 2].abs() < 0.3)
        & (torch.linalg.vector_norm(angular_velocity, dim=1) < 1.2)
        & (base_height > 0.20) & (base_height < 0.55)
        & (base_position[:, 0] > BOX_FRONT + 0.15)
        & (base_position[:, 0] < BOX_FRONT + length - 0.15)
    )


class BoxCourse:
    """Advance only after two disjoint successful frontier-episode batches."""

    def __init__(self):
        self.level = 0
        self.count = 0
        self.successes = 0
        self.streak = 0
        self.last_rate = 0.0

    def observe(self, count, successes, batch_size=128, threshold=0.8):
        self.count += count
        self.successes += successes
        if self.count < batch_size:
            return False
        self.last_rate = self.successes / self.count
        self.streak = self.streak + 1 if self.last_rate >= threshold else 0
        self.count = self.successes = 0
        if self.streak < 2 or self.level == len(BOX_HEIGHTS) - 1:
            return False
        self.level += 1
        self.streak = 0
        return True
