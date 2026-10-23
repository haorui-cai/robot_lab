"""Performance gates for independently expanding uphill/downhill rolling ranges."""

import math

import numpy as np
import trimesh


class SlopeCourse:
    limits = {"uphill": (0, 2, 4, 6, 8, 10), "downhill": (0, 3, 6, 10, 15, 20, 25, 30)}

    def __init__(self):
        self.levels = {name: 0 for name in self.limits}
        self.batches = {name: dict(count=0, roll=0, joint=0, streak=0, last_step=0) for name in self.limits}

    def maximum(self, name):
        return self.limits[name][self.levels[name]]

    def observe(self, name, count, roll_success, joint_success, step, batch_size=512, min_steps=500):
        batch = self.batches[name]
        batch["count"] += count
        batch["roll"] += roll_success
        batch["joint"] += joint_success
        if batch["count"] < batch_size or step - batch["last_step"] < min_steps:
            return False
        passed = batch["roll"] / batch["count"] >= 0.75 and batch["joint"] / batch["count"] >= 0.70
        batch["streak"] = batch["streak"] + 1 if passed else 0
        batch.update(count=0, roll=0, joint=0, last_step=step)
        if batch["streak"] < 2 or self.levels[name] == len(self.limits[name]) - 1:
            return False
        self.levels[name] += 1
        batch["streak"] = 0
        return True


def build_slope_mesh(width=40.0, length=60.0):
    """41 isolated real collision ramps, 1 degree apart; +y is the travel direction.

    The top surface is z = tan(angle) * local_y. No underlying horizontal plane
    is added, so negative slopes remain genuine downhill collision surfaces.
    """
    angles = np.arange(-30, 11, dtype=np.float32)
    origins, meshes = [], []
    for index, angle in enumerate(angles):
        origin = np.array([(index % 6) * (width + 8), (index // 6) * (length + 8), 20.0])
        mesh = trimesh.creation.box(extents=(width, length, 0.2))
        mesh.vertices[:, 2] += -0.1 + math.tan(math.radians(float(angle))) * mesh.vertices[:, 1]
        mesh.apply_translation(origin)
        origins.append(origin)
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes), np.asarray(origins), angles
