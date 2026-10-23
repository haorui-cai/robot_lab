"""CPU regression: rolling-center transport, fixed axes and reset-safe derivatives."""

import importlib.util
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "base_kinematics", ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/roll/mdp/base_kinematics.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BaseAccelerationTests(unittest.TestCase):
    def test_offset_matches_xacro_and_urdf(self):
        model = ROOT / "source/robot_lab/data/Robots/TQBot/TQBot2_description"
        for name in ("xacro/robot.xacro", "urdf/TQBot2.urdf"):
            origin = ET.parse(model / name).find("joint[@name='floating_base']/origin")
            self.assertEqual(origin.get("rpy"), "0 0 0")
            self.assertEqual(tuple(-float(x) for x in origin.get("xyz").split()), module.TQBOT2_BASE_OFFSET)

    def test_uniform_rolling_center_has_zero_acceleration(self):
        # Trunk circles an offset base while the base translates uniformly.
        # Penalizing trunk would spuriously punish the centripetal acceleration.
        h = module.AccelerationHistory(1, "cpu")
        offset = torch.tensor([module.TQBOT2_BASE_OFFSET])
        omega = torch.tensor([[-10.0, 0.0, 0.0]])
        center_velocity = torch.tensor([[0.0, 1.15, 0.0]])
        for step in range(100):
            angle = -10.0 * 0.02 * step
            q = torch.tensor([[math.cos(angle / 2), math.sin(angle / 2), 0, 0]])
            rotated = torch.tensor([[0.0, offset[0, 1] * math.cos(angle) - offset[0, 2] * math.sin(angle),
                                     offset[0, 1] * math.sin(angle) + offset[0, 2] * math.cos(angle)]])
            trunk_velocity = center_velocity - torch.linalg.cross(omega, rotated)
            base_velocity = module.base_point_velocity(trunk_velocity, omega, q, offset)
            torch.testing.assert_close(base_velocity, center_velocity, atol=1e-6, rtol=0)
            # Float32 velocity subtraction / 20 ms amplifies rounding by 50.
            torch.testing.assert_close(h.update(base_velocity, 0.02, step), torch.zeros(1, 3), atol=3e-5, rtol=0)

    def test_rotating_offset_includes_centripetal_acceleration(self):
        # With stationary trunk and constant omega, the offset point accelerates.
        h = module.AccelerationHistory(1, "cpu")
        omega = torch.tensor([[0.0, 0.0, 2.0]])
        offset = torch.tensor([[0.1, 0.0, 0.0]])
        dt = 0.001
        for step in range(2):
            theta = 2.0 * step * dt
            q = torch.tensor([[math.cos(theta / 2), 0, 0, math.sin(theta / 2)]])
            velocity = module.base_point_velocity(torch.zeros(1, 3), omega, q, offset)
            result = h.update(velocity, dt, step)
        torch.testing.assert_close(result, torch.tensor([[-0.4, -0.0004, 0.0]]), atol=2e-5, rtol=0)

    def test_free_fall_world_z_and_partial_reset(self):
        h = module.AccelerationHistory(2, "cpu")
        velocity = torch.zeros(2, 3)
        h.update(velocity, 0.02, 0)
        velocity[:, 2] -= 9.81 * 0.02
        expected = torch.tensor([[0, 0, -9.81], [0, 0, -9.81]])
        torch.testing.assert_close(h.update(velocity, 0.02, 1), expected)
        # Repeated reward evaluation cannot replace acceleration by zero.
        torch.testing.assert_close(h.update(velocity, 0.02, 1), expected)
        h.reset(torch.tensor([0]))
        velocity[0] = 1000
        velocity[1, 2] -= 9.81 * 0.02
        expected[0] = 0
        torch.testing.assert_close(h.update(velocity, 0.02, 2), expected)


if __name__ == "__main__":
    unittest.main()
