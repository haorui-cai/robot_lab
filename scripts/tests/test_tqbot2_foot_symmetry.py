"""Geometry and scale checks for the replacement diagonal coordination cost."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import least_squares
import torch

from test_tqbot2_walk_stance import forward_kinematics, STANCE

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "source/robot_lab/data/Robots/TQBot/TQBot2_description/urdf/TQBot2_walk.urdf"
SPEC = importlib.util.spec_from_file_location(
    "foot_symmetry", ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/foot_symmetry.py",
)
symmetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(symmetry)
FEET = ("FR_foot", "RL_foot", "FL_foot", "RR_foot")
NAMES = [f"{foot[:-5]}_{kind}_joint" for foot in FEET for kind in ("hip", "thigh", "calf")]
REFERENCE = {name: 0. if "hip" in name else STANCE.WALK_THIGH_ANGLE if "thigh" in name
             else STANCE.WALK_CALF_ANGLE for name in NAMES}


class FootSymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = symmetry.calibrate_diagonal_displacement(URDF, FEET, REFERENCE)
        cls.reference = torch.tensor(cls.calibration["reference_positions_body"], dtype=torch.float64)

    def test_standing_and_coordinated_pairs_are_free(self):
        position = self.reference[None].repeat(2, 1, 1)
        position[1, :2] += torch.tensor([.03, 0., .04])
        position[1, 2:] += torch.tensor([-.02, 0., -.01])
        torch.testing.assert_close(symmetry.diagonal_displacement_cost(position, self.reference), torch.zeros(2, dtype=torch.float64))

    def test_mismatch_has_expected_pair_average_and_lateral_motion_is_free(self):
        position = self.reference[None].repeat(2, 1, 1)
        position[0, 0] += torch.tensor([.02, 0., .03], dtype=torch.float64)
        position[1, :, 1] += torch.tensor([.1, -.1, .2, -.2], dtype=torch.float64)
        torch.testing.assert_close(symmetry.diagonal_displacement_cost(position, self.reference),
                                   torch.tensor([.00065, 0.], dtype=torch.float64))

    def test_batched_fk_matches_independent_stance_implementation(self):
        rng = np.random.default_rng(301)
        samples = {name: value + rng.uniform(-.04, .04, 3) for name, value in reversed(list(REFERENCE.items()))}
        actual = symmetry.urdf_foot_positions(URDF, samples, FEET)
        for index in range(3):
            _, poses = forward_kinematics(URDF, "trunk", {n: q[index] for n, q in samples.items()})
            np.testing.assert_allclose(actual[index], np.array([poses[n][:3, 3] for n in FEET]), atol=1e-12)

    def test_equal_foot_motion_is_free_despite_opposite_joint_changes(self):
        q = REFERENCE.copy()
        for index, foot in enumerate(FEET):
            names = [f"{foot[:-5]}_{kind}_joint" for kind in ("thigh", "calf")]
            target = self.reference[index, (0, 2)].numpy() + np.array([.02, .02])

            def error(angles):
                candidate = {**REFERENCE, **dict(zip(names, angles))}
                _, poses = forward_kinematics(URDF, "trunk", candidate)
                return poses[foot][[0, 2], 3] - target

            solution = least_squares(error, [q[n] for n in names], gtol=1e-13, ftol=1e-13, xtol=1e-13)
            self.assertLess(np.linalg.norm(solution.fun), 1e-9)
            q.update(zip(names, solution.x))
        _, poses = forward_kinematics(URDF, "trunk", q)
        feet = torch.tensor(np.array([poses[n][:3, 3] for n in FEET]))[None]
        self.assertLess(symmetry.diagonal_displacement_cost(feet, self.reference).item(), 1e-16)
        old = np.array([q[n] for n in NAMES]).reshape(4, 3)
        self.assertGreater(np.square(old[::2] - old[1::2]).sum(), .01)

    def test_old_equal_angles_can_move_diagonal_feet_in_opposite_directions(self):
        q = {name: [value + (.04 if "thigh" in name else 0.)] for name, value in REFERENCE.items()}
        feet = torch.tensor(symmetry.urdf_foot_positions(URDF, q, FEET))
        delta = feet[0] - self.reference
        self.assertLess(delta[0, 0] * delta[1, 0], 0.)
        self.assertGreater(symmetry.diagonal_displacement_cost(feet, self.reference).item(), .0001)

    def test_calibration_matches_old_cost_on_same_samples_and_on_holdout(self):
        for seed in (42, 20261015):
            rng = np.random.default_rng(seed)
            q = np.array([REFERENCE[n] for n in NAMES]) + rng.uniform(-.05, .05, (8192, 12))
            feet = torch.tensor(symmetry.urdf_foot_positions(URDF, dict(zip(NAMES, q.T)), FEET))
            new = symmetry.diagonal_displacement_cost(feet, self.reference).mean().item() * self.calibration["scale"]
            legs = q.reshape(-1, 4, 3)
            old = np.square(legs[:, ::2] - legs[:, 1::2]).sum(axis=-1).mean()
            self.assertAlmostEqual(new / old, 1., delta=1e-12 if seed == 42 else .04)

    def test_calibration_does_not_consume_global_rng(self):
        np.random.seed(123)
        before = np.random.get_state()
        again = symmetry.calibrate_diagonal_displacement(URDF, FEET, dict(reversed(list(REFERENCE.items()))))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(again, self.calibration)


if __name__ == "__main__":
    unittest.main()
