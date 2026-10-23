"""Check the walking stance against real URDF geometry and reference frames."""
import ast
import importlib.util
import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'source/robot_lab/data/Robots/TQBot/TQBot2_description/urdf'
SPEC = importlib.util.spec_from_file_location('walk_stance', ROOT / 'source/robot_lab/robot_lab/assets/tqbot_walk_stance.py')
STANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STANCE)


def forward_kinematics(path, root_link, angles):
    root = ET.parse(path).getroot()
    poses = {root_link: np.eye(4)}
    pending = list(root.findall('joint'))
    while pending:
        count = len(pending)
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in poses:
                continue
            transform = np.eye(4)
            origin = joint.find('origin')
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            if joint.get('type') == 'revolute':
                matches = [q for pattern, q in angles.items() if re.fullmatch(pattern, joint.get('name'))]
                if len(matches) != 1:
                    raise ValueError((joint.get('name'), matches))
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis / np.linalg.norm(axis) * matches[0]).as_matrix()
            poses[joint.find('child').get('link')] = poses[parent] @ transform
            pending.remove(joint)
        if len(pending) == count:
            raise ValueError('Unresolved URDF topology')
    return root, poses


class WalkStanceTests(unittest.TestCase):
    def test_all_feet_reach_requested_height_and_approach_hip(self):
        old = dict(STANCE.WALK_JOINT_POS)
        old.update({'F.*_thigh_joint': .8, 'R.*_thigh_joint': .8, '.*_calf_joint': -1.5})
        _, before = forward_kinematics(MODEL / 'TQBot2_walk.urdf', 'trunk', old)
        _, after = forward_kinematics(MODEL / 'TQBot2_walk.urdf', 'trunk', STANCE.WALK_JOINT_POS)
        for leg in ('FR', 'FL', 'RR', 'RL'):
            old_delta = before[leg + '_foot'][:3, 3] - before[leg + '_hip'][:3, 3]
            new_delta = after[leg + '_foot'][:3, 3] - after[leg + '_hip'][:3, 3]
            self.assertLess(abs(new_delta[0]), .04)
            self.assertLess(abs(new_delta[0]), abs(old_delta[0]) * .75)
            self.assertAlmostEqual(after[leg + '_foot'][2, 3], -.25, places=12)
            self.assertAlmostEqual(new_delta[2], STANCE.WALK_DEFAULT_FOOT_Z_TRUNK, places=12)
            self.assertAlmostEqual(STANCE.WALK_FOOT_HEIGHT_TARGET_TRUNK - after[leg + '_foot'][2, 3], .085, places=12)

    def test_new_stance_inside_actual_soft_limits_without_reducing_original_thigh_margin(self):
        root, _ = forward_kinematics(MODEL / 'TQBot2_walk.urdf', 'trunk', STANCE.WALK_JOINT_POS)
        for joint in root.findall('joint'):
            if joint.get('type') != 'revolute':
                continue
            q = next(q for p, q in STANCE.WALK_JOINT_POS.items() if re.fullmatch(p, joint.get('name')))
            limits = joint.find('limit')
            low, high = float(limits.get('lower')), float(limits.get('upper'))
            midpoint, half = (low + high) / 2, (high - low) * .9 / 2
            self.assertGreater(q, midpoint - half)
            self.assertLess(q, midpoint + half)
            if 'thigh_joint' in joint.get('name'):
                self.assertGreaterEqual(q - (midpoint - half), .8 - (midpoint - half))

    def test_ros_base_translation_is_distinct_from_imported_trunk(self):
        source, in_base = forward_kinematics(MODEL / 'TQBot2.urdf', 'base', STANCE.WALK_JOINT_POS)
        imported, in_trunk = forward_kinematics(MODEL / 'TQBot2_walk.urdf', 'trunk', STANCE.WALK_JOINT_POS)
        self.assertIsNone(imported.find("link[@name='base']"))
        translation = np.fromstring(source.find("joint[@name='floating_base']/origin").get('xyz'), sep=' ')
        self.assertGreater(abs(translation[2]), .02)
        for leg in ('FR', 'FL', 'RR', 'RL'):
            np.testing.assert_allclose(in_base[leg + '_foot'][:3, 3] - in_trunk[leg + '_foot'][:3, 3], translation, atol=1e-12)

    def test_shared_asset_keeps_other_tasks_default_angles(self):
        tree = ast.parse((ROOT / 'source/robot_lab/robot_lab/assets/tqbot.py').read_text())
        cfg = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TQBOT2_CFG' for t in n.targets))
        init = next(k.value for k in cfg.keywords if k.arg == 'init_state')
        positions = ast.literal_eval(next(k.value for k in init.keywords if k.arg == 'joint_pos'))
        self.assertEqual(positions['F.*_thigh_joint'], .8)
        self.assertEqual(positions['R.*_thigh_joint'], .8)
        self.assertEqual(positions['.*_calf_joint'], -1.5)

    def test_policy_zero_is_inside_limits_and_halfway_between_previous_thigh_zeros(self):
        root = ET.parse(MODEL / 'TQBot2_walk.urdf').getroot()
        for joint in root.findall('joint'):
            if joint.get('type') != 'revolute':
                continue
            name = joint.get('name')
            matches = [q for p, q in STANCE.WALK_POLICY_ZERO_JOINT_POS.items() if re.fullmatch(p, name)]
            self.assertEqual(len(matches), 1, name)
            q = matches[0]
            low, high = (float(joint.find('limit').get(k)) for k in ('lower', 'upper'))
            center, half = (low + high) / 2, .9 * (high - low) / 2
            self.assertLess(center - half, q)
            self.assertGreater(center + half, q)
            if '_thigh_' in name:
                self.assertEqual((low, high), STANCE.WALK_POLICY_THIGH_LIMITS)
                # Retained alternative zero is independent of later limit changes.
                self.assertAlmostEqual(q, (1.90 + .8) / 2)
                self.assertAlmostEqual(q, 1.35)
            elif '_hip_' in name:
                self.assertEqual(q, 0.)
            else:
                self.assertAlmostEqual(q, -1.6445578923423)

    def test_policy_zero_and_standing_reference_are_independent(self):
        self.assertIsNot(STANCE.WALK_STANDING_JOINT_POS, STANCE.WALK_POLICY_ZERO_JOINT_POS)
        standing = STANCE.WALK_STANDING_JOINT_POS.copy()
        policy = STANCE.WALK_POLICY_ZERO_JOINT_POS.copy()
        for name in ('FR', 'FL', 'RR', 'RL'):
            q_stand = next(q for p, q in standing.items() if re.fullmatch(p, name + '_thigh_joint'))
            q_zero = next(q for p, q in policy.items() if re.fullmatch(p, name + '_thigh_joint'))
            self.assertAlmostEqual((q_stand - q_zero) / .25, -2.2)
        self.assertAlmostEqual(STANCE.WALK_FOOT_HEIGHT_TARGET_TRUNK, -.165)

    def test_legacy_zero_matches_current_standing_angles_without_aliasing_references(self):
        legacy = STANCE.WALK_LEGACY_POLICY_ZERO_JOINT_POS
        self.assertIsNot(legacy, STANCE.WALK_STANDING_JOINT_POS)
        self.assertIsNot(legacy, STANCE.WALK_POLICY_ZERO_JOINT_POS)
        for leg in ('FR', 'FL', 'RR', 'RL'):
            for kind in ('hip', 'thigh', 'calf'):
                name = f'{leg}_{kind}_joint'
                old = next(q for p, q in legacy.items() if re.fullmatch(p, name))
                standing = next(q for p, q in STANCE.WALK_STANDING_JOINT_POS.items() if re.fullmatch(p, name))
                self.assertAlmostEqual(old, standing)
        self.assertEqual(STANCE.WALK_POLICY_ZERO_JOINT_POS['.*_thigh_joint'], 1.35)


if __name__ == '__main__':
    unittest.main()
