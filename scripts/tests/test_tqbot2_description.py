# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the synchronized ROS model and the Isaac-only adaptation."""

import hashlib
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MODEL = Path(__file__).resolve().parents[2] / "source/robot_lab/data/Robots/TQBot/TQBot2_description"


def structure(element):
    return element.tag, element.attrib, (element.text or "").strip(), [structure(child) for child in element]


class DescriptionTests(unittest.TestCase):
    def test_walking_derivative_preserves_foot_physics(self):
        source_path = MODEL / "urdf/TQBot2_isaac.urdf"
        target_path = MODEL / "urdf/TQBot2_walk.urdf"
        manifest = json.loads((MODEL / "walk_manifest.json").read_text())
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), manifest["source_sha256"])
        self.assertEqual(hashlib.sha256(target_path.read_bytes()).hexdigest(), manifest["derived_sha256"])
        source, target = ET.parse(source_path).getroot(), ET.parse(target_path).getroot()
        constants = ET.parse(MODEL / "xacro/const.xacro").getroot()
        self.assertEqual(float(next(p.get("value") for p in constants if p.get("name") == "thigh_position_min")), .2)
        preserved = set()
        for joint in target.findall("joint"):
            if "dont_collapse" in joint.attrib:
                self.assertEqual(joint.attrib.pop("dont_collapse"), "true")
                self.assertEqual(joint.get("type"), "fixed")
                preserved.add(joint.get("name"))
        self.assertEqual(preserved, {f"{leg}_foot_fixed" for leg in ("FR", "FL", "RR", "RL")})
        for leg in ("FR", "FL", "RR", "RL"):
            name = f"{leg}_thigh_joint"
            self.assertEqual(float(source.find(f"joint[@name='{name}']/limit").get("lower")), .2)
            limits = target.find(f"joint[@name='{name}']/limit")
            self.assertEqual([float(limits.get(k)) for k in ("lower", "upper")], [.2, 3.2])
            self.assertEqual(manifest["thigh_hard_limits_rad"][name], [.2, 3.2])
            # Allow only the builder's explicit lower-bound adaptation.
            limits.set("lower", source.find(f"joint[@name='{name}']/limit").get("lower"))
        self.assertEqual(structure(source), structure(target))
        self.assertEqual(len([j for j in target.findall("joint") if j.get("type") == "revolute"]), 12)

    def test_left_thigh_derivative(self):
        source_path = MODEL / "urdf/TQBot2_isaac.urdf"
        target_path = MODEL / "urdf/TQBot2_left_thigh_roll.urdf"
        manifest = json.loads((MODEL / "left_thigh_roll_manifest.json").read_text())
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), manifest["source_sha256"])
        self.assertEqual(hashlib.sha256(target_path.read_bytes()).hexdigest(), manifest["derived_sha256"])
        source, target = ET.parse(source_path).getroot(), ET.parse(target_path).getroot()
        self.assertEqual([structure(link) for link in source.findall("link")],
                         [structure(link) for link in target.findall("link")])
        active = {joint.get("name") for joint in target.findall("joint") if joint.get("type") == "revolute"}
        self.assertEqual(active, {f"{leg}_hip_joint" for leg in ("FR", "FL", "RR", "RL")} |
                         {"FL_thigh_joint", "RL_thigh_joint"})
        for leg in ("FL", "RL"):
            name = f"{leg}_thigh_joint"
            original = source.find(f"joint[@name='{name}']")
            derived = target.find(f"joint[@name='{name}']")
            self.assertEqual(structure(original.find("origin")), structure(derived.find("origin")))
            self.assertEqual(structure(original.find("axis")), structure(derived.find("axis")))
            self.assertEqual(float(derived.find("limit").get("lower")), 1.68)
            self.assertEqual(float(derived.find("limit").get("upper")), 2.45)
        self.assertEqual(len(manifest["locked_joint_angles_rad"]), 6)
        old_heights = json.loads((MODEL / "roll_spawn_heights.json").read_text())
        new_heights = json.loads((MODEL / "left_thigh_roll_spawn_heights.json").read_text())
        self.assertEqual(new_heights["urdf_sha256"], manifest["derived_sha256"])
        # At the same curl, restoring the left hinge must not rotate geometry twice.
        np.testing.assert_allclose(new_heights["heights_m"], old_heights["heights_m"], atol=1e-12)

    def test_hip_only_derivative(self):
        source_path = MODEL / "urdf/TQBot2_isaac.urdf"
        derived_path = MODEL / "urdf/TQBot2_hip_only.urdf"
        manifest = json.loads((MODEL / "hip_only_manifest.json").read_text())
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), manifest["source_sha256"])
        self.assertEqual(hashlib.sha256(derived_path.read_bytes()).hexdigest(), manifest["derived_sha256"])
        source, derived = ET.parse(source_path).getroot(), ET.parse(derived_path).getroot()
        self.assertEqual([structure(link) for link in source.findall("link")],
                         [structure(link) for link in derived.findall("link")])
        active = [joint.get("name") for joint in derived.findall("joint") if joint.get("type") == "revolute"]
        self.assertEqual(set(active), {f"{leg}_hip_joint" for leg in ("FR", "FL", "RR", "RL")})
        for name, angle in manifest["locked_joint_angles_rad"].items():
            original = source.find(f"joint[@name='{name}']")
            fixed = derived.find(f"joint[@name='{name}']")
            self.assertEqual(fixed.get("type"), "fixed")
            self.assertEqual(original.find("origin").get("xyz"), fixed.find("origin").get("xyz"))
            rpy = np.fromstring(original.find("origin").get("rpy"), sep=" ")
            axis = np.fromstring(original.find("axis").get("xyz"), sep=" ")
            expected = Rotation.from_euler("xyz", rpy) * Rotation.from_rotvec(angle * axis / np.linalg.norm(axis))
            actual = Rotation.from_euler("xyz", np.fromstring(fixed.find("origin").get("rpy"), sep=" "))
            np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=1e-12)

    def test_source_hashes_and_derived_hash(self):
        manifest = json.loads((MODEL / "sync_manifest.json").read_text())
        for path, expected in manifest["source_files_sha256"].items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((MODEL / path).read_bytes()).hexdigest(), expected)
        self.assertEqual(
            hashlib.sha256((MODEL / "urdf/TQBot2_isaac.urdf").read_bytes()).hexdigest(),
            manifest["isaac_urdf_sha256"],
        )

    def test_adaptation_preserves_physics(self):
        source = ET.parse(MODEL / "urdf/TQBot2.urdf").getroot()
        adapted = ET.parse(MODEL / "urdf/TQBot2_isaac.urdf").getroot()
        source.remove(source.find("link[@name='base']"))
        source.remove(source.find("joint[@name='floating_base']"))
        aliases = json.loads((MODEL / "sync_manifest.json").read_text())["isaac_mesh_aliases"]
        for mesh in adapted.iter("mesh"):
            relative = mesh.get("filename").removeprefix("package://TQBot2_description/")
            self.assertTrue((MODEL / relative).is_file())
            self.assertNotIn(".", Path(relative).stem)
            if relative in aliases:
                self.assertEqual((MODEL / relative).read_bytes(), (MODEL / aliases[relative]).read_bytes())
                mesh.set("filename", "package://TQBot2_description/" + aliases[relative])
        self.assertEqual(structure(source), structure(adapted))

    def test_prone_pose_within_updated_limits(self):
        source = ET.parse(MODEL / "urdf/TQBot2.urdf").getroot()
        actuated = [j for j in source.findall("joint") if j.get("type") == "revolute"]
        self.assertEqual(len(actuated), 12)
        targets = {"hip": 0.0, "thigh": 1.36, "calf": -2.65}
        for joint in actuated:
            kind = joint.get("name").split("_")[1]
            limit = joint.find("limit")
            self.assertLess(float(limit.get("lower")), targets[kind])
            self.assertGreater(float(limit.get("upper")), targets[kind])
        for leg in ("FR", "FL", "RR", "RL"):
            sphere = source.find(f"link[@name='{leg}_calf']/collision[@name='knee_collision']/geometry/sphere")
            self.assertAlmostEqual(float(sphere.get("radius")), 0.02)


if __name__ == "__main__":
    unittest.main()
