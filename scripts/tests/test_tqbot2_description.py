# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the synchronized ROS model and the Isaac-only adaptation."""

import hashlib
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

MODEL = Path(__file__).resolve().parents[2] / "source/robot_lab/data/Robots/TQBot/TQBot2_description"


def structure(element):
    return element.tag, element.attrib, (element.text or "").strip(), [structure(child) for child in element]


class DescriptionTests(unittest.TestCase):
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
