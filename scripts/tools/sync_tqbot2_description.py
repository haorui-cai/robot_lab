# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Sync the ROS model and generate an Isaac URDF with the physical trunk as root."""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DESTINATION = REPO / "source/robot_lab/data/Robots/TQBot/TQBot2_description"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/eureka/docker-20.04/unitree_guide/src/unitree_ros/robots/TQBot2_description"),
    )
    args = parser.parse_args()
    source = args.source.resolve()
    urdf = source / "urdf/TQBot2.urdf"
    parser_xml = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(urdf, parser=parser_xml)
    root = tree.getroot()
    base = root.find("link[@name='base']")
    joint = root.find("joint[@name='floating_base']")
    if base is None or joint is None or joint.get("type") != "fixed":
        raise ValueError("Expected a massless virtual base linked to trunk by floating_base")
    if (
        base.find("inertial") is not None
        or base.find("collision") is not None
        or joint.find("parent").get("link") != "base"
        or joint.find("child").get("link") != "trunk"
    ):
        raise ValueError("Virtual base topology changed; review adaptation before syncing")
    if len(root.findall("joint/parent[@link='base']")) != 1:
        raise ValueError("Virtual base has additional children; adaptation needs review")
    hashes = {}
    for relative in ("urdf", "meshes", "xacro", "config", "launch", "package.xml", "CMakeLists.txt"):
        entry = source / relative
        for file in sorted(entry.rglob("*")) if entry.is_dir() else [entry]:
            if not file.is_file():
                continue
            dest = DESTINATION / file.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, dest)
            hashes[str(file.relative_to(source))] = hashlib.sha256(file.read_bytes()).hexdigest()

    # Change coordinates from shell centre to trunk, not physical geometry.
    # Removing it preserves existing trunk sensor paths after fixed-link merging.
    root.remove(base)
    root.remove(joint)
    root.insert(0, ET.Comment(" Isaac adaptation: virtual base/floating_base removed; physical trunk is root. "))
    # Isaac Sim 5.1 uses the filename stem as a USD prim name. A double suffix
    # such as base_link.STL.dae creates an invalid path (base_link.STL).
    aliases = {}
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        original = Path(filename.removeprefix("package://TQBot2_description/"))
        if "." in original.stem:
            alias = original.with_name(original.name.split(".")[0] + original.suffix)
            alias_path = DESTINATION / alias
            if alias_path.exists():
                if alias_path.read_bytes() != (DESTINATION / original).read_bytes():
                    raise ValueError(f"Mesh alias conflicts with an existing file: {alias}")
            else:
                # Relative symlink preserves the source mesh without duplicating it.
                alias_path.symlink_to(original.name)
            mesh.set("filename", f"package://TQBot2_description/{alias}")
            aliases[str(alias)] = str(original)
    derived = DESTINATION / "urdf/TQBot2_isaac.urdf"
    tree.write(derived, encoding="utf-8", xml_declaration=True)
    manifest = {
        "source": str(source),
        "source_files_sha256": hashes,
        "isaac_adaptation": "Remove massless base and floating_base; retain all trunk-relative physical data.",
        "isaac_mesh_aliases": aliases,
        "isaac_urdf_sha256": hashlib.sha256(derived.read_bytes()).hexdigest(),
    }
    (DESTINATION / "sync_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Synced {len(hashes)} source files; generated {derived}")


if __name__ == "__main__":
    main()
