"""Render actual URDF collision geometry and label the four foot-sphere centres."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "source/robot_lab/data/Robots/TQBot/TQBot2_description"


def origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ET.parse(MODEL / "urdf/TQBot2_walk.urdf").getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    transforms = {"trunk": np.eye(4)}
    transforms["trunk"][2, 3] = .35
    pending = list(root.findall("joint"))
    while pending:
        count = len(pending)
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in transforms:
                continue
            local = origin(joint.find("origin"))
            if joint.get("type") == "revolute":
                angle = .8 if "thigh_joint" in joint.get("name") else -1.5 if "calf_joint" in joint.get("name") else 0.
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                rotation = np.eye(4)
                rotation[:3, :3] = Rotation.from_rotvec(angle * axis / np.linalg.norm(axis)).as_matrix()
                local = local @ rotation
            transforms[joint.find("child").get("link")] = transforms[parent] @ local
            pending.remove(joint)
        if len(pending) == count:
            raise ValueError("Unresolved URDF topology")

    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 11})
    fig = plt.figure(figsize=(14, 8), facecolor="#f7f9fc")
    ax = fig.add_axes([.015, .15, .65, .72], projection="3d")
    ax.set_facecolor("#f7f9fc")
    vertices = []
    for name, link in links.items():
        for collision in link.findall("collision"):
            geo = collision.find("geometry")
            child = next(iter(geo))
            if child.tag == "sphere":
                mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(child.get("radius")))
            elif child.tag == "cylinder":
                mesh = trimesh.creation.cylinder(radius=float(child.get("radius")), height=float(child.get("length")), sections=24)
            elif child.tag == "box":
                mesh = trimesh.creation.box(extents=np.fromstring(child.get("size"), sep=" "))
            elif child.tag == "mesh":
                file = MODEL / child.get("filename").removeprefix("package://TQBot2_description/")
                mesh = trimesh.load(file, force="mesh", process=False)
                mesh.apply_scale(np.fromstring(child.get("scale", "1 1 1"), sep=" "))
            else:
                raise ValueError(f"Unsupported geometry: {child.tag}")
            mesh.apply_transform(transforms[name] @ origin(collision.find("origin")))
            vertices.append(mesh.vertices)
            foot = name.endswith("_foot")
            faces = mesh.faces[::max(1, len(mesh.faces) // 5000)]
            ax.add_collection3d(Poly3DCollection(mesh.vertices[faces],
                facecolors="#ef4444" if foot else "#7b91a8", edgecolors="none", alpha=1. if foot else .24))
    positions = {}
    for leg in ("FR", "FL", "RR", "RL"):
        points = np.array([transforms[f"{leg}_{segment}"][:3, 3] for segment in ("hip", "thigh", "calf", "foot")])
        ax.plot(*points.T, color="#425a70", linewidth=2., alpha=.8)
        position = points[-1]
        positions[f"{leg}_foot"] = position.tolist()
        ax.scatter(*position, color="#ef4444", s=35, depthshade=False)
        textpos = position + np.array([.025 if leg.startswith("F") else -.065, .025 if leg.endswith("L") else -.035, -.02])
        ax.plot(*np.array([position, textpos]).T, color="#b91c1c", linewidth=1.)
        ax.text(*textpos, f"{leg}_foot", color="#b91c1c", fontsize=12, weight="bold")
    extent = np.concatenate(vertices)
    centre = (extent.min(axis=0) + extent.max(axis=0)) / 2
    span = np.ptp(extent, axis=0)
    for i, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setter(centre[i] - max(span[i] * .65, .13), centre[i] + max(span[i] * .65, .13))
    ax.set_box_aspect(np.maximum(span, .2))
    ax.view_init(elev=18, azim=-57)
    ax.set_xlabel("X / 前向 (m)")
    ax.set_ylabel("Y / 左向 (m)")
    ax.set_zlabel("Z / 向上 (m)")
    ax.set_title("四个 foot：红色足端球体的球心", pad=16, fontsize=15)

    detail = fig.add_axes([.72, .2, .25, .62])
    detail.set_facecolor("#f7f9fc")
    detail.add_patch(Rectangle((-8, -153), 16, 153, color="#7b91a8", alpha=.4))
    detail.add_patch(Circle((0, 0), 20, color="#7b91a8", alpha=.4))
    detail.add_patch(Circle((0, -153), 12, color="#ef4444", alpha=.9))
    detail.scatter([0], [0], color="#425a70", s=35)
    detail.scatter([0], [-153], color="#b91c1c", s=35)
    detail.annotate("calf 原点 / 膝关节附近", (0, 0), (-68, 32), arrowprops={"arrowstyle": "->", "color": "#425a70"}, color="#425a70")
    detail.annotate("foot 球心", (0, -153), (28, -167), arrowprops={"arrowstyle": "->", "color": "#b91c1c"}, color="#b91c1c", weight="bold")
    detail.annotate("", (-30, 0), (-30, -153), arrowprops={"arrowstyle": "<->", "color": "#425a70"})
    detail.text(-36, -77, "153 mm", rotation=90, va="center", ha="right", color="#425a70")
    detail.text(0, -199, "球心偏移 (0, 0, −153 mm)\n碰撞球半径 12 mm", ha="center", color="#b91c1c")
    detail.set_xlim(-80, 90)
    detail.set_ylim(-218, 55)
    detail.set_aspect("equal")
    detail.set_xlabel("calf 局部 X (mm)")
    detail.set_ylabel("calf 局部 Z (mm)")
    detail.set_title("单腿位置确认", fontsize=15, pad=12)
    detail.grid(alpha=.15)
    fig.suptitle("TQBot2 独立 foot 位置", fontsize=23, weight="bold", y=.965, color="#243b53")
    fig.text(.5, .06, "来自当前 URDF 的碰撞几何与正运动学；示例站姿：hip=0、thigh=0.8、calf=−1.5 rad。\n红色标记是 foot 球心，不是膝盖，也不是球面最下方的接触点。", ha="center", color="#526779", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
    args.output.with_suffix(".json").write_text(json.dumps({"source": str(MODEL / "urdf/TQBot2_walk.urdf"), "world_foot_centres_m": positions}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
