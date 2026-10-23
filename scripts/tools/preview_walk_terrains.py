"""Render saved walking terrain parameters with the installed Isaac Lab mesh functions.

Runs on CPU without starting Isaac Sim. Random layouts are representative
examples, not a reconstruction of the live simulator's unrecoverable RNG state.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.font_manager import FontProperties
import numpy as np
import torch
import trimesh
import yaml


LABELS = {
    "pyramid_stairs": ("凸形阶梯", "台阶高", "step_height_range"),
    "pyramid_stairs_inv": ("凹形阶梯", "台阶高", "step_height_range"),
    "boxes": ("随机方块", "高度扰动 ±", "grid_height_range"),
    "random_rough": ("随机起伏", "高度采样 2–10 cm；不随等级变", None),
    "hf_pyramid_slope": ("凸形坡面", "坡度", "slope_range"),
    "hf_pyramid_slope_inv": ("凹形坡面", "坡度", "slope_range"),
    "high_friction_flat": ("高摩擦平地", "足端摩擦 2.5；不随等级变", None),
}


def load_cpu_generators(terrain_root):
    # Only the pure geometry modules execute; the simulator package initializer does not.
    for name, path in (("_terrain_preview", terrain_root),
                       ("_terrain_preview.trimesh", terrain_root / "trimesh"),
                       ("_terrain_preview.height_field", terrain_root / "height_field")):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    return {kind: importlib.import_module(f"_terrain_preview.{kind}.{module}")
            for kind, module in (("trimesh", "mesh_terrains"), ("height_field", "hf_terrains"))}


def plot_mesh(ax, mesh, title, subtitle, font, color):
    # Orthographic CPU depth buffer avoids painter-order artifacts where the
    # generator's stair boxes overlap. Rasterize the original triangles.
    azimuth, elevation = np.radians([-52, 33])
    view = np.array([np.cos(elevation) * np.cos(azimuth),
                     np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    up = np.cross(view, right)
    basis = np.stack([right, up, view], axis=1)
    projected = mesh.vertices @ basis
    visible = mesh.face_normals @ view > 0
    triangles = mesh.faces[visible]
    normals = mesh.face_normals[visible]
    light = LightSource(azdeg=315, altdeg=45).direction
    intensity = 0.5 + 0.5 * np.clip(normals @ light, 0, 1)
    base = np.array(matplotlib.colors.to_rgb(color))
    colors = np.clip(intensity[:, None] * base, 0, 1)
    bounds = np.array([projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)])
    width = 900
    span = bounds[1] - bounds[0]
    scale = (width - 60) / span[0]
    height = int(span[1] * scale) + 60
    screen = projected.copy()
    screen[:, :2] = (projected[:, :2] - bounds[0]) * scale + 30
    screen[:, 1] = height - 1 - screen[:, 1]
    pixels = np.tile(matplotlib.colors.to_rgb("#f4f7f9"), (height, width, 1))
    depth = np.full((height, width), -np.inf)
    for face, face_color in zip(triangles, colors):
        a, b, c = screen[face]
        x0, y0 = np.maximum(np.floor(np.minimum.reduce([a[:2], b[:2], c[:2]])), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(np.maximum.reduce([a[:2], b[:2], c[:2]])), [width - 1, height - 1]).astype(int)
        determinant = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(determinant) < 1e-10:
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / determinant
        wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / determinant
        wc = 1 - wa - wb
        z = wa * a[2] + wb * b[2] + wc * c[2]
        block = depth[y0:y1+1, x0:x1+1]
        covered = (wa >= -1e-8) & (wb >= -1e-8) & (wc >= -1e-8) & (z > block)
        block[covered] = z[covered]
        pixels[y0:y1+1, x0:x1+1][covered] = face_color
    ax.imshow(pixels)
    ax.set_axis_off()
    ax.set_title(title + "\n" + subtitle, fontproperties=font, fontsize=12, color="#183447", pad=0)
    origin = np.array(mesh.metadata["preview_origin"]) @ basis
    origin_pixel = (origin[:2] - bounds[0]) * scale + 30
    ax.scatter([origin_pixel[0]], [height - 1 - origin_pixel[1]], s=12, c="#d94b3d", edgecolors="white", linewidths=.5)
    ax.set_facecolor("#f4f7f9")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaaclab", type=Path, default=Path("/home/eureka/IsaacLab/source/isaaclab/isaaclab/terrains"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    saved = yaml.load(args.config.read_text(), Loader=yaml.BaseLoader)
    cfg = saved["scene"]["terrain"]["terrain_generator"]
    generators = load_cpu_generators(args.isaaclab)
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    meshes = {}
    subtitles = {}
    manifest = {"saved_config": str(args.config.resolve()), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "representative_seed": 42, "levels": [0, 4, 9], "difficulty": [.05, .45, .95],
                "note": "Representative mesh realizations from the saved configuration, not live terrain snapshots.",
                "source_sha256": {}, "terrains": {}}
    for source in (args.isaaclab / "trimesh/mesh_terrains.py", args.isaaclab / "trimesh/utils.py",
                   args.isaaclab / "height_field/hf_terrains.py", args.isaaclab / "height_field/utils.py"):
        manifest["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    for index, (name, raw) in enumerate(cfg["sub_terrains"].items()):
        parameters = {}
        for key, value in raw.items():
            if key in ("function", "flat_patch_sampling"):
                continue
            if isinstance(value, list):
                parameters[key] = tuple(map(float, value))
            elif value in ("true", "false"):
                parameters[key] = value == "true"
            elif value == "null":
                parameters[key] = None
            else:
                parameters[key] = float(value)
        function_name = raw["function"].split(":")[-1]
        kind = "height_field" if "height_field" in raw["function"] else "trimesh"
        function = getattr(generators[kind], function_name)
        label, detail, parameter = LABELS[name]
        manifest["terrains"][name] = {"parameters": parameters, "function": raw["function"]}
        for level, difficulty in ((0, .05), (4, .45), (9, .95)):
            np.random.seed(42 + index)
            torch.manual_seed(42 + index)
            pieces, origin = function(difficulty, SimpleNamespace(**parameters))
            mesh = trimesh.util.concatenate(pieces)
            mesh.apply_translation([-4, -4, 0])
            mesh.metadata["preview_origin"] = (origin + np.array([-4, -4, 0])).tolist()
            meshes[name, level] = mesh
            if parameter:
                lo, hi = parameters[parameter]
                value = lo + difficulty * (hi - lo)
                detail_text = f"{detail} {np.degrees(np.arctan(value)):.1f}°" if parameter == "slope_range" else f"{detail} {100 * value:.1f} cm"
            else:
                detail_text = detail
            subtitles[name, level] = detail_text
        manifest["terrains"][name]["roughness_changes_with_level"] = name not in ("random_rough", "high_friction_flat")
    names = list(cfg["sub_terrains"])
    palette = ["#69a6b3", "#84ad93", "#ccac78", "#94adbd", "#8bb5aa", "#a2a3c6", "#aab6c2"]
    fig = plt.figure(figsize=(15, 13), facecolor="#f4f7f9")
    fig.suptitle("TQBot2 当前训练地形 · 7 类", fontproperties=font, fontsize=23, color="#183447", y=.98)
    fig.text(.5, .935, "同一保存配置生成的代表样例 · 第 4 级 · 单块 8 × 8 m · 几何比例未夸大", ha="center", fontproperties=font, fontsize=12)
    for i, name in enumerate(names):
        ax = fig.add_subplot(4, 2, i + 1)
        plot_mesh(ax, meshes[name, 4], f"{i+1:02d}  {LABELS[name][0]}", subtitles[name, 4], font, palette[i])
    fig.text(.75, .115, "红点标记中心位置\n等级按环境独立变化，地形类型固定\n随机起伏和平地不随等级改变几何", ha="center", va="center", fontproperties=font, fontsize=13, color="#183447", linespacing=1.9)
    fig.subplots_adjust(top=.9, bottom=.02, hspace=.18, wspace=.03)
    fig.savefig(args.output / "overview.png", dpi=160)
    plt.close(fig)
    grid = plt.figure(figsize=(16, 22), facecolor="#f4f7f9")
    grid.suptitle("地形课程对照 · 第 0 / 4 / 9 级", fontproperties=font, fontsize=22, y=.99)
    for i, name in enumerate(names):
        comparison = plt.figure(figsize=(14, 4), facecolor="#f4f7f9")
        for j, level in enumerate((0, 4, 9)):
            title = f"{LABELS[name][0]} · 第 {level} 级"
            for figure, position in ((grid, (7, 3, i * 3 + j + 1)), (comparison, (1, 3, j + 1))):
                ax = figure.add_subplot(*position)
                plot_mesh(ax, meshes[name, level], title, subtitles[name, level], font, palette[i])
        comparison.subplots_adjust(top=.83, bottom=.02, left=.01, right=.99, wspace=.04)
        comparison.savefig(args.output / f"{name}.png", dpi=150)
        plt.close(comparison)
    grid.subplots_adjust(top=.96, bottom=.01, left=.01, right=.99, wspace=.03, hspace=.25)
    grid.savefig(args.output / "levels.png", dpi=150)
    plt.close(grid)
    # Verify the two difficulty-independent generators produce identical realizations.
    for name in ("random_rough", "high_friction_flat"):
        np.testing.assert_array_equal(meshes[name, 0].vertices, meshes[name, 9].vertices)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "terrains": names, "cpu": not torch.cuda.is_available()}))


if __name__ == "__main__":
    main()
