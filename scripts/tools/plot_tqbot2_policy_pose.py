"""Render standing or policy-zero pose from walking URDF forward kinematics."""

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/TQBot2_policy_pose_20261014")
parser.add_argument("--pose", choices=("standing", "policy_zero"), default="policy_zero")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("stance_fk", ROOT / "scripts/tests/test_tqbot2_walk_stance.py")
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)
urdf = fk.MODEL / "TQBot2_walk.urdf"
joint_pos = (fk.STANCE.WALK_STANDING_JOINT_POS if args.pose == "standing"
             else fk.STANCE.WALK_POLICY_ZERO_JOINT_POS)
_, poses = fk.forward_kinematics(urdf, "trunk", joint_pos)
angle = lambda kind: next(q for pattern, q in joint_pos.items() if re.fullmatch(pattern, f"FR_{kind}_joint"))
pose_title = "默认站立姿态（旧策略零位）" if args.pose == "standing" else "保留的新策略零位姿态"
stem = "standing_pose" if args.pose == "standing" else "policy_zero_pose"
legs = ("FR", "FL", "RR", "RL")
points = {leg: np.array([poses[f"{leg}_{part}"][:3, 3] for part in ("hip", "thigh", "calf", "foot")])
          for leg in legs}
feet = np.array([points[leg][-1] for leg in legs])
assert np.ptp(feet[:, 2]) < 1e-12
for chain in points.values():
    np.testing.assert_allclose(np.linalg.norm(np.diff(chain, axis=0), axis=1), [.08, .213, .153], atol=1e-12)
z_foot = feet[0, 2]
x_span, y_span = np.ptp(feet[:, 0]), np.ptp(feet[:, 1])
x_inset = points["FR"][0, 0] - points["FR"][-1, 0]

plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
ink, gray, thigh, calf, foot = "#203348", "#8A98A7", "#C77A29", "#2675AE", "#C44448"
fig = plt.figure(figsize=(15.4, 10), facecolor="white")
grid = fig.add_gridspec(2, 2, width_ratios=[1.14, 1], left=.04, right=.95, top=.83, bottom=.17,
                       wspace=.16, hspace=.50)
ax3 = fig.add_subplot(grid[:, 0], projection="3d")
side = fig.add_subplot(grid[0, 1])
front = fig.add_subplot(grid[1, 1])

# The gray frame connects actual hip axes, and is not a fabricated body shell.
frame = np.array([points[leg][0] for leg in ("FR", "FL", "RL", "RR", "FR")])
ax3.plot(*frame.T, color=gray, lw=5, alpha=.65)
ax3.add_collection3d(Poly3DCollection([frame[:-1]], facecolors="#CAD5DF", alpha=.15, edgecolor="none"))
plane = feet[[0, 1, 3, 2]]
ax3.add_collection3d(Poly3DCollection([plane], facecolors="#C44448", alpha=.055,
                                    edgecolors="#D8B8BC", linewidths=.8))
for leg, chain in points.items():
    ax3.plot(*chain[:2].T, color=gray, lw=4)
    ax3.plot(*chain[1:3].T, color=thigh, lw=5)
    ax3.plot(*chain[2:].T, color=calf, lw=4)
    ax3.scatter(*chain[1].T, c=thigh, s=36, depthshade=False)
    ax3.scatter(*chain[2].T, c=thigh, s=36, depthshade=False)
    ax3.scatter(*chain[3].T, c=foot, s=65, depthshade=False, edgecolors="white", linewidths=.7)
    offset_y = -.022 if leg.endswith("R") else .022
    ax3.text(chain[3, 0], chain[3, 1] + offset_y, chain[3, 2] - .018, leg,
             ha="center", color=foot, fontsize=11)
ax3.scatter(0, 0, 0, marker="+", s=75, color=ink, depthshade=False)
ax3.text(0, 0, .029, "trunk 原点", color=ink, ha="center", fontsize=10)
ax3.quiver(.27, 0, .025, .075, 0, 0, color=ink, arrow_length_ratio=.2)
ax3.text(.33, 0, .055, "前方 +X", color=ink, fontsize=10, ha="center")
ax3.set(xlim=(-.32, .36), ylim=(-.19, .19), zlim=(z_foot - .05, .09))
ax3.set_box_aspect((.68, .38, .14 - z_foot))
ax3.view_init(elev=22, azim=-53)
ax3.set_proj_type("ortho")
ax3.set_xlabel("X / m", labelpad=8)
ax3.set_ylabel("Y / m", labelpad=8)
ax3.set_zlabel("Z / m", labelpad=8)
ax3.set_xticks([-.3, 0, .3])
ax3.set_yticks([-.15, 0, .15])
ax3.set_zticks([-.2, -.1, 0])
for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
    axis.pane.fill = False
    axis._axinfo["grid"]["color"] = (.82, .86, .9, .45)
ax3.set_title("三维关节连杆示意", color=ink, fontsize=15, pad=22)

def setup_projection(ax, xlabel, xlim):
    ax.scatter([0], [0], marker="+", s=70, c=ink, zorder=6)
    ax.axhline(z_foot, color=foot, lw=.8, ls=":", alpha=.55)
    ax.set(xlim=xlim, ylim=(z_foot - .075, .055), xlabel=xlabel, ylabel="Z / m（相对 trunk）")
    ax.set_aspect("equal")
    ax.set_yticks([-.2, -.1, 0])
    ax.grid(alpha=.13)
    ax.tick_params(labelsize=9)


def draw_leg(ax, chain, coordinate):
    ax.plot(chain[:2, coordinate], chain[:2, 2], color=gray, lw=4)
    ax.plot(chain[1:3, coordinate], chain[1:3, 2], color=thigh, lw=3.8)
    ax.plot(chain[2:, coordinate], chain[2:, 2], color=calf, lw=3.4)
    ax.scatter(chain[1:3, coordinate], chain[1:3, 2], c=thigh, s=33, zorder=4)
    ax.scatter(chain[3, coordinate], chain[3, 2], c=foot, s=46, zorder=5)


setup_projection(side, "X / m（机器人前方 →）", (-.32, .34))
side.plot(frame[:, 0], frame[:, 2], color=gray, lw=6, alpha=.5)
for leg in ("RR", "FR"):
    draw_leg(side, points[leg], 0)
    side.plot([points[leg][0, 0]] * 2, [0, z_foot], color=gray, ls="--", lw=.8, alpha=.65)
side.text(-.2381, .022, "后腿 RR / RL", ha="center", fontsize=10, color=ink)
side.text(.2381, .022, "前腿 FR / FL", ha="center", fontsize=10, color=ink)
side.annotate("", (feet[0, 0], z_foot - .025), (-feet[0, 0], z_foot - .025),
              arrowprops={"arrowstyle": "<->", "color": ink, "lw": 1})
side.text(0, z_foot - .052, f"前后足距 {x_span * 1000:.1f} mm", ha="center", fontsize=10, color=ink)
side.annotate("", (.29, 0), (.29, z_foot), arrowprops={"arrowstyle": "<->", "color": foot, "lw": 1})
side.text(.307, z_foot / 2, f"{-z_foot * 1000:.1f} mm", rotation=90, ha="center", va="center", fontsize=10, color=foot)
side.set_title("侧视 XZ · 左右腿投影重合", color=ink, fontsize=14, pad=16)

setup_projection(front, "Y / m（机器人左侧 →）", (-.205, .205))
front.plot([-.04675, .04675], [0, 0], color=gray, lw=6, alpha=.5)
for leg in ("FR", "FL"):
    draw_leg(front, points[leg], 1)
front.text(-.12675, .023, "右腿 FR / RR", ha="center", fontsize=10, color=ink)
front.text(.12675, .023, "左腿 FL / RL", ha="center", fontsize=10, color=ink)
front.annotate("", (-.12675, z_foot - .025), (.12675, z_foot - .025),
               arrowprops={"arrowstyle": "<->", "color": ink, "lw": 1})
front.text(0, z_foot - .052, f"左右足距 {y_span * 1000:.1f} mm", ha="center", fontsize=10, color=ink)
front.set_title("前视 YZ · 前后腿投影重合", color=ink, fontsize=14, pad=16)

fig.suptitle(f"TQBot2 {pose_title}", fontsize=23, color=ink, y=.974)
fig.text(.5, .928, f"hip = {angle('hip'):.2f} rad，thigh = {angle('thigh'):.2f} rad（{math.degrees(angle('thigh')):.2f}°），"
         f"calf = {angle('calf'):.5f} rad（{math.degrees(angle('calf')):.2f}°）",
         ha="center", fontsize=13, color=ink)
fig.text(.5, .89, f"足端中心 Z = {z_foot:.5f} m   |   各足向机身中部内收 {x_inset * 1000:.1f} mm（相对 hip 的 X 偏移）",
         ha="center", fontsize=12, color="#586C80")
fig.legend(handles=[Line2D([0], [0], color=gray, lw=4, label="hip 轴心参考框 / 横向连接"),
                    Line2D([0], [0], color=thigh, lw=4, label="大腿"),
                    Line2D([0], [0], color=calf, lw=4, label="小腿"),
                    Line2D([0], [0], marker="o", color=foot, ls="", markersize=7, label="足端中心")],
           loc="lower center", bbox_to_anchor=(.5, .074), frameon=False, ncol=4, fontsize=11)
fig.text(.5, .035, "依据当前 TQBot2_walk.urdf 的关节轴与连杆尺寸计算；坐标原点为 trunk。\n"
         "图示为指定关节角的几何姿态，未模拟受载形变；灰框仅连接 hip 轴心，不表示机身外壳。",
         ha="center", fontsize=10.5, color="#586C80", linespacing=1.7)
for ext in ("png", "pdf"):
    fig.savefig(args.output_dir / f"{stem}.{ext}", dpi=180, facecolor="white")
(args.output_dir / "kinematics.json").write_text(json.dumps({
    "urdf": str(urdf), "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
    "pose": args.pose, "joint_pos_rad": joint_pos,
    "coordinate_frame": "trunk link; geometric target, no contact dynamics",
    "point_order": ["hip_axis", "thigh_axis", "knee_axis", "foot_center"],
    "positions_m": {leg: chain.tolist() for leg, chain in points.items()},
    "foot_z_m": float(z_foot), "foot_span_x_m": float(x_span), "foot_span_y_m": float(y_span),
    "inward_foot_x_offset_from_hip_m": float(x_inset),
}, indent=2, ensure_ascii=False) + "\n")
print(args.output_dir / f"{stem}.png")
