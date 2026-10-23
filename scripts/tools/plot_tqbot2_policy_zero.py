"""Plot policy zeros and standing references against the actual walking URDF limits."""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/TQBot2_policy_zero_decoupling_20261014")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
stance_path = ROOT / "source/robot_lab/robot_lab/assets/tqbot_walk_stance.py"
spec = importlib.util.spec_from_file_location("stance", stance_path)
stance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stance)
urdf_path = ROOT / "source/robot_lab/data/Robots/TQBot/TQBot2_description/urdf/TQBot2_walk.urdf"
urdf = ET.parse(urdf_path).getroot()
asset_tree = ast.parse((ROOT / "source/robot_lab/robot_lab/assets/tqbot.py").read_text())
asset_cfg = next(node.value for node in asset_tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "TQBOT2_CFG" for t in node.targets))
soft_factor = ast.literal_eval(next(k.value for k in asset_cfg.keywords if k.arg == "soft_joint_pos_limit_factor"))


def angle(mapping, name):
    matches = [value for pattern, value in mapping.items() if re.fullmatch(pattern, name)]
    assert len(matches) == 1
    return matches[0]


records = []
for kind in ("hip", "calf", "thigh"):
    joints = [j for j in urdf.findall("joint") if j.get("name", "").endswith(f"_{kind}_joint")]
    assert len(joints) == 4
    limits = {(float(j.find("limit").get("lower")), float(j.find("limit").get("upper"))) for j in joints}
    assert len(limits) == 1
    low, high = limits.pop()
    middle = (low + high) / 2
    soft_low, soft_high = middle - (high - low) * soft_factor / 2, middle + (high - low) * soft_factor / 2
    if kind == "thigh":
        soft_low, soft_high = stance.WALK_THIGH_SOFT_LIMITS
    zero = angle(stance.WALK_POLICY_ZERO_JOINT_POS, joints[0].get("name"))
    standing = angle(stance.WALK_STANDING_JOINT_POS, joints[0].get("name"))
    assert all(angle(stance.WALK_POLICY_ZERO_JOINT_POS, j.get("name")) == zero for j in joints)
    records.append(dict(joint=kind, hard_limits_rad=[low, high], soft_limits_rad=[soft_low, soft_high],
                        midpoint_rad=middle, policy_zero_rad=zero, standing_rad=standing,
                        policy_zero_percent_from_hard_lower=100 * (zero - low) / (high - low),
                        policy_zero_distance_to_hard_limits_rad=[zero - low, high - zero]))

plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 11})
fig, axes = plt.subplots(3, 1, figsize=(13.2, 10.7), facecolor="white")
blue, green, orange, ink, gray = "#2267AC", "#3A9872", "#C77622", "#203348", "#718096"
for ax, rec, title in zip(axes, records, ("HIP / 髋关节", "CALF / 小腿关节", "THIGH / 大腿关节")):
    low, high = rec["hard_limits_rad"]
    soft_low, soft_high = rec["soft_limits_rad"]
    zero, standing = rec["policy_zero_rad"], rec["standing_rad"]
    fraction = lambda q: (q - low) / (high - low)
    ax.set_xlim(-.04, 1.04)
    ax.set_ylim(-.12, 1.25)
    ax.axis("off")
    ax.text(0, 1.15, title, color=ink, weight="bold", fontsize=14, ha="left")
    ax.text(1, 1.15, f"四腿数值范围相同  ·  零位在硬范围的 {rec['policy_zero_percent_from_hard_lower']:.2f}%", ha="right", color=gray)
    ax.text(0, .96, f"硬限位 [{low:.4f}, {high:.4f}] rad   |   软限位 [{soft_low:.5f}, {soft_high:.5f}] rad", color=gray, fontsize=10.5)
    y = .43
    ax.plot([0, 1], [y, y], color="#BDD0E3", lw=14, solid_capstyle="butt")
    ax.plot([fraction(soft_low), fraction(soft_high)], [y, y], color=green, lw=7, solid_capstyle="butt")
    for edge in (0, 1):
        ax.plot([edge, edge], [y - .06, y + .06], color=blue, lw=2)
    ax.plot([.5, .5], [y - .13, y + .17], color=gray, ls=":", lw=1.4)
    ax.scatter([fraction(zero)], [y], marker="D", s=100, c=blue, edgecolors="white", linewidths=1.2, zorder=5)
    if abs(zero - standing) < 1e-10:
        ax.text(fraction(zero), .64, f"策略零位 = 站立参考\n{zero:.5f} rad（{math.degrees(zero):.2f}°）", ha="center", color=blue, fontsize=12)
    else:
        ax.scatter([fraction(standing)], [y], marker="o", s=90, c=orange, edgecolors="white", linewidths=1, zorder=5)
        ax.text(fraction(zero), .64, f"策略零位\n{zero:.2f} rad（{math.degrees(zero):.2f}°）", ha="center", color=blue, fontsize=12)
        ax.text(.015, .64, f"站立参考\n{standing:.2f} rad（{math.degrees(standing):.2f}°）", ha="left", color=orange, fontsize=12)
    ax.text(0, .22, f"硬下限 {low:.4f}\n{math.degrees(low):.2f}°", ha="left", va="top", color=ink, fontsize=10.5)
    ax.text(1, .22, f"硬上限 {high:.4f}\n{math.degrees(high):.2f}°", ha="right", va="top", color=ink, fontsize=10.5)
    ax.text(.5, .19, f"范围中点 {rec['midpoint_rad']:.4f} rad", ha="center", va="top", color=gray, fontsize=10.5)
    left, right = rec["policy_zero_distance_to_hard_limits_rad"]
    ax.text(.5, -.045, f"策略零位 → 硬下限：{left:.4f} rad     |     策略零位 → 硬上限：{right:.4f} rad", ha="center", va="top", color=gray, fontsize=10)

fig.suptitle("TQBot2：策略零位与站立参考分离", fontsize=21, color=ink, y=.977)
fig.text(.5, .938, "策略零位：action = 0 的目标角；站立参考：复位、姿态惩罚及足高目标的基准", ha="center", color=gray, fontsize=12)
fig.legend(handles=[Line2D([0], [0], color="#BDD0E3", lw=8, label="硬限位范围"),
                    Line2D([0], [0], color=green, lw=5, label="软限位（大腿单独配置）"),
                    Line2D([0], [0], marker="D", color=blue, ls="", markersize=8, label="策略零位"),
                    Line2D([0], [0], marker="o", color=orange, ls="", markersize=8, label="站立参考（与零位不同时）")],
           loc="lower center", bbox_to_anchor=(.5, .047), ncol=4, frameon=False, fontsize=10.5)
fig.text(.5, .026, "每行按各自硬范围归一化显示位置；行间长度不表示相同弧度。角度按 URDF 关节坐标，非世界坐标轨迹。", ha="center", color=gray, fontsize=10)
fig.subplots_adjust(top=.90, bottom=.105, left=.075, right=.94, hspace=.22)
for ext in ("png", "pdf"):
    fig.savefig(args.output_dir / f"joint_ranges_and_policy_zero.{ext}", dpi=180, facecolor="white")
(args.output_dir / "joint_ranges.json").write_text(json.dumps({
    "urdf": str(urdf_path), "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
    "soft_joint_pos_limit_factor": soft_factor, "joints": records,
}, ensure_ascii=False, indent=2) + "\n")
print(args.output_dir / "joint_ranges_and_policy_zero.png")
