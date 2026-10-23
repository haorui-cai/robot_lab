"""Plot recorded world-vertical link/CoM kinematics without crossing reset boundaries."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
args = parser.parse_args()
directory = Path(args.directory).resolve()
data = np.genfromtxt(directory / "vertical_motion.csv", delimiter=",", names=True)
metadata = json.loads((directory / "metadata.json").read_text())
if "xacro_base_z_m" not in data.dtype.names:
    raise ValueError("Recollect with the corrected diagnostic: the older CSV measured trunk, not xacro base")
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "#fbfcfe", "savefig.facecolor": "white"})
colors = dict(base="#1766b2", root="#718096", com="#df7612", left="#238b57", right="#7952b3")


def segments(d):
    return [d[d["episode"] == ep] for ep in np.unique(d["episode"])]


def line(ax, d, key, label, **kwargs):
    for index, part in enumerate(segments(d)):
        ax.plot(part["t_s"], part[key], label=label if index == 0 else None, **kwargs)


def contact_shading(ax, d):
    for part in segments(d):
        mask = part["max_contact_force_N"] < metadata["contact_threshold_N"]
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        for start, end in zip(edges[::2], edges[1::2]):
            ax.axvspan(part["t_s"][start], part["t_s"][min(end, len(part)-1)], color="#f3c45b", alpha=0.20, lw=0)


summaries = []
for env_id in sorted(np.unique(data["env"]).astype(int)):
    d = data[data["env"] == env_id]
    mask = np.zeros(len(d), dtype=bool)
    for ep in np.unique(d["episode"]):
        ids = np.flatnonzero(d["episode"] == ep)
        mask[ids] = d["t_s"][ids] >= d["t_s"][ids[0]] + 0.5
    valid = d[mask & np.isfinite(d["xacro_base_az_m_s2"])]
    assert len(valid) > 0
    # Independent consistency check: position derivative and matching velocity.
    errors = []
    for part in segments(d):
        dz = np.diff(part["xacro_base_z_m"]) / np.diff(part["t_s"])
        errors.extend((dz - 0.5*(part["xacro_base_vz_m_s"][1:]+part["xacro_base_vz_m_s"][:-1])).tolist())
    no_contact = valid["max_contact_force_N"] < 5
    summary = dict(env=int(env_id), command=float(d["cmd_roll_rad_s"][0]), episodes=len(np.unique(d["episode"])),
                   base_z_min=float(valid["xacro_base_z_m"].min()), base_z_max=float(valid["xacro_base_z_m"].max()),
                   base_z_peak_to_peak_m=float(np.ptp(valid["xacro_base_z_m"])),
                   trunk_z_peak_to_peak_m=float(np.ptp(valid["base_link_z_m"])),
                   whole_com_z_peak_to_peak_m=float(np.ptp(valid["whole_com_z_m"])),
                   base_abs_vz_peak=float(np.abs(valid["xacro_base_vz_m_s"]).max()),
                   base_abs_az_peak_5ms=float(np.abs(valid["xacro_base_az_m_s2"]).max()),
                   base_abs_az_peak_20ms=float(np.nanmax(np.abs(valid["xacro_base_az_20ms_m_s2"]))),
                   base_az_rms_5ms=float(np.sqrt(np.mean(valid["xacro_base_az_m_s2"]**2))),
                   com_az_rms_5ms=float(np.sqrt(np.mean(valid["whole_com_az_m_s2"]**2))),
                   no_contact_fraction=float(no_contact.mean()),
                   kinematic_velocity_error_rms=float(np.sqrt(np.mean(np.asarray(errors)**2))))
    summaries.append(summary)

main_id = metadata["commands"].index(-10.0) if -10.0 in metadata["commands"] else 0
d = data[data["env"] == main_id]
main_summary = summaries[main_id]
fig, axs = plt.subplots(4, 1, figsize=(13.8, 11), sharex=True)
fig.subplots_adjust(top=0.87, bottom=0.09, left=0.105, right=0.975, hspace=0.23)
fig.suptitle("蹬地弹跳诊断：xacro base 与 trunk 的 Z 向运动", x=0.105, y=0.975, ha="left", fontsize=21, fontweight="bold")
fig.text(0.105, 0.935, f"模型 4700 · 平地 0° · 滚速命令 −10 rad/s · 偏航命令 0 · 物理采样 200 Hz", color="#465469", fontsize=12)
fig.text(0.105, 0.905, "base 在 trunk 局部 Z 下方 24.7 mm，已随姿态旋转变换；黄色区间为近似离地。", color="#465469", fontsize=11)
line(axs[0], d, "xacro_base_z_m", "xacro base（外壳中心参考点）", color=colors["base"], lw=1.4)
line(axs[0], d, "base_link_z_m", "trunk 原点", color=colors["root"], lw=0.9, alpha=0.75, ls="--")
line(axs[0], d, "whole_com_z_m", "整机质心", color=colors["com"], lw=1.2)
axs[0].set_ylabel("世界 Z 坐标 [m]")
axs[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
axs[0].legend(loc="upper right", ncol=3, fontsize=9)
line(axs[1], d, "xacro_base_vz_m_s", "xacro base", color=colors["base"], lw=1.2)
line(axs[1], d, "whole_com_vz_m_s", "整机质心", color=colors["com"], lw=1.0)
axs[1].set_ylabel("世界 Z 速度 [m/s]")
axs[1].axhline(0, color="#8290a0", lw=0.7)
line(axs[2], d, "xacro_base_az_m_s2", "base：5 ms 原始差分", color="#9aaec5", alpha=0.55, lw=0.65)
line(axs[2], d, "xacro_base_az_20ms_m_s2", "base：20 ms 差分", color=colors["base"], lw=1.0)
line(axs[2], d, "whole_com_az_20ms_m_s2", "整机质心：20 ms 差分", color=colors["com"], lw=1.0)
axs[2].axhline(-9.81, color="#8290a0", ls=":", lw=0.9)
axs[2].set_ylabel("世界 Z 加速度 [m/s²]")
axs[2].legend(loc="upper right", ncol=3, fontsize=8.5)
line(axs[3], d, "left_Fz_N", "左侧接触力", color=colors["left"], lw=1.0)
line(axs[3], d, "right_Fz_N", "右侧接触力", color=colors["right"], lw=0.9)
line(axs[3], d, "total_Fz_N", "全部接触力", color="#3e4958", lw=0.8, alpha=0.75)
weight = sum(metadata["body_masses_kg"][main_id]) * 9.81
axs[3].axhline(weight, color="#8290a0", ls="--", lw=0.9, label="整机重力")
axs[3].set_ylabel("接触力 Fz [N]")
axs[3].set_xlabel("回放时间 [s]")
axs[3].legend(loc="upper right", ncol=4, fontsize=9)
for ax in axs:
    contact_shading(ax, d)
    ax.grid(alpha=0.18)
    ax.set_xlim(0, metadata["duration_s"])
    for ep in np.unique(d["episode"])[1:]:
        ax.axvline(d["t_s"][d["episode"] == ep][0], color="#b94a5b", ls="--", lw=0.9)
fig.text(0.105, 0.031,
         f"统计剔除每回合前 0.5 s：base 高度峰峰值 {1000*main_summary['base_z_peak_to_peak_m']:.1f} mm；"
         f"整机质心 {1000*main_summary['whole_com_z_peak_to_peak_m']:.1f} mm；"
         f"base |az| 峰值 {main_summary['base_abs_az_peak_5ms']:.1f} m/s²（5 ms）。", fontsize=11)
fig.savefig(directory / "base_z_diagnostic.png", dpi=170)
fig.savefig(directory / "base_z_diagnostic.pdf")
plt.close(fig)

# A short interval resolves individual kicks without hiding high-frequency peaks.
fig, axs = plt.subplots(3, 1, figsize=(13.8, 8), sharex=True)
fig.subplots_adjust(top=0.87, bottom=0.1, left=0.10, right=0.98, hspace=0.22)
fig.suptitle("蹬地周期内的 Z 向运动（2～4 s 放大）", x=0.1, y=0.97, ha="left", fontsize=19, fontweight="bold")
fig.text(0.1, 0.91, "蓝：xacro base　橙：整机质心　灰：base 5 ms 原始加速度　黄色：近似离地", color="#465469")
d = d[(d["t_s"] >= 2) & (d["t_s"] <= 4)]
for key, color, label in (("xacro_base_z_m", colors["base"], "base"), ("whole_com_z_m", colors["com"], "整机质心")):
    line(axs[0], d, key, label, color=color, lw=1.5)
axs[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
axs[0].set_ylabel("世界 Z [m]")
line(axs[1], d, "xacro_base_az_m_s2", "5 ms", color="#9aaec5", lw=0.8)
line(axs[1], d, "xacro_base_az_20ms_m_s2", "20 ms", color=colors["base"], lw=1.5)
line(axs[1], d, "whole_com_az_20ms_m_s2", "质心 20 ms", color=colors["com"], lw=1.3)
axs[1].axhline(-9.81, color="#8290a0", ls=":", lw=0.9)
axs[1].set_ylabel("世界 az [m/s²]")
line(axs[2], d, "left_Fz_N", "左侧", color=colors["left"], lw=1.2)
line(axs[2], d, "total_Fz_N", "全部", color="#3e4958", lw=1.0)
axs[2].set_ylabel("接触力 Fz [N]")
axs[2].set_xlabel("回放时间 [s]")
for ax in axs:
    contact_shading(ax, d)
    ax.grid(alpha=0.18)
    ax.set_xlim(2, 4)
fig.savefig(directory / "base_z_zoom.png", dpi=170)
plt.close(fig)
(directory / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
print(json.dumps(summaries, indent=2))
