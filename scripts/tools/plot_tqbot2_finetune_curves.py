"""Export auditable TensorBoard trends for the unchanged high-friction finetune profile."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LABELS = {
    "Train/mean_reward": "总回报（episode 均值）",
    "Train/mean_episode_length": "平均 episode 长度（控制步）",
    "Curriculum/terrain_levels": "全部地形平均课程等级（0–9）",
    "Metrics/base_velocity/error_vel_xy": "平面线速度误差指标（m/s，越低越好）",
    "Metrics/base_velocity/error_vel_yaw": "偏航角速度误差指标（rad/s，越低越好）",
    "lin_vel_z_l2": "竖直速度惩罚", "ang_vel_xy_l2": "横滚/俯仰角速度惩罚",
    "joint_torques_l2": "关节力矩平方惩罚", "joint_acc_l2": "关节加速度惩罚",
    "joint_pos_limits": "关节位置限位惩罚", "joint_power": "关节机械功率惩罚",
    "stand_still": "静止命令关节偏离惩罚", "joint_pos_penalty": "关节姿态偏离惩罚",
    "joint_mirror": "关节对称性惩罚", "action_rate_l2": "动作变化惩罚",
    "undesired_contacts": "非期望身体接触惩罚", "contact_forces": "足端大接触力惩罚",
    "track_lin_vel_xy_exp": "线速度跟踪奖励", "track_ang_vel_z_exp": "偏航速度跟踪奖励",
    "feet_air_time": "足端腾空时间项", "feet_air_time_variance": "腾空时间差异惩罚",
    "feet_gait": "对角步态奖励", "feet_contact_without_cmd": "无运动命令接地奖励",
    "feet_slide": "足端地面滑移惩罚", "upward": "机身正立奖励",
    "feet_swing_clearance": "摆动足底低间隙惩罚", "recovery_effort": "恢复力矩惩罚",
    "recovery_torque_change": "恢复力矩变化惩罚", "recovery_joint_velocity": "恢复关节速度惩罚",
    "walking_energy_efficiency": "行走距离能耗代理奖励", "walking_joule_proxy": "行走铜耗代理惩罚",
}
COLORS = {"pre_interrupt": "#1864ab", "resumed": "#d65c17"}


def load_run(root, relative):
    meta = json.loads((root / relative / "launch_metadata.json").read_text())
    accumulator = EventAccumulator(meta["run_directory"], size_guidance={"scalars": 0})
    accumulator.Reload()
    selected = [tag for tag in accumulator.Tags()["scalars"]
                if tag.startswith("Episode_Reward/") or tag in LABELS]
    data = {}
    for tag in selected:
        events = accumulator.Scalars(tag)
        x = np.array([e.step for e in events], dtype=int)
        y = np.array([e.value for e in events], dtype=float)
        wall = np.array([e.wall_time for e in events], dtype=float)
        assert np.isfinite(y).all(), tag
        assert (np.diff(x) > 0).all(), f"Duplicate or unsorted steps: {tag}"
        data[tag] = (x, y, wall)
    return meta, data


def trailing_mean(values, width=100):
    sums = np.concatenate(([0.], np.cumsum(values)))
    starts = np.maximum(np.arange(len(values)) + 1 - width, 0)
    return (sums[np.arange(len(values)) + 1] - sums[starts]) / (np.arange(len(values)) + 1 - starts)


def label(tag):
    return LABELS.get(tag, LABELS[tag.split("/")[-1]] if tag.split("/")[-1] in LABELS else tag)


def decorate(ax, tag):
    ax.set_title(label(tag), fontsize=12, loc="left", pad=9)
    ax.set_xlabel("高摩擦配置阶段的迭代编号")
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    if tag.startswith("Episode_Reward/"):
        ax.set_ylabel("已加权分项奖励")
    ax.margins(x=.01)


def whole_plot(ax, tag, runs):
    for name, data in runs.items():
        x, y, _ = data[tag]
        color = COLORS[name]
        ax.plot(x, y, color=color, alpha=.17, linewidth=.65)
        ax.plot(x, trailing_mean(y), color=color, linewidth=1.8)
    ax.axvspan(3000, 3072, color="#d9dde3", alpha=.6)
    ax.axvline(3000, color="#68717d", linestyle="--", linewidth=.9)
    ax.axvline(3072, color="#68717d", linestyle=":", linewidth=.9)
    decorate(ax, tag)


def footer(fig):
    fig.text(.055, .034, "蓝：磁盘中断前 100–3072；橙：恢复后 3000–4499。细线为原始值，粗线为100轮尾随均值。", fontsize=10)
    fig.text(.055, .014, "灰带：3000–3072重叠记录；未保存72轮未继承。恢复启动时环境/统计重建。负奖励趋近0表示惩罚减小。", fontsize=9, color="#4c5561")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white"})
    inputs = {"pre_interrupt": "outputs/TQBot2_walk_flat_mu2p5_20261012",
              "resumed": "outputs/TQBot2_walk_flat_mu2p5_resume3000_20261013"}
    metadata, runs, raw_runs, incomplete_steps = {}, {}, {}, {}
    for name, relative in inputs.items():
        metadata[name], raw_runs[name] = load_run(args.root, relative)
        raw = raw_runs[name]
        common = set.intersection(*(set(values[0]) for values in raw.values()))
        union = set.union(*(set(values[0]) for values in raw.values()))
        incomplete_steps[name] = sorted(int(step) for step in union - common)
        runs[name] = {}
        for tag, (x, y, wall) in raw.items():
            mask = np.isin(x, list(common))
            runs[name][tag] = (x[mask], y[mask], wall[mask])
    assert set(runs["pre_interrupt"]) == set(runs["resumed"])
    rewards = [tag for tag in runs["resumed"] if tag.startswith("Episode_Reward/")]
    assert len(rewards) == 26
    tags = list(runs["resumed"])
    for name, data in raw_runs.items():
        steps = sorted(set.union(*(set(v[0]) for v in data.values())))
        lookup = {tag: {int(step): (float(value), float(wall)) for step, value, wall in zip(*data[tag])}
                  for tag in tags}
        with (args.output / f"{name}_scalars.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["iteration", "wall_time_unix", *tags])
            for step in steps:
                wall = next(lookup[t][step][1] for t in tags if step in lookup[t])
                writer.writerow([int(step), wall, *[lookup[t][step][0] if step in lookup[t] else "" for t in tags]])
    windows = {"pre_interrupt": ((2073, 2572), (2573, 3072)),
               "resumed": ((3500, 3999), (4000, 4499))}
    statistics = []
    for name, data in runs.items():
        for tag, (x, y, _) in data.items():
            means, counts = [], []
            for low, high in windows[name]:
                selected = y[(x >= low) & (x <= high)]
                assert len(selected) == 500
                means.append(float(selected.mean())); counts.append(len(selected))
            low, high = windows[name][-1]
            mask = (x >= low) & (x <= high)
            slope = float(np.polyfit(x[mask] - low, y[mask], 1)[0] * 1000)
            statistics.append({"run": name, "tag": tag, "label": label(tag), "previous_mean": means[0],
                               "last_mean": means[1], "delta": means[1] - means[0],
                               "relative_change_percent": 100 * (means[1] - means[0]) / abs(means[0]) if means[0] else None,
                               "last_500_slope_per_1000": slope,
                               "better_direction": "lower" if tag.startswith("Metrics/") else "higher",
                               "previous_window": windows[name][0], "last_window": windows[name][1]})
    (args.output / "tail_statistics.json").write_text(json.dumps(statistics, indent=2, ensure_ascii=False) + "\n")
    with (args.output / "tail_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(statistics[0]))
        writer.writeheader(); writer.writerows(statistics)
    manifest = {"inputs": {name: meta["run_directory"] for name, meta in metadata.items()},
                "scope": "Unchanged high-friction finetune profile; earlier reward changes excluded",
                "smoothing": "100-iteration trailing mean within each run separately; no stitching at restart",
                "windows": windows, "reward_count": len(rewards), "incomplete_scalar_steps": incomplete_steps,
                "caveats": ["No per-terrain curriculum states recorded", "Training curves are not held-out evaluations",
                            "Command errors are Isaac command metrics, not RMSE",
                            "Reward terms already include weights and episode-duration normalization",
                            "Earlier unsaved iterations 3001-3072 are shown separately and not inherited"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    overview = ["Train/mean_reward", "Curriculum/terrain_levels", "Metrics/base_velocity/error_vel_xy",
                "Metrics/base_velocity/error_vel_yaw", "Episode_Reward/walking_energy_efficiency", "Episode_Reward/feet_swing_clearance"]
    with PdfPages(args.output / "TQBot2_finetune_curves.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
        for ax, tag in zip(axes.flat, overview): whole_plot(ax, tag, runs)
        fig.suptitle("TQBot2 续训总览｜相同高摩擦平地配置", fontsize=19, x=.055, ha="left", y=.98)
        fig.subplots_adjust(left=.06, right=.975, top=.90, bottom=.14, hspace=.47, wspace=.28)
        footer(fig)
        fig.savefig(args.output / "overview.png", dpi=150); pdf.savefig(fig); plt.close(fig)
        for page, selected in enumerate((rewards[:13], rewards[13:]), 1):
            fig, axes = plt.subplots(4, 4, figsize=(18, 13))
            for ax, tag in zip(axes.flat, selected):
                whole_plot(ax, tag, runs)
                ax.set_title(label(tag), fontsize=11, loc="left")
                ax.text(.02, .98, tag.split("/")[-1], transform=ax.transAxes, fontsize=8, va="top", color="#5b6470")
            for ax in list(axes.flat)[len(selected):]: ax.set_visible(False)
            fig.suptitle(f"TQBot2 全部分项奖励｜{page}/2", fontsize=19, x=.055, ha="left", y=.98)
            fig.subplots_adjust(left=.055, right=.975, top=.925, bottom=.11, hspace=.65, wspace=.38)
            footer(fig)
            fig.savefig(args.output / f"rewards_{page}.png", dpi=150); pdf.savefig(fig); plt.close(fig)
        final_tags = overview[:4]
        fig, axes = plt.subplots(4, 2, figsize=(15, 13))
        for col, (name, data) in enumerate(runs.items()):
            start, end = windows[name][0][0], windows[name][1][1]
            for row, tag in enumerate(final_tags):
                ax = axes[row, col]; x, y, _ = data[tag]; mask = (x >= start) & (x <= end)
                ax.plot(x[mask], y[mask], color=COLORS[name], alpha=.22, linewidth=.7)
                ax.plot(x[mask], trailing_mean(y[mask], 50), color=COLORS[name], linewidth=1.8)
                ax.axvspan(windows[name][1][0], end, color="#f0ede5", alpha=.7)
                stat = next(s for s in statistics if s["run"] == name and s["tag"] == tag)
                for window, mean in zip(windows[name], [stat["previous_mean"], stat["last_mean"]]):
                    ax.hlines(mean, window[0], window[1], color="#374151", linestyles="--", linewidth=1.1)
                decorate(ax, tag)
                ax.text(.02, .96, f"500轮均值：{stat['previous_mean']:.4f} → {stat['last_mean']:.4f}",
                        transform=ax.transAxes, va="top", fontsize=10,
                        bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        fig.suptitle("末段是否仍在改善？｜左右分别比较各自最后两个500轮窗口", fontsize=18, x=.06, ha="left", y=.988)
        axes[0, 0].text(.5, 1.24, "磁盘中断前：2073–3072", transform=axes[0, 0].transAxes, ha="center", fontsize=12)
        axes[0, 1].text(.5, 1.24, "恢复后结束前：3500–4499", transform=axes[0, 1].transAxes, ha="center", fontsize=12)
        fig.subplots_adjust(left=.075, right=.975, top=.915, bottom=.08, hspace=.55, wspace=.25)
        fig.text(.075, .026, "细线：原始值；粗线：50轮尾随均值；虚线：500轮窗口均值。右半背景为最后500轮。比较均值不等同于统计显著性检验。", fontsize=9)
        fig.savefig(args.output / "tail_comparison.png", dpi=150); pdf.savefig(fig); plt.close(fig)
    selected_tags = overview + ["Episode_Reward/feet_slide", "Episode_Reward/recovery_effort",
                                "Episode_Reward/recovery_torque_change", "Episode_Reward/recovery_joint_velocity"]
    report = ["# TQBot2 续训曲线与末段趋势（2026-10-13）", "",
              "结论：末段整体接近平台。总奖励小幅增加，不等于所有分项继续改善；课程与跟踪误差未持续改善。",
              "范围：仅比较相同高摩擦平地配置的100–3072与3000–4499。更早的奖励结构变化不拼接；恢复时环境和统计重建。",
              "误差为Isaac记录的命令时间归一化指标，不是固定测试RMSE。课程为全部环境均值，无法定位具体地形。",
              f"完整迭代分析排除不完整记录：{incomplete_steps}。原始CSV保留这些部分记录，缺项为空，不插值。", ""]
    for name in runs:
        report.extend([f"## {'磁盘中断前' if name == 'pre_interrupt' else '最终结束前'}", "",
                       f"窗口：{windows[name][0]} 与 {windows[name][1]}，各500轮。", "",
                       "| 指标 | 前500轮均值 | 后500轮均值 | 相对变化 |", "|---|---:|---:|---:|"])
        for tag in selected_tags:
            stat = next(s for s in statistics if s["run"] == name and s["tag"] == tag)
            report.append(f"| {label(tag)} | {stat['previous_mean']:.6f} | {stat['last_mean']:.6f} | {stat['relative_change_percent']:+.2f}% |")
        report.append("")
    report += ["## 解读", "", "最终末段恢复力矩/变化/速度惩罚趋近0约8.55%/4.68%/6.14%。这也受倒地持续时间与恢复门控影响，不能据此断言恢复峰值力矩已降低。",
               "末段滑移与摆动间隙惩罚变得稍负，能耗代理奖励略降。末段仍有少量收益，但证据不支持课程和跟踪误差持续明显改善。",
               "全部26项奖励的窗口统计见tail_statistics.csv；原始逐轮数据见pre_interrupt_scalars.csv和resumed_scalars.csv。", ""]
    (args.output / "report.md").write_text("\n".join(report))
    print(json.dumps({"output": str(args.output.resolve()), "reward_terms": len(rewards), "figures": 4, "pdf_pages": 4}))


if __name__ == "__main__":
    main()
