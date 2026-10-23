"""Aggregate frozen rollout records and create a reproducible diagnostic report."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from evaluation_metrics import summarize, stratify

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',required=True)
args=parser.parse_args()
root=Path(args.input).resolve()
groups={}
for path in sorted(root.glob('*/summary.json')):
    info=json.loads(path.read_text())
    job=info['job']; key=f"{job['label']}/{job['profile']}/{job['sampling']}"
    group=groups.setdefault(key,{'episodes':[],'cycles':[],'runs':[]})
    ep=json.loads(path.with_name('episodes.json').read_text())
    cy=json.loads(path.with_name('cycles.json').read_text())
    assert sum(r['cycles'] for r in ep)==len(cy),f'Lost completed cycles: {path}'
    assert len({(r['seed'],r['env'],r['episode']) for r in ep})==len(ep)
    assert info['complete'],f'Incomplete rollout: {path}'
    group['episodes']+=ep;group['cycles']+=cy;group['runs'].append(info)
results={}
for key,g in groups.items():
    s=summarize(g['episodes'],g['cycles'])
    s['by_roll_command']=stratify(g['episodes'],g['cycles'],'roll_command',[6,8,10,12,14])
    s['by_yaw_command']=stratify(g['episodes'],g['cycles'],'yaw_command')
    s['by_slope']=stratify(g['episodes'],g['cycles'],'slope')
    for field in ['executed_action_saturation','policy_mean_saturation','positive_limit_fraction','policy_mean_rms','policy_std']:
        s[field]=np.mean([r[field] for r in g['runs']],axis=0).tolist()
    s['base_world_z_acceleration_rms']=float(np.mean([r['base_world_z_acceleration_rms'] for r in g['runs']]))
    s['mean_absolute_joint_power_W']=float(np.mean([r['mean_absolute_joint_power_W'] for r in g['runs']]))
    s['seeds']=[r['job']['seed'] for r in g['runs']]
    s['checkpoint']=g['runs'][0]['job']['checkpoint']
    results[key]=s
(root/'aggregate.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))

font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(2,3,figsize=(16,9))
colors={'A/current/gaussian':'#64748b','D/current/gaussian':'#2563eb','D/current/mean':'#08916e','C/current/mean':'#d97706'}
labels={'A/current/gaussian':'A 采样','D/current/gaussian':'D 采样','D/current/mean':'D 均值','C/current/mean':'C 均值'}
keys=[k for k in colors if k in results]
ax=axes[0,0]
for i,(metric,title,color) in enumerate([('roll_success','滚速','#4f46e5'),('yaw_success','偏航','#0d9488'),('joint_success','联合','#d97706')]):
    y=np.array([results[k][metric]*100 for k in keys]);interval=np.array([results[k][metric+'_ci95'] for k in keys])*100
    ax.bar(np.arange(len(keys))+(i-1)*.23,y,width=.22,label=title,color=color,
           yerr=np.stack([y-interval[:,0],interval[:,1]-y]),capsize=2)
ax.axhline(70,color='#d97706',ls='--',alpha=.6)
ax.set_xticks(range(len(keys)),[labels[k] for k in keys]);ax.set_ylim(0,105);ax.set_ylabel('达标回合比例（%）')
ax.set_title('① 回合成功率及 95% 评估置信区间');ax.legend(fontsize=9)

ax=axes[0,1]
ax.plot([6,14],[6,14],color='black',ls='--',label='理想响应')
for k in keys:
    bins=results[k]['by_roll_command'];valid=[(b,bins[b]) for b in ['6-8','8-10','10-12','12-14'] if b in bins and bins[b]['mean_measured_roll'] is not None]
    ax.plot([np.mean([float(v) for v in b.split('-')]) for b,r in valid],[-r['mean_measured_roll'] for b,r in valid],'-o',color=colors[k],label=labels[k])
ax.set_xlabel('滚速命令幅值分箱中心（rad/s）');ax.set_ylabel('完整圈实测滚速幅值（rad/s）');ax.set_title('② 是否按命令改变滚速');ax.legend(fontsize=9)

ax=axes[0,2]
for k in keys:
    errors=np.sort([abs(r['yaw_rate']-r['yaw_command']) for r in groups[k]['cycles']])
    if len(errors):ax.plot(errors,np.arange(1,len(errors)+1)/len(errors),color=colors[k],label=labels[k])
ax.axvline(.06,color='#dc2626',ls='--');ax.set_xlim(0,.25);ax.set_ylim(0,1);ax.set_xlabel('单圈平均偏航速度绝对误差（rad/s）');ax.set_ylabel('完整圈累计比例');ax.set_title('③ 所有完整圈的偏航误差分布');ax.legend(fontsize=9)

ax=axes[1,0]
for k in keys:
    fractions=np.sort([r['yaw_good']/r['cycles'] if r['cycles'] else 0 for r in groups[k]['episodes']])
    ax.plot(fractions,np.arange(1,len(fractions)+1)/len(fractions),color=colors[k],label=labels[k])
ax.axvline(.8,color='#dc2626',ls='--');ax.set_xlabel('一个回合内偏航达标圈占比');ax.set_ylabel('回合累计比例');ax.set_title('④ 为什么圈均值好仍可能回合失败');ax.legend(fontsize=9)

ax=axes[1,1]
for k in keys:ax.plot(range(6),np.array(results[k]['executed_action_saturation'])*100,'-o',label=labels[k],color=colors[k])
ax.set_xticks(range(6),['FR hip','FL hip','RR hip','RL hip','FL thigh','RL thigh'],rotation=25)
ax.set_ylim(-3,105);ax.set_ylabel('原始动作超出 ±1 的比例（%）');ax.set_title('⑤ 动作是否退化为限位输出');ax.legend(fontsize=9)

ax=axes[1,2]
probe='D/yaw_probe/mean'
if probe in results:
    bins=results[probe]['by_yaw_command'];pairs=sorted((float(k),v) for k,v in bins.items() if v['mean_measured_yaw'] is not None)
    ax.plot([p[0] for p in pairs],[p[1]['mean_measured_yaw'] for p in pairs],'-o',color='#2563eb',label='D 实测完整圈均值')
    ax.plot([-.5,.5],[-.5,.5],'k--',label='理想响应');ax.legend(fontsize=9)
ax.set_xlabel('偏航命令（rad/s）');ax.set_ylabel('实测偏航角速度（rad/s）');ax.set_title('⑥ 非零偏航响应（超出当前课程）')
for ax in axes.flat:
    ax.grid(alpha=.14);ax.spines[['top','right']].set_visible(False)
fig.suptitle('滚动策略独立评估｜固定检查点、完整回合、统一条件',fontsize=18,fontweight='bold')
fig.text(.05,.018,'当前条件：初速辅助 1、相位 −2±0.1、偏航命令 0、滚速命令 −14～−6、上坡 0～2° / 下坡 0～3°。\n采样＝训练时的高斯动作；均值＝部署时确定性动作。无完整圈记为回合失败，不纳入圈误差均值。置信区间不代表多次独立训练的不确定性。',fontsize=10,color='#475569')
fig.tight_layout(rect=(0,.065,1,.95))
fig.savefig(root/'evaluation_dashboard.png',dpi=160);fig.savefig(root/'evaluation_dashboard.pdf')

def percent(x):return f'{x*100:.1f}%'
lines=['# 固定检查点独立评估报告','',
       '这是独立仿真回放，不更新网络，不推进课程。训练 C、D 继续运行。所有模型在同一当前窄物理随机化配置下评估，奖励统一为 B 形式且偏航权重 10；用任务指标而非奖励比较。', '',
       '| 模型/场景/动作 | 回合数 | 滚速达标 | 偏航达标 | 联合达标（95%区间） | 圈偏航 MAE |',
       '|---|---:|---:|---:|---:|---:|']
for key,s in results.items():
    ci=s['joint_success_ci95'];err=s['cycle_yaw_mae']
    lines.append(f"| {key} | {s['episodes']} | {percent(s['roll_success'])} | {percent(s['yaw_success'])} | {percent(s['joint_success'])} [{percent(ci[0])}, {percent(ci[1])}] | {err if err is None else round(err,5)} |")
lines += ['', '## 课程距离（独立评估估计，不是在线训练批次的精确计数）','']
if 'D/current/gaussian' in results:
    s=results['D/current/gaussian']
    lines += [f"D 滚速成功率距 75% 门槛缺 {s['roll_gap_percentage_points']:.2f} 个百分点，联合成功率距 70% 门槛缺 {s['joint_gap_percentage_points']:.2f} 个百分点。",
              f"只滚速合格、偏航不合格的回合占 {percent(s['roll_only_fraction'])}；只偏航合格、滚速不合格占 {percent(s['yaw_only_fraction'])}；两者都不合格占 {percent(s['neither_fraction'])}。",
              '这估计单个冻结模型在当前分布下的能力；真正晋级还要求在线两个有效批次连续通过。当前日志没有保存在线批次达标率，不能追溯恢复。']
lines += ['', '## 统计与复现','',
          '- 每个环境恰好取前两个完整回合，避免短失败回合被重复计数。所有失败回合留在分母。',
          '- 每个完整滚动圈记录均值误差；另外计算等回合权重的平均误差，明确区分圈与回合分母。',
          '- 回合需 ≥6 秒且 ≥4 圈；至少 80% 圈滚速相对误差 ≤20% 才滚速合格，至少 80% 圈偏航绝对误差 ≤0.06 才偏航合格。',
          '- 95% 区间按环境整组重采样，保留同一固定物理参数下两回合的相关性。三评估随机种子不是三次独立训练。',
          '- startup 质量、质心、材料与 reset 电机增益、观察噪声保留当前窄随机化；质心每次从名义值重新随机，避免重复累加。',
          '- 当前场景：20% 平地、40% 0～2°上坡、40% 0～3°下坡；其他泛化场景见方法文档。',
          '- 只缩小 GPU 分配容量，不改时间步、接触模型、求解器参数。运行日志必须无接触缓冲溢出。',
          '- 原始逐圈、逐回合 JSON、动作裁剪统计、物理质量/质心、稀疏时序、有效配置及模型 SHA256 均保存在各子目录。',
          '- Z 加速度是修正 base 点的世界 Z 速度在控制步长 20ms 上差分，排除跨重置；不是 5ms 冲击峰值或 IMU 比力。',
          '- 单次评估不能证明奖励改动的因果效果；A→D 是续训前后对照，C 与 D 的初始化不同。',
          '', '## 固定检查点','']
for label in sorted(set(k.split('/')[0] for k in results)):
    item=next(s for k,s in results.items() if k.startswith(label+'/'))
    lines.append(f"- {label}: `{item['checkpoint']}`")
(root/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:{field:v[field] for field in ['episodes','roll_success','yaw_success','joint_success','cycle_yaw_mae']} for k,v in results.items()},indent=2))
