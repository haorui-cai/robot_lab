"""Episode-balanced metrics for frozen rolling-policy evaluation (NumPy only)."""
import numpy as np


def flags(row):
    enough = row['seconds'] >= 6.0 and row['cycles'] >= 4
    roll = enough and row['roll_good'] / max(row['cycles'], 1) >= .8
    yaw = enough and row['yaw_good'] / max(row['cycles'], 1) >= .8
    return roll, yaw, roll and yaw


def summarize(episodes, cycles, bootstrap=2000):
    if not episodes:
        return {'episodes': 0}
    f = np.array([flags(r) for r in episodes], dtype=float)
    # Environments carry fixed sampled physics: bootstrap whole environment clusters.
    ids = sorted(set((r.get('seed', 0), r['env']) for r in episodes))
    groups = [np.array([i for i, r in enumerate(episodes)
                       if (r.get('seed', 0), r['env']) == k]) for k in ids]
    rng = np.random.default_rng(12345)
    ci = np.empty((bootstrap, 3))
    for b in range(bootstrap):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        ci[b] = f[indices].mean(0)
    result = {'episodes':len(episodes), 'physics_clusters':len(groups),
              'mean_seconds':float(np.mean([r['seconds'] for r in episodes])),
              'invalid_episode_fraction':float(np.mean([r['seconds'] < 6 or r['cycles'] < 4 for r in episodes])),
              'roll_only_fraction':float(np.mean((f[:,0] == 1) & (f[:,1] == 0))),
              'yaw_only_fraction':float(np.mean((f[:,0] == 0) & (f[:,1] == 1))),
              'neither_fraction':float(np.mean((f[:,0] == 0) & (f[:,1] == 0))),
              'ci_method':'95% percentile bootstrap of whole environment clusters; evaluation uncertainty only'}
    for i, name in enumerate(['roll_success','yaw_success','joint_success']):
        result[name] = float(f[:,i].mean())
        result[name+'_ci95'] = np.quantile(ci[:,i], [.025,.975]).tolist()
    result['roll_gap_percentage_points'] = max(0., (.75-result['roll_success'])*100)
    result['joint_gap_percentage_points'] = max(0., (.70-result['joint_success'])*100)
    valid = [r for r in episodes if r['cycles'] > 0]
    result['valid_episode_yaw_mae_mean'] = float(np.mean([r['yaw_error_sum']/r['cycles'] for r in valid])) if valid else None
    result['valid_episode_coverage'] = len(valid)/len(episodes)
    if cycles:
        ey = np.array([abs(r['yaw_rate']-r['yaw_command']) for r in cycles])
        er = np.array([abs(r['roll_rate']-r['roll_command']) for r in cycles])
        result.update({'cycles':len(cycles), 'cycle_yaw_mae':float(ey.mean()),
                       'cycle_yaw_error_p50_p80_p95':np.quantile(ey,[.5,.8,.95]).tolist(),
                       'cycle_yaw_pass_fraction':float(np.mean(ey <= .06+1e-7)),
                       'cycle_roll_mae':float(er.mean())})
    else:
        result.update({'cycles':0,'cycle_yaw_mae':None,'cycle_yaw_pass_fraction':None})
    reasons = sorted(set(k for r in episodes for k in r['terminations']))
    result['termination_fraction'] = {k:float(np.mean([k in r['terminations'] for r in episodes])) for k in reasons}
    return result


def stratify(episodes, cycles, key, edges=None):
    values = sorted(set(r[key] for r in episodes)) if edges is None else list(zip(edges[:-1], edges[1:]))
    out = {}
    for v in values:
        if edges is None:
            chosen = [r for r in episodes if r[key] == v]
            name = str(v)
        else:
            low, high = v
            chosen = [r for r in episodes if low <= abs(r[key]) < high or
                      (high == edges[-1] and abs(r[key]) == high)]
            name = f'{low}-{high}'
        if not chosen:
            continue
        ids = {(r.get('seed',0),r['env'],r['episode']) for r in chosen}
        cc = [r for r in cycles if (r.get('seed',0),r['env'],r['episode']) in ids]
        out[name] = summarize(chosen,cc,bootstrap=500)
        out[name]['mean_measured_roll'] = float(np.mean([r['roll_rate'] for r in cc])) if cc else None
        out[name]['mean_measured_yaw'] = float(np.mean([r['yaw_rate'] for r in cc])) if cc else None
    return out
