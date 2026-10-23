"""Exact planar ramps and the fixed walking acceptance rule (no simulator imports)."""
import math
import numpy as np


def ramp_vertices(angle_deg, size=(24.0, 10.0)):
    length, width = size
    xy = np.array([[0., 0.], [length, 0.], [length, width], [0., width]])
    z = (xy[:, 0] - length / 2) * math.tan(math.radians(angle_deg))
    return np.column_stack((xy, z)), np.array([[0, 1, 2], [0, 2, 3]])


def accepted(row):
    """Every failed/short trial remains in the denominator."""
    return bool(
        row['duration_s'] >= 11.9 and not row['failed']
        and row['tracking_rmse'] <= max(.10, .25 * row['command_m_s'])
        and row['forward_m'] >= .75 * row['command_m_s'] * row['measured_s']
        and row['lateral_max_m'] <= .5
        and row['aligned_fraction'] >= .95
        and row['heading_good_fraction'] >= .95
    )


def summarize(rows):
    groups = {}
    for row in rows:
        key = f"{row['slope_deg']:+.0f}deg_v{row['command_m_s']:.2f}_mu{row['friction']:.2f}"
        groups.setdefault(key, []).append(row)
    cells = {}
    for key, group in groups.items():
        cells[key] = {
            'episodes': len(group), 'successes': sum(accepted(r) for r in group),
            'success_rate': float(np.mean([accepted(r) for r in group])),
            'mean_tracking_rmse': float(np.mean([r['tracking_rmse'] for r in group])),
            'mean_forward_m': float(np.mean([r['forward_m'] for r in group])),
            'failure_rate': float(np.mean([r['failed'] for r in group])),
        }
    target = [v for k, v in cells.items() if k.startswith(('+30deg', '-30deg', '+0deg'))]
    return {'cells': cells, 'episodes': len(rows),
            'worst_target_success_rate': min((x['success_rate'] for x in target), default=0.),
            'passed': len(target) == 27 and all(x['success_rate'] >= .95 for x in target)}
