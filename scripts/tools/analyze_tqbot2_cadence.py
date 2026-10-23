"""Offline physics-rate cadence report; NumPy only, never loads a policy or simulator."""
import argparse
import json
from pathlib import Path
import numpy as np
from tqbot2_cadence_audit import summarize_foot, sensitivity_status


def analyze(path, discard=2.):
    with np.load(path, allow_pickle=False) as data:
        t, force, height = data['t'], data['foot_force'], data['foot_z_body']
        velocity, upright = data['velocity_b'], data['upright']
        episode = data['episode'] if 'episode' in data else np.zeros(force.shape[:2], dtype=int)
        command = data['command'] if 'command' in data else None
        if force.ndim != 3 or force.shape[-1] != 4 or height.shape != force.shape:
            raise ValueError('Expected [physics samples, environments, four feet] force/height')
        dt = float(np.median(np.diff(t)))
        if dt <= 0 or not np.allclose(np.diff(t), dt, atol=1e-7):
            raise ValueError('Uniform physics timestamps required; do not interpolate control samples')
        if upright.shape != force.shape[:2] or velocity.shape != (*force.shape[:2], 3):
            raise ValueError('Velocity/upright shape mismatch')
        speed = np.linalg.norm(velocity[..., :2], axis=-1)
        eligible = (t[:, None] >= discard) & (upright > np.cos(np.pi/6)) & (speed > .05)
        finite = np.isfinite(force).all(-1) & np.isfinite(velocity).all(-1) & np.isfinite(upright)
        eligible &= finite
        settings, frequencies = [], []
        for on, off in ((1., .5), (3., 1.), (5., 2.)):
            for confirmation in (.02, .04):
                groups = {}
                for label, lower, upper in (('low', .05, .25), ('medium', .25, .5), ('high', .5, float('inf'))):
                    mask = eligible & (speed >= lower) & (speed < upper)
                    feet = [summarize_foot(force[:, e, f], height[:, e, f], mask[:, e], dt,
                                          on, off, confirmation, episode=episode[:, e])
                            for e in range(force.shape[1]) for f in range(4)]
                    contact = [p for foot in feet for p in foot['contact_cycles']['periods_s']]
                    checked = [p for foot in feet for p in foot['kinematically_checked_cycles']['periods_s']]
                    groups[label] = {'eligible_fraction': float(mask.mean()), 'contact_cycles': len(contact),
                                     'contact_hz': float(1/np.mean(contact)) if contact else None,
                                     'checked_cycles': len(checked), 'checked_hz': float(1/np.mean(checked)) if checked else None,
                                     'per_environment_foot': feet}
                checked = [p for g in groups.values() for f in g['per_environment_foot']
                           for p in f['kinematically_checked_cycles']['periods_s']]
                frequencies.append(float(1/np.mean(checked)) if checked else None)
                settings.append({'on_N': on, 'off_N': off, 'confirmation_s': confirmation, 'speed_groups': groups})
        report = {'input': str(path), 'physics_dt': dt, 'discard_s': discard,
                  'eligible_fraction': float(eligible.mean()), 'nonfinite_fraction': float((~finite).mean()),
                  'tilt_over_60deg_fraction': float((upright < .5).mean()),
                  'mean_velocity_b': np.mean(velocity[t >= discard], axis=(0,1)).tolist(),
                  'sensitivity': sensitivity_status(frequencies), 'settings': settings,
                  'interpretation': 'Checked cycles require >=40ms stance/air and >=1cm body-relative foot excursion; diagnostic thresholds, not training rewards. No events means unavailable, never zero-Hz success. Speed bins use actual horizontal speed. Compare identical eligible environments and include failures separately.'}
        if command is not None:
            report['velocity_rmse_m_s'] = float(np.sqrt(np.mean((velocity[t >= discard, :, :2]-command[t >= discard, :, :2])**2)))
        if 'foot_tangent_speed' in data:
            slip = data['foot_tangent_speed']
            support = (force > 3.) & np.isfinite(slip) & (t[:, None, None] >= discard)
            report['support_slip_mean_m_s_on3N'] = float(np.mean(slip[support])) if support.any() else None
            report['support_slip_valid_samples'] = int(support.sum())
        report['per_environment_failure_diagnostics'] = [
            {'environment':e, 'tilt_over_60deg_any':bool((upright[:, e] < .5).any()),
             'nonfinite_any':bool((~finite[:, e]).any()), 'eligible_fraction':float(eligible[:, e].mean())}
            for e in range(force.shape[1])]
        if 'root_pos' in data:
            report['root_displacement_world_m'] = (data['root_pos'][-1]-data['root_pos'][0]).tolist()
        if 'local_slope_deg' in data:
            slope = data['local_slope_deg']
            report['slope_diagnostics'] = [
                {'environment':e, 'sampled_foot_seconds_above_3deg':float(np.sum(np.isfinite(slope[:, e]) & (slope[:, e] > 3.))*dt/4),
                 'maximum_sampled_local_slope_deg':float(np.nanmax(slope[:, e])) if np.isfinite(slope[:, e]).any() else None}
                for e in range(force.shape[1])]
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--discard', type=float, default=2.)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.trace, args.discard), indent=2, allow_nan=False)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
