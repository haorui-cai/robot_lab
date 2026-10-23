"""Contact cadence with hysteresis, confirmation and separate kinematic checks.

No merging of nonconsecutive accepted cycles; no-event frequency is unavailable.
The kinematic check is a diagnostic, not a proof of a visually natural gait.
"""
import numpy as np


def confirmed_contact(force, dt=.005, on=3., off=1., confirmation=.02):
    force = np.asarray(force, dtype=float)
    if force.ndim != 1 or not np.isfinite(force).all() or not 0 <= off < on or dt <= 0:
        raise ValueError('Finite scalar force, dt>0 and 0<=off<on required')
    required = max(1, int(np.ceil(confirmation / dt - 1e-10)))
    state = bool(force[0] >= on)
    result = np.full(len(force), state, dtype=bool)
    count = 0
    for i, value in enumerate(force):
        transition = value < off if state else value >= on
        count = count + 1 if transition else 0
        if count >= required:
            state = not state
            result[i - required + 1:i + 1] = state
            count = 0
        result[i] = state
    return result


def summarize_foot(force, body_z, eligible, dt=.005, on=3., off=1., confirmation=.02,
                   minimum_air=.04, minimum_stance=.04, minimum_excursion=.01, episode=None):
    contact = confirmed_contact(force, dt, on, off, confirmation)
    body_z = np.asarray(body_z, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if not (body_z.shape == eligible.shape == contact.shape):
        raise ValueError('Shape mismatch')
    episode = np.zeros(len(contact), dtype=int) if episode is None else np.asarray(episode)
    if episode.shape != contact.shape:
        raise ValueError('Episode labels must match physics samples')
    touchdowns = np.flatnonzero(np.diff(contact.astype(int)) == 1) + 1
    periods, checked_periods, duty, excursions = [], [], [], []
    rejected = {'ineligible': 0, 'episode_boundary': 0, 'multiple_swing_segments': 0, 'short_phase': 0, 'small_excursion': 0, 'missing_kinematics': 0}
    for start, end in zip(touchdowns[:-1], touchdowns[1:]):
        if np.any(episode[start:end + 1] != episode[start]):
            rejected['episode_boundary'] += 1
            continue
        if not eligible[start:end + 1].all():
            rejected['ineligible'] += 1
            continue
        period = (end - start) * dt
        periods.append(period)
        duty.append(float(contact[start:end].mean()))
        air = np.flatnonzero(~contact[start:end]) + start
        if not len(air) or np.any(np.diff(air) != 1):
            rejected['multiple_swing_segments'] += 1
            continue
        if len(air) * dt < minimum_air - 1e-10 or (end-start-len(air))*dt < minimum_stance - 1e-10:
            rejected['short_phase'] += 1
            continue
        values = body_z[air[0] - 1:end + 1]
        if not np.isfinite(values).all():
            rejected['missing_kinematics'] += 1
            continue
        excursion = float(body_z[air].max() - max(body_z[air[0] - 1], body_z[end]))
        excursions.append(excursion)
        if excursion < minimum_excursion:
            rejected['small_excursion'] += 1
            continue
        checked_periods.append(period)

    def stats(values):
        a = np.asarray(values)
        return {'cycles': len(values), 'frequency_hz': float(1/a.mean()) if len(a) else None,
                'period_cv': float(a.std()/a.mean()) if len(a) >= 3 else None,
                'periods_s': values}
    return {'contact_cycles': stats(periods), 'kinematically_checked_cycles': stats(checked_periods),
            'checked_fraction': len(checked_periods)/len(periods) if periods else None,
            'duty_factor_mean': float(np.mean(duty)) if duty else None,
            'excursion_median_m': float(np.median(excursions)) if excursions else None,
            'rejected': rejected}


def sensitivity_status(frequencies, tolerance=.10):
    """Expose the full diagnostic envelope; never select the lowest Hz as truth."""
    available = [float(v) for v in frequencies if v is not None and np.isfinite(v)]
    if len(available) < 2:
        return {'status': 'insufficient_measurements', 'relative_span': None}
    relative = (max(available) - min(available)) / np.median(available)
    return {'status': 'filter_sensitive' if relative > tolerance else 'consistent_across_settings',
            'relative_span': float(relative), 'range_hz': [min(available), max(available)],
            'threshold_is_proposed_diagnostic': True}
