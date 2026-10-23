import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import numpy as np
from tqbot2_cadence_audit import summarize_foot, sensitivity_status


class CadenceTests(unittest.TestCase):
    def synthetic(self, lift=.03):
        t = np.arange(0, 5, .005)
        phase = (t * 2) % 1
        force = np.where(phase < .6, 30., 0.)
        z = -.25 + lift * np.maximum(0, np.sin(np.pi * (phase - .6) / .4)) * (phase >= .6)
        return force, z, np.ones(len(t), dtype=bool)

    def test_true_cycle_and_duty_are_separate(self):
        result = summarize_foot(*self.synthetic())
        self.assertAlmostEqual(result['contact_cycles']['frequency_hz'], 2., places=2)
        self.assertAlmostEqual(result['kinematically_checked_cycles']['frequency_hz'], 2., places=2)
        self.assertAlmostEqual(result['duty_factor_mean'], .6, delta=.02)

    def test_force_bounce_without_lift_is_not_full_swing(self):
        force, z, valid = self.synthetic(lift=0.)
        result = summarize_foot(force, z, valid)
        self.assertIsNotNone(result['contact_cycles']['frequency_hz'])
        self.assertIsNone(result['kinematically_checked_cycles']['frequency_hz'])
        self.assertGreater(result['rejected']['small_excursion'], 0)

    def test_single_physics_sample_dropout_does_not_create_cycle(self):
        force, z, valid = self.synthetic()
        baseline = summarize_foot(force, z, valid)
        force[10::100] = 0.
        changed = summarize_foot(force, z, valid)
        self.assertEqual(baseline['contact_cycles'], changed['contact_cycles'])

    def test_stationary_and_failed_samples_do_not_become_zero_hz_success(self):
        f, z, valid = self.synthetic()
        self.assertIsNone(summarize_foot(np.full_like(f, 30), z, valid)['contact_cycles']['frequency_hz'])
        valid[:] = False
        self.assertIsNone(summarize_foot(f, z, valid)['contact_cycles']['frequency_hz'])

    def test_missing_kinematics_does_not_fabricate_good_cycle(self):
        f, z, valid = self.synthetic()
        z[:] = np.nan
        r = summarize_foot(f, z, valid)
        self.assertIsNotNone(r['contact_cycles']['frequency_hz'])
        self.assertIsNone(r['kinematically_checked_cycles']['frequency_hz'])

    def test_reset_boundary_cannot_become_a_long_stride(self):
        f, z, valid = self.synthetic()
        epoch = np.zeros(len(f), dtype=int)
        epoch[250:] = 1
        r = summarize_foot(f, z, valid, episode=epoch)
        self.assertEqual(r['rejected']['episode_boundary'], 1)
        self.assertAlmostEqual(r['contact_cycles']['frequency_hz'], 2., places=2)

    def test_filter_sensitive_estimate_is_flagged_instead_of_selecting_low_hz(self):
        r = sensitivity_status([3.25, 3.2, 2.48, None])
        self.assertEqual(r['status'], 'filter_sensitive')
        self.assertEqual(r['range_hz'], [2.48, 3.25])
        self.assertEqual(sensitivity_status([None])['status'], 'insufficient_measurements')


if __name__ == '__main__':
    unittest.main()
