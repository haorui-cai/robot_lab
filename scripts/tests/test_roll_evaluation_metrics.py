"""Guard against misleading means and episode/physics weighting in evaluation."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'roll'))
from evaluation_metrics import summarize, flags


def episode(env=0,cycles=10,yaw_good=8,roll_good=8,seconds=10,error_sum=.42):
    return dict(env=env,episode=1,seed=7,seconds=seconds,cycles=cycles,
                yaw_good=yaw_good,roll_good=roll_good,yaw_error_sum=error_sum,terminations=['time_out'])


class EvaluationMetricsTest(unittest.TestCase):
    def test_same_mean_does_not_imply_same_success(self):
        self.assertEqual(flags(episode(yaw_good=7)),(True,False,False))
        self.assertEqual(flags(episode(yaw_good=10)),(True,True,True))

    def test_invalid_episode_is_failure_not_zero_error(self):
        out=summarize([episode(),episode(env=1,cycles=0,yaw_good=0,roll_good=0,seconds=3,error_sum=0)],[],100)
        self.assertEqual(out['joint_success'],.5)
        self.assertEqual(out['valid_episode_coverage'],.5)
        self.assertAlmostEqual(out['valid_episode_yaw_mae_mean'],.042)
        self.assertEqual(out['joint_gap_percentage_points'],19.999999999999996)

    def test_eighty_percent_inclusive_and_minimum_duration(self):
        self.assertTrue(flags(episode())[2])
        self.assertFalse(flags(episode(seconds=5.99))[2])
        self.assertFalse(flags(episode(cycles=3,yaw_good=3,roll_good=3))[2])

    def test_bootstrap_clusters_are_environments(self):
        rows=[episode(),episode(),episode(env=1)]
        out=summarize(rows,[],100)
        self.assertEqual(out['physics_clusters'],2)
        self.assertEqual(out['joint_success_ci95'],[1.,1.])


if __name__=='__main__':
    unittest.main()
