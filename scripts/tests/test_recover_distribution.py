# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Recover noise bounds and PPO probability consistency."""

import importlib.util
import unittest
from pathlib import Path

import torch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/recover/config/tqbot2/agents/distribution.py"
)
spec = importlib.util.spec_from_file_location("recover_distribution", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DistributionTests(unittest.TestCase):
    def test_bounds_and_ppo_statistics(self):
        dist = module.BoundedGaussianDistribution(12)
        mean = torch.zeros(8, 12, requires_grad=True)
        dist.update(mean)
        self.assertTrue(torch.allclose(dist.std, torch.full_like(mean, 0.35)))
        dist.log_prob(torch.ones_like(mean) * 0.1).mean().backward()
        self.assertTrue(torch.isfinite(dist.noise_logit.grad).all())
        for logit in (-1000.0, 0.0, 1000.0):
            with torch.no_grad():
                dist.noise_logit.fill_(logit)
            dist.update(mean)
            self.assertTrue((dist.std >= 0.03).all() and (dist.std <= 0.600001).all())
            self.assertTrue(torch.isfinite(dist.log_prob(dist.sample())).all())
            self.assertTrue(torch.isfinite(dist.entropy).all())
            self.assertTrue(torch.allclose(dist.kl_divergence(dist.params, dist.params), torch.zeros(8)))
        self.assertTrue(torch.equal(dist.deterministic_output(mean), mean))
        self.assertTrue(torch.equal(torch.jit.script(dist.as_deterministic_output_module())(mean), mean))

    def test_invalid_ranges(self):
        for limits in ((0.0, 0.35, 0.6), (0.3, 0.2, 0.6), (0.03, 0.6, 0.6)):
            with self.assertRaises(ValueError):
                module.BoundedGaussianDistribution(12, init_std=limits[1], min_std=limits[0], max_std=limits[2])


if __name__ == "__main__":
    unittest.main()
