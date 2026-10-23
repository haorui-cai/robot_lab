"""CPU regression checks for terrain grouping and read-only logging."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

path = Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/terrain_statistics.py"
spec = importlib.util.spec_from_file_location("terrain_statistics", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TerrainStatisticsTests(unittest.TestCase):
    def test_column_mapping_matches_generator_and_actual_population(self):
        names = ["stairs", "stairs_inv", "boxes", "rough", "slope", "slope_inv", "flat"]
        weights = [.18, .18, .18, .18, .09, .09, .1]
        cfg = SimpleNamespace(num_cols=20, num_rows=10, sub_terrains={
            name: SimpleNamespace(proportion=weight) for name, weight in zip(names, weights)
        })
        actual_names, groups = module.terrain_column_groups(cfg)
        reference = [np.min(np.where(col / 20 + .001 < np.cumsum(np.array(weights) / sum(weights)))[0])
                     for col in range(20)]
        self.assertEqual(actual_names, names)
        np.testing.assert_array_equal(groups, reference)
        types = torch.div(torch.arange(4096), 4096 / 20, rounding_mode="floor").long()
        stats = module.summarize_terrain_levels(torch.zeros(4096, dtype=torch.long), types, cfg)
        self.assertEqual(stats["flat/env_count"], 409)
        self.assertEqual(sum(stats[f"{name}/env_count"] for name in names), 4096)

    def test_independent_distributions_reset_rates_and_absent_groups(self):
        cfg = SimpleNamespace(num_cols=3, num_rows=10, sub_terrains={
            name: SimpleNamespace(proportion=1) for name in ("stairs", "slope", "absent")
        })
        levels = torch.tensor([0, 2, 4, 9])
        types = torch.tensor([0, 0, 1, 1])
        original = levels.clone()
        decisions = (torch.tensor([0, 3]), torch.tensor([False, True]),
                     torch.tensor([True, False]), torch.tensor([False, True]))
        stats = module.summarize_terrain_levels(levels, types, cfg, decisions)
        self.assertEqual(stats["stairs/mean"], 1)
        self.assertEqual(stats["slope/mean"], 6.5)
        self.assertEqual(stats["stairs/move_down_rate"], 1)  # Includes failure clamped at zero.
        self.assertEqual(stats["slope/top_completed_rate"], 1)
        self.assertFalse(any(key.startswith("absent/") for key in stats))
        for name in ("stairs", "slope"):
            self.assertEqual(sum(stats[f"{name}/level_{i}_fraction"] for i in range(10)), 1)
        torch.testing.assert_close(levels, original)

    def test_update_runs_once_and_top_completion_is_not_failure(self):
        generator = SimpleNamespace(num_cols=2, num_rows=10, size=(8, 8), sub_terrains={
            name: SimpleNamespace(proportion=1) for name in ("stairs", "slope")
        })
        terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_generator=generator),
                                  terrain_levels=torch.tensor([0, 9]),
                                  terrain_types=torch.tensor([0, 1]), max_terrain_level=10)
        robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0., 0., 0.], [5., 0., 0.]])))
        class Scene(dict):
            pass
        scene = Scene(robot=robot)
        scene.terrain = terrain
        scene.env_origins = torch.zeros(2, 3)
        env = SimpleNamespace(scene=scene, num_envs=2, device="cpu", max_episode_length_s=20,
                              command_manager=SimpleNamespace(get_command=lambda _: torch.ones(2, 3)))
        updates = []
        def update_once(env, ids):
            updates.append(ids)
            # An unsuccessful level-zero environment remains clamped; the top
            # completion is randomly reassigned, represented here by level 2.
            terrain.terrain_levels[:] = torch.tensor([0, 2])
            return terrain.terrain_levels.float().mean()
        upstream = SimpleNamespace(terrain_levels_vel=update_once)
        name = "isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums"
        with patch.dict("sys.modules", {name: upstream}):
            self.assertEqual(module.terrain_levels_with_statistics(env, slice(None)), 1)
        stats = module.terrain_level_statistics(env, slice(None))
        self.assertEqual(len(updates), 1)
        self.assertEqual(stats["stairs/move_down_rate"], 1)
        self.assertEqual(stats["slope/move_up_rate"], 1)
        self.assertEqual(stats["slope/move_down_rate"], 0)
        self.assertEqual(stats["slope/top_completed_rate"], 1)
        torch.testing.assert_close(terrain.terrain_levels, torch.tensor([0, 2]))


if __name__ == "__main__":
    unittest.main()
