"""CPU final-post-init and evaluation configuration checks, using a lightweight V4 parent.

Executes actual free-gait post-init / recorder-selection AST, not an Isaac simulation.
"""
import ast
from copy import deepcopy
import importlib
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
from test_tqbot2_free_gait_control import CONFIG, ROOT, fixture, policy

terrain_policy = importlib.import_module('tqbot2_cpu_control.free_gait_terrain')


def flat_terrain(): pass

def pyramid_sloped_terrain(): pass


def fake_plane(**kwargs):
    return NS(function=flat_terrain, **kwargs)


class Terrain(NS):
    def copy(self): return deepcopy(self)


def generator_fixture():
    return NS(num_rows=10, num_cols=20, curriculum=True, difficulty_range=(0.,1.),
              horizontal_scale=.1, vertical_scale=.005, size=(8.,8.), sub_terrains={
        'pyramid_stairs':Terrain(proportion=.2), 'pyramid_stairs_inv':Terrain(proportion=.2),
        'boxes':Terrain(proportion=.2), 'random_rough':Terrain(proportion=.2),
        'hf_pyramid_slope':Terrain(proportion=.1,function=pyramid_sloped_terrain,inverted=False,
                                   slope_range=(0.,math.tan(math.radians(30.))),platform_width=2.,border_width=.25),
        'hf_pyramid_slope_inv':Terrain(proportion=.1,function=pyramid_sloped_terrain,inverted=True,
                                       slope_range=(0.,math.tan(math.radians(30.))),platform_width=2.,border_width=.25)})


def final_configuration():
    parent = fixture()
    parent.scene=NS(terrain=NS(terrain_generator=generator_fixture()))
    class FakeV4:
        def __init__(self): self.__post_init__()
        def __post_init__(self): self.__dict__.update(deepcopy(vars(parent)))
    source=ast.parse((CONFIG/'free_gait_env_cfg.py').read_text())
    cls=next(node for node in source.body if isinstance(node,ast.ClassDef))
    namespace={'configclass':lambda value:value, 'TQBot2RewardV4EnvCfg':FakeV4,
               'apply_free_gait':policy.apply_free_gait,
               'configure_free_gait_terrain':terrain_policy.configure_free_gait_terrain,
               'MeshPlaneTerrainCfg':fake_plane}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(CONFIG/'free_gait_env_cfg.py'),'exec'),namespace)
    return namespace['TQBot2FreeGaitEnvCfg'](),parent


class TerrainTests(unittest.TestCase):
    def test_actual_final_free_post_init_has_only_flat_and_two_slopes(self):
        cfg,parent=final_configuration()
        generator=cfg.scene.terrain.terrain_generator
        self.assertEqual(list(generator.sub_terrains),list(terrain_policy.FREE_TERRAIN_PROPORTIONS))
        self.assertEqual({k:v.proportion for k,v in generator.sub_terrains.items()},terrain_policy.FREE_TERRAIN_PROPORTIONS)
        self.assertAlmostEqual(sum(v.proportion for v in generator.sub_terrains.values()),1.)
        self.assertEqual(len(policy.active_reward_names(cfg.rewards)),14)
        self.assertEqual(len(parent.scene.terrain.terrain_generator.sub_terrains),6)
        for key in ('timing','terminations','commands','protection'):
            self.assertEqual(vars(getattr(cfg,key)) if key=='protection' else getattr(cfg,key),
                             vars(getattr(parent,key)) if key=='protection' else getattr(parent,key))

    def test_original_slope_geometry_difficulty_and_curriculum_are_preserved(self):
        cfg,parent=final_configuration()
        generator=cfg.scene.terrain.terrain_generator
        before=parent.scene.terrain.terrain_generator
        for key in ('num_rows','num_cols','curriculum','difficulty_range','horizontal_scale','vertical_scale','size'):
            self.assertEqual(getattr(generator,key),getattr(before,key))
        for name in ('hf_pyramid_slope','hf_pyramid_slope_inv'):
            actual=generator.sub_terrains[name]
            original=before.sub_terrains[name]
            self.assertEqual(actual.slope_range,original.slope_range)
            self.assertEqual(actual.platform_width,original.platform_width)
            self.assertEqual(actual.border_width,original.border_width)
            self.assertEqual(actual.inverted,original.inverted)
            self.assertEqual(original.proportion,.1)

    def test_actual_curriculum_column_grouping_has_four_eight_eight_columns(self):
        cfg,_=final_configuration()
        source=ast.parse((ROOT/'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/terrain_statistics.py').read_text())
        function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='terrain_column_groups')
        namespace={'np':np}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'terrain_statistics.py','exec'),namespace)
        names,groups=namespace['terrain_column_groups'](cfg.scene.terrain.terrain_generator)
        self.assertEqual(names,['flat','hf_pyramid_slope','hf_pyramid_slope_inv'])
        self.assertEqual(np.bincount(groups).tolist(),[4,8,8])

    def test_zero_weight_old_type_and_bad_proportions_are_rejected(self):
        cfg,_=final_configuration()
        generator=cfg.scene.terrain.terrain_generator
        generator.sub_terrains['boxes']=Terrain(proportion=0.)
        with self.assertRaises(ValueError):terrain_policy.validate_free_gait_terrain(generator)
        del generator.sub_terrains['boxes']
        generator.sub_terrains['flat'].proportion=0.
        with self.assertRaises(ValueError):terrain_policy.validate_free_gait_terrain(generator)

    def test_wrong_generator_inversion_and_larger_slope_are_rejected(self):
        cfg,_=final_configuration()
        generator=cfg.scene.terrain.terrain_generator
        generator.sub_terrains['hf_pyramid_slope_inv'].inverted=False
        with self.assertRaises(ValueError):terrain_policy.validate_free_gait_terrain(generator)
        generator.sub_terrains['hf_pyramid_slope_inv'].inverted=True
        generator.sub_terrains['hf_pyramid_slope'].slope_range=(0.,.8)
        with self.assertRaises(ValueError):terrain_policy.validate_free_gait_terrain(generator)
        generator.sub_terrains['hf_pyramid_slope'].slope_range=(0.,math.tan(math.radians(30.)))
        generator.sub_terrains['flat'].function=pyramid_sloped_terrain
        with self.assertRaises(ValueError):terrain_policy.validate_free_gait_terrain(generator)

    def test_actual_recorder_selection_never_reintroduces_disallowed_terrain(self):
        source=ast.parse((ROOT/'scripts/reinforcement_learning/rsl_rl/record_tqbot2_ablation.py').read_text())
        main=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        start=next(i for i,n in enumerate(main.body) if isinstance(n,ast.If) and ast.unparse(n.test).startswith('args.terrain in'))
        selection=main.body[start:start+2]
        for name,expected in [('mixed',{'flat','hf_pyramid_slope','hf_pyramid_slope_inv'}),
                              ('flat',{'flat'}),('slope',{'hf_pyramid_slope'}),('slope-inv',{'hf_pyramid_slope_inv'})]:
            cfg,_=final_configuration()
            generator=cfg.scene.terrain.terrain_generator
            namespace={'args':NS(terrain=name,variant='free',slope_deg=10.),'generator':generator,
                       'math':math,'MeshPlaneTerrainCfg':fake_plane,
                       'validate_free_gait_terrain':terrain_policy.validate_free_gait_terrain}
            exec(compile(ast.Module(body=selection,type_ignores=[]),'recorder_selection','exec'),namespace)
            self.assertEqual(set(generator.sub_terrains),expected)
            terrain_policy.validate_free_gait_terrain(generator,allow_subset=True)
            if name!='mixed':
                with self.assertRaises(ValueError):terrain_policy.validate_free_gait_terrain(generator)

    def test_play_uses_selected_task_config_without_injecting_terrain_types(self):
        source=(ROOT/'scripts/reinforcement_learning/rsl_rl/play.py').read_text()
        self.assertIn('@hydra_task_config(args_cli.task, args_cli.agent)',source)
        # Play does not inject old types, and only free-gait disables mismatched random-tile statistics.
        self.assertNotIn('sub_terrains',source)
        tree=ast.parse(source)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        block=next(n for n in main.body if isinstance(n,ast.If) and 'FreeGaitScratch-v0' in ast.unparse(n.test))
        block=deepcopy(block)
        block.body=[n for n in block.body if not isinstance(n,ast.ImportFrom)]
        for variant in ('FreeGaitScratch','RewardV4Scratch'):
            cfg,_=final_configuration()
            cfg.curriculum=NS(terrain_levels='training_level_update',terrain_breakdown='training_column_statistics')
            namespace={'env_cfg':cfg,'args_cli':NS(task=f'RobotLab-Isaac-Velocity-Rough-TQBot2-{variant}-v0'),
                       'validate_free_gait_terrain':terrain_policy.validate_free_gait_terrain}
            exec(compile(ast.Module(body=[block],type_ignores=[]),'play_free_branch','exec'),namespace)
            if variant=='FreeGaitScratch':
                self.assertIsNone(cfg.curriculum.terrain_levels)
                self.assertIsNone(cfg.curriculum.terrain_breakdown)
            else:
                self.assertEqual(cfg.curriculum.terrain_levels,'training_level_update')


if __name__=='__main__':unittest.main()
