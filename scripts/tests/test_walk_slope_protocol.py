import importlib.util
from pathlib import Path
import math
import numpy as np

spec = importlib.util.spec_from_file_location('slope_geometry', Path(__file__).parents[1]/'reinforcement_learning/walk_slope/geometry.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_physical_ramp_angles_and_upward_normals():
    for angle in [-30, -25, 0, 10, 30]:
        vertices, faces = module.ramp_vertices(angle)
        triangle = vertices[faces[0]]
        normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
        normal /= np.linalg.norm(normal)
        np.testing.assert_allclose(normal, [-math.sin(math.radians(angle)), 0, math.cos(math.radians(angle))], atol=1e-12)
        assert abs(math.degrees(math.atan2(vertices[1, 2]-vertices[0, 2], vertices[1, 0]-vertices[0, 0]))-angle) < 1e-10


def test_standing_sliding_and_early_failure_cannot_pass():
    row = dict(duration_s=12., failed=False, tracking_rmse=.04, command_m_s=.45,
               forward_m=4.7, measured_s=11., lateral_max_m=.1, aligned_fraction=1., heading_good_fraction=1.)
    assert module.accepted(row)
    for change in [dict(forward_m=0.), dict(forward_m=-5.), dict(duration_s=3.),
                   dict(failed=True), dict(tracking_rmse=.3), dict(lateral_max_m=.8), dict(heading_good_fraction=.7)]:
        assert not module.accepted({**row, **change})


def test_missing_or_failed_conditions_cannot_pass_matrix():
    row = dict(duration_s=12., failed=False, tracking_rmse=.04, command_m_s=.45,
               forward_m=5., measured_s=11., lateral_max_m=.1, aligned_fraction=1., heading_good_fraction=1.,
               slope_deg=30., friction=.85)
    result = module.summarize([row, {**row, 'duration_s': 1., 'failed': True}])
    assert not result['passed']
    assert next(iter(result['cells'].values()))['success_rate'] == .5
