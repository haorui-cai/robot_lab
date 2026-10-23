"""Flat and smooth slopes only; pure configuration operations, no simulator imports."""
from copy import deepcopy
import math

FREE_TERRAIN_PROPORTIONS = {'flat': .2, 'hf_pyramid_slope': .4, 'hf_pyramid_slope_inv': .4}


def validate_free_gait_terrain(generator, allow_subset=False):
    terrains = generator.sub_terrains
    expected = set(FREE_TERRAIN_PROPORTIONS)
    names = set(terrains)
    invalid_names = not names <= expected if allow_subset else names != expected
    if not terrains or invalid_names:
        raise ValueError(f'Free-gait terrain must contain only reviewed flat/slopes: {list(terrains)}')
    if any(not math.isfinite(t.proportion) or t.proportion <= 0 for t in terrains.values()):
        raise ValueError('Remove disallowed types instead of zeroing proportions')
    if not math.isclose(sum(t.proportion for t in terrains.values()), 1., abs_tol=1e-10):
        raise ValueError('Terrain proportions must sum to one')
    for name, terrain in terrains.items():
        function = getattr(terrain.function, '__name__', '')
        if name == 'flat':
            if function != 'flat_terrain':
                raise ValueError('Flat must use IsaacLab MeshPlaneTerrainCfg')
        else:
            if function != 'pyramid_sloped_terrain' or terrain.inverted != name.endswith('_inv'):
                raise ValueError('Unexpected slope generator/inversion')
            low, high = terrain.slope_range
            if not 0 <= low <= high <= math.tan(math.radians(30.)) + 1e-12:
                raise ValueError('Do not exceed inherited V4 slope range0..tan(30deg)')
    return list(terrains)


def configure_free_gait_terrain(generator, flat):
    """Replace the dictionary, retain level/difficulty/scales and original slope geometry."""
    original = generator.sub_terrains
    # Copies avoid modifying the frozen V4 config or sibling instances.
    slopes = {name: deepcopy(original[name]) for name in ('hf_pyramid_slope', 'hf_pyramid_slope_inv')}
    generator.sub_terrains = {'flat': flat, **slopes}
    for name, terrain in generator.sub_terrains.items():
        terrain.proportion = FREE_TERRAIN_PROPORTIONS[name]
    validate_free_gait_terrain(generator)
