"""Selective self-collision for the full TQBot2 walking articulation."""

from itertools import product

from pxr import Usd, UsdPhysics

from isaaclab.sim.utils import clone

from robot_lab.assets.tqbot_spawn import spawn_tqbot2_from_urdf


LEG_LINKS = {
    leg: tuple(f"{leg}_{segment}" for segment in ("hip", "thigh", "calf"))
    for leg in ("FL", "FR", "RL", "RR")
}
WALK_SELF_COLLISION_PAIRS = frozenset(
    frozenset(pair)
    for left, right in (("FL", "FR"), ("RL", "RR"))
    for pair in product(LEG_LINKS[left], LEG_LINKS[right])
)
LEG_LINKS_WITH_FEET = {
    leg: links + (f"{leg}_foot",) for leg, links in LEG_LINKS.items()
}
WALK_SELF_COLLISION_PAIRS_WITH_FEET = frozenset(
    frozenset(pair)
    for left, right in (("FL", "FR"), ("RL", "RR"))
    for pair in product(LEG_LINKS_WITH_FEET[left], LEG_LINKS_WITH_FEET[right])
)


@clone
def spawn_tqbot2_walk_with_leg_collisions(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Author body-pair exclusions before cloning and creating PhysX handles.

    Allow every link combination between opposite front legs and opposite
    rear legs. All other self-collision pairs stay disabled. Environment and
    ground collision filtering is handled independently by the scene.
    """
    if not cfg.articulation_props.enabled_self_collisions:
        raise ValueError("TQBot2 walking leg collisions require enabled_self_collisions=True")
    root = spawn_tqbot2_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
    bodies = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    expected = {"trunk"} | {name for links in LEG_LINKS.values() for name in links}
    feet = {f"{leg}_foot" for leg in LEG_LINKS}
    if set(bodies) == expected | feet:
        allowed_pairs = WALK_SELF_COLLISION_PAIRS_WITH_FEET
    elif set(bodies) == expected:
        allowed_pairs = WALK_SELF_COLLISION_PAIRS
    else:
        raise RuntimeError(f"Unexpected TQBot2 walking rigid bodies: {sorted(bodies)}")
    trunk_colliders = [
        prim.GetPath()
        for prim in Usd.PrimRange(bodies["trunk"], Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not trunk_colliders:
        raise RuntimeError("TQBot2 trunk has no collision shapes")

    # Clear importer-authored internal filters before setting the exact policy.
    # Keep any filters pointing outside this robot.
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            rel = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
            rel.SetTargets([path for path in rel.GetTargets() if not path.HasPrefix(root.GetPath())])
    for name, body in bodies.items():
        # The trunk is also the articulation root. A filter on/to that prim
        # excludes the whole articulation, including our allowed leg pairs.
        # Exclude its individual shapes from each leg body instead.
        if name == "trunk":
            continue
        rel = UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel()
        external = list(rel.GetTargets())
        excluded = [
            other.GetPath()
            for other_name, other in bodies.items()
            if other_name not in (name, "trunk") and frozenset((name, other_name)) not in allowed_pairs
        ]
        rel.SetTargets(external + excluded + trunk_colliders)
    return root
