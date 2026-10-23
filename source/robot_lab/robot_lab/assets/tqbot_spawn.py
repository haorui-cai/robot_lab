"""TQBot2 import overrides shared by walking, rolling and recovery."""

from pxr import PhysxSchema, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone


@clone
def spawn_tqbot2_from_urdf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Disable legacy friction before cloning; actuators supply the new model.

    The URDF importer populates physxJoint:jointFriction from URDF dynamics.
    Leaving it nonzero permits the deprecated model to take over when the
    new static/dynamic/viscous properties are zero. Author zero on the stage
    before PhysX initialization without changing the source ROS description.
    """
    root = sim_utils.spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateJointFrictionAttr(0.0)
    return root
