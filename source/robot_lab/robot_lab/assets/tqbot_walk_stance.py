"""Independent standing reference and policy action zero for StandardWalk.

The articulation's default_joint_pos remains the standing reference: resets,
stand_still and joint_pos_penalty use it. The policy action offset is configured
separately and must never be substituted for that standing reference.
"""

import math


WALK_DEFAULT_FOOT_Z_TRUNK = -0.25
WALK_THIGH_ANGLE = 0.80
# Solve the bent-knee branch for the requested trunk-relative foot Z,
# retaining the original thigh soft-limit margin instead of crouching more.
WALK_CALF_ANGLE = -WALK_THIGH_ANGLE - math.acos(
    (-WALK_DEFAULT_FOOT_Z_TRUNK - 0.213 * math.cos(WALK_THIGH_ANGLE)) / 0.153
)
WALK_STANDING_JOINT_POS = {
    ".*L_hip_joint": 0.0,
    ".*R_hip_joint": 0.0,
    "F.*_thigh_joint": WALK_THIGH_ANGLE,
    "R.*_thigh_joint": WALK_THIGH_ANGLE,
    ".*_calf_joint": WALK_CALF_ANGLE,
}

# Retain the existing public name for stance geometry tools and old callers.
WALK_JOINT_POS = WALK_STANDING_JOINT_POS

# Independent policy coordinates. Do not derive offsets from the
# standing reference: changing the standing pose must not move action zero.
# These thigh limits are checked against all four joints in TQBot2_walk.urdf.
WALK_POLICY_THIGH_LIMITS = (0.20, 3.20)
# Keep a 0.13-rad reward margin inside each mechanical thigh limit.
WALK_THIGH_SOFT_LIMITS = (0.33, 3.07)
# Keep both policy references independent of the standing geometry. The
# legacy reference is active for the 12000-iteration implicit-PD experiment.
WALK_LEGACY_POLICY_ZERO_JOINT_POS = {
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.80,
    ".*_calf_joint": -1.6445578923423,
}
WALK_POLICY_ZERO_JOINT_POS = {
    ".*_hip_joint": 0.0,
    # Halfway between the previous policy zeros, 1.90 and 0.80 rad.
    ".*_thigh_joint": 1.35,
    ".*_calf_joint": -1.6445578923423,
}

# Both joint origins are at trunk Z=0 with hip=0. The rear leg is mirrored
# in X; its foot has the same Z. URDF FK is checked by test_tqbot2_walk_stance.
WALK_FOOT_HEIGHT_INCREMENT = 0.085
WALK_FOOT_HEIGHT_TARGET_TRUNK = WALK_DEFAULT_FOOT_Z_TRUNK + WALK_FOOT_HEIGHT_INCREMENT
