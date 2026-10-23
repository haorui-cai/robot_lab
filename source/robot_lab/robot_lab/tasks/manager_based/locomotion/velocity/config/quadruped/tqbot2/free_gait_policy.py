"""Reviewed reward-only ablation. This module can be tested without Isaac Sim."""

V4_ACTIVE = {
    "lin_vel_z_l2", "ang_vel_xy_l2", "joint_torques_l2", "joint_acc_l2", "joint_pos_limits",
    "joint_power", "stand_still", "joint_pos_penalty", "joint_mirror", "action_rate_l2",
    "undesired_contacts", "contact_forces", "track_lin_vel_xy_exp", "track_ang_vel_z_exp",
    "feet_air_time_variance", "feet_gait", "feet_contact_without_cmd", "feet_slide", "upward",
    "feet_frequency", "feet_ground_clearance",
}

REMOVED = {
    "feet_frequency": "Explicit2Hz full-cycle target",
    "feet_gait": "Diagonal synchronization and opposite-pair phase timing",
    "feet_air_time_variance": "Equalizes air/contact durations across feet",
    "joint_mirror": "FootDisplacementSymmetry specifies diagonal foot geometry",
    "joint_pos_penalty": "SwingJointPositionPenalty exempts swing legs using contact phase",
    "feet_ground_clearance": "Specifies5cm swing clearance and uses swing-phase gating",
    "feet_contact_without_cmd": "Pays per FIRST CONTACT event at zero command, not continuous support",
}

INACTIVE_GUARDS = (
    "feet_air_time", "feet_contact", "feet_height", "feet_height_body", "feet_swing_clearance",
    "feet_frequency_excess", "feet_cycle_overdue",
)


def active_reward_names(rewards):
    return {
        name for name in dir(rewards) if not name.startswith("_")
        and getattr(getattr(rewards, name), "weight", 0) != 0
    }


def apply_free_gait(rewards):
    """Disable seven reviewed active terms; change no event, sensor, action or termination."""
    before = active_reward_names(rewards)
    if before != V4_ACTIVE:
        raise ValueError(f"Unexpected V4 reward set: missing={V4_ACTIVE-before}, extra={before-V4_ACTIVE}")
    for name in REMOVED:
        setattr(rewards, name, None)
    for name in INACTIVE_GUARDS:
        if hasattr(rewards, name):
            setattr(rewards, name, None)
    if active_reward_names(rewards) != V4_ACTIVE - REMOVED.keys():
        raise ValueError("Free-gait reward ablation did not match reviewed contract")
