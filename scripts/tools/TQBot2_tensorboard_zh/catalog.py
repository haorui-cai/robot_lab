"""Chinese and English descriptions of existing TensorBoard scalar tags."""

LABELS = {'Train/mean_reward': '总回报（episode 均值）', 'Train/mean_episode_length': '平均 episode 长度（控制步）', 'Curriculum/terrain_levels': '全部地形平均课程等级（0–9）', 'Metrics/base_velocity/error_vel_xy': '平面线速度误差指标（m/s，越低越好）', 'Metrics/base_velocity/error_vel_yaw': '偏航角速度误差指标（rad/s，越低越好）', 'lin_vel_z_l2': '竖直速度惩罚', 'ang_vel_xy_l2': '横滚/俯仰角速度惩罚', 'joint_torques_l2': '关节力矩平方惩罚', 'joint_acc_l2': '关节加速度惩罚', 'joint_pos_limits': '关节位置限位惩罚', 'joint_power': '关节机械功率惩罚', 'stand_still': '静止命令关节偏离惩罚', 'joint_pos_penalty': '关节姿态偏离惩罚', 'joint_mirror': '关节对称性惩罚', 'action_rate_l2': '动作变化惩罚', 'undesired_contacts': '非期望身体接触惩罚', 'contact_forces': '足端大接触力惩罚', 'track_lin_vel_xy_exp': '线速度跟踪奖励', 'track_ang_vel_z_exp': '偏航速度跟踪奖励', 'feet_air_time': '足端腾空时间项', 'feet_air_time_variance': '腾空时间差异惩罚', 'feet_gait': '对角步态奖励', 'feet_contact_without_cmd': '无运动命令接地奖励', 'feet_slide': '足端地面滑移惩罚', 'upward': '机身正立奖励', 'feet_swing_clearance': '摆动足底低间隙惩罚', 'recovery_effort': '恢复力矩惩罚', 'recovery_torque_change': '恢复力矩变化惩罚', 'recovery_joint_velocity': '恢复关节速度惩罚', 'walking_energy_efficiency': '行走距离能耗代理奖励', 'walking_joule_proxy': '行走铜耗代理惩罚'}

ENGLISH = {
    'Train/mean_reward': 'Mean episode return',
    'Train/mean_episode_length': 'Mean episode length',
    'Curriculum/terrain_levels': 'Mean curriculum level · all terrains',
    'Metrics/base_velocity/error_vel_xy': 'Planar velocity error metric',
    'Metrics/base_velocity/error_vel_yaw': 'Yaw rate error metric',
    'lin_vel_z_l2': 'Vertical velocity penalty',
    'ang_vel_xy_l2': 'Roll / pitch rate penalty',
    'joint_torques_l2': 'Squared joint torque penalty',
    'joint_acc_l2': 'Joint acceleration penalty',
    'joint_pos_limits': 'Joint position limit penalty',
    'joint_power': 'Joint mechanical power penalty',
    'stand_still': 'Standstill joint deviation penalty',
    'joint_pos_penalty': 'Joint posture deviation penalty',
    'joint_mirror': 'Joint symmetry penalty',
    'action_rate_l2': 'Action change penalty',
    'undesired_contacts': 'Undesired body contact penalty',
    'contact_forces': 'Large foot contact force penalty',
    'track_lin_vel_xy_exp': 'Planar velocity tracking reward',
    'track_ang_vel_z_exp': 'Yaw rate tracking reward',
    'feet_air_time': 'Foot air-time term',
    'feet_air_time_variance': 'Air-time variance penalty',
    'feet_gait': 'Diagonal gait reward',
    'feet_contact_without_cmd': 'No-command ground contact reward',
    'feet_slide': 'Ground-relative foot slip penalty',
    'feet_height_body': 'Body-relative calf height penalty',
    'upward': 'Upright orientation reward',
    'feet_swing_clearance': 'Low swing-foot clearance penalty',
    'recovery_effort': 'Recovery torque penalty',
    'recovery_torque_change': 'Recovery torque change penalty',
    'recovery_joint_velocity': 'Recovery joint speed penalty',
    'walking_energy_efficiency': 'Distance-normalized energy proxy reward',
    'walking_joule_proxy': 'Walking copper-loss proxy penalty',
    'Loss/value': 'Value function loss', 'Loss/surrogate': 'Policy surrogate loss',
    'Loss/entropy': 'Entropy term', 'Loss/learning_rate': 'Learning rate',
    'Policy/mean_std': 'Action exploration standard deviation',
    'Perf/total_fps': 'Sampling throughput', 'Perf/collection_time': 'Sampling time',
    'Perf/learning_time': 'Learning time',
    'Episode_Termination/time_out': 'Episode time-limit fraction',
    'Episode_Termination/terrain_out_of_bounds': 'Terrain boundary termination fraction',
    'Episode_Termination/high_friction_flat_exit': 'High-friction flat exit fraction',
}
LABELS['feet_height_body'] = '相对机身的足端刚体高度惩罚'
ENGLISH['feet_height_body'] = 'Body-relative foot-body height penalty'
LABELS['feet_slide'] = '足端滑移项'
ENGLISH['feet_slide'] = 'Foot slip term'
LABELS['feet_frequency'] = '足端完整周期步频奖励'
ENGLISH['feet_frequency'] = 'Measured full-cycle foot frequency reward'
LABELS['feet_frequency_excess'] = '足端过高步频惩罚'
ENGLISH['feet_frequency_excess'] = 'Excess full-cycle foot frequency penalty'
LABELS['feet_ground_clearance'] = '摆动足底对地净空误差惩罚'
ENGLISH['feet_ground_clearance'] = 'Terrain-relative swing-foot clearance error penalty'

TERRAINS = {
    'pyramid_stairs': ('凸形阶梯', 'Pyramid stairs'),
    'pyramid_stairs_inv': ('凹形阶梯', 'Inverted pyramid stairs'),
    'boxes': ('随机方块', 'Random boxes'),
    'random_rough': ('随机起伏', 'Random rough terrain'),
    'hf_pyramid_slope': ('凸形坡面', 'Pyramid slope'),
    'hf_pyramid_slope_inv': ('凹形坡面', 'Inverted pyramid slope'),
    'high_friction_flat': ('高摩擦平地', 'High-friction flat'),
}
TERRAIN_STATS = {
    'mean': ('平均课程等级', 'Mean curriculum level', '等级', 'level'),
    'max': ('最高当前等级', 'Highest current level', '等级', 'level'),
    'env_count': ('环境数量', 'Environment count', '环境', 'environments'),
    'move_up_rate': ('复位批次升级请求比例', 'Reset-batch promotion request fraction', '比例', 'fraction'),
    'move_down_rate': ('复位批次降级请求比例', 'Reset-batch demotion request fraction', '比例', 'fraction'),
    'top_completed_rate': ('复位批次最高级完成比例', 'Reset-batch top-level completion fraction', '比例', 'fraction'),
}


def terrain_descriptor(tag):
    parts = tag.split('/')
    if len(parts) != 4 or parts[:2] != ['Curriculum', 'terrain_breakdown']:
        return None
    name, name_en = TERRAINS.get(parts[2], (parts[2], parts[2]))
    stat = parts[3]
    if stat.startswith('level_') and stat.endswith('_fraction'):
        level = stat.removeprefix('level_').removesuffix('_fraction')
        zh, en, unit, unit_en = f'等级 {level} 占比', f'Level {level} fraction', '比例', 'fraction'
    else:
        zh, en, unit, unit_en = TERRAIN_STATS.get(stat, (stat, stat, '', ''))
    return {
        'tag': tag, 'name': f'{name} · {zh}', 'name_en': f'{name_en} · {en}',
        'unit': unit, 'unit_en': unit_en, 'direction': 'neutral',
        'hint': '按出生地形列分组，各环境独立升级。批次比例不是累计通过率；当前等级不等于固定坡度性能验收。',
        'hint_en': 'Grouped by spawn-terrain column; environments progress independently. Batch fractions are not cumulative success rates or fixed-slope evaluations.',
    }
