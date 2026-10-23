# TQBot2 新关节摩擦配置（2026-10-14）

按用户要求，后续使用当前源码的 TQBot2 仿真仅启用新版关节摩擦参数：

| 参数 | 全部活动关节 |
|---|---:|
| 旧 PhysX 关节摩擦系数 | 0 |
| 新静摩擦力矩 | 0.4 N·m |
| 新动摩擦力矩 | 0.4 N·m |
| 新黏性摩擦系数 | 0 |

`assets/tqbot_spawn.py` 在 URDF 导入后、环境复制和 PhysX 初始化前，将关节的 `physxJoint:jointFriction` 显式覆盖为零。不能删除这个 USD 属性：删除覆盖可能重新露出导入文件中的旧值。原 ROS URDF 及其物理描述不修改；TQBot2 的共享、行走碰撞、恢复碰撞三个生成入口都使用此覆盖。

`assets/tqbot.py` 的共享执行器设置新三项参数；StandardWalk 在转换为 DCMotor 时复制静、动、黏性三项。Kp14/Kd0.6、5ms 物理步长和现有地形参数保留。本次修改没有启动新训练。

验证脚本：`scripts/tests/tqbot2_joint_friction_smoke.py`。实际 GPU PhysX 中生成 StandardWalk 显式 DC、旧行走隐式 PD、滚动、恢复、四 hip、四 hip 加左大腿六类模型，每类四个环境副本。验证所有副本的旧系数均为零、新参数均为 `[0.4, 0.4, 0]`，物理步进和 scene reset 后仍一致；再将新三项全部清零并步进，确认旧系数仍为零，避免旧模型回退干扰。该检查验证参数接入，不代表已经验证训练收敛或消除先前振荡。

实际数据：[physx_verification.json](../outputs/TQBot2_new_joint_friction_20261014/physx_verification.json)，日志：[physx_verification.log](../outputs/TQBot2_new_joint_friction_20261014/physx_verification.log)。
