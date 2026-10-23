# TQBot1 / TQBot2 描述依赖

模型唯一来源为私有仓库 `ustc-407lab/tqbot_description`。本项目通过
`external/tqbot_description` 子模块固定一个准确提交，不跟随分支自动升级。
几何、质量、惯量、关节机械范围和派生模型参数均在私有描述仓库修改。

## 初始化

需要已有的私有仓库读取权限和 Git SSH 认证：

```bash
git submodule update --init external/tqbot_description
```

公开仓库中的链接不授予私有资源访问权限。本项目不再保存新的私有模型副本。
历史上已公开的模型对象仍在旧提交中；本次迁移没有改写历史。

## ROS 项目

兼容目录 `unitree_ros/robots/TQBot_description`（TQBot1）和
`unitree_ros/robots/TQBot2_description`（TQBot2）指向子模块中的对应包，保留旧
ROS 包名和资源 URI。`external/CATKIN_IGNORE` 避免 Catkin 重复发现同一包。

Xacro 入口保留当前行为；TQBot2 的生成 URDF 使用明确的历史兼容 profile，保留
迁移前的控制插件和碰撞设置。几何及 profile 的变更必须在描述仓库验证后升级。

## robot_lab

在已安装描述仓库 `requirements.txt` 依赖的 Python 环境中执行：

```bash
python scripts/tools/sync_tqbot2_description.py
```

入口检查子模块提交与记录的 pin 一致，生成 Isaac、walking、hip-only 和
left-thigh rolling 四种 URDF 及几何高度表。原有模型路径是指向子模块忽略目录
`build/isaac/TQBot2_description` 的链接，训练代码可继续使用既有资源路径。
旧的 `build_tqbot2_*` 生成命令转发到同一私有生成器，避免另设物理来源。
转换和模型缓存保留本地；不要提交生成的 USD、日志、报告或训练快照。

## 升级与验证

先在描述仓库修改、验证并提交，然后在本项目更新子模块准确提交；审查父仓库
gitlink 差异后提交。不要在消费目录直接修改 Xacro 或 URDF。

迁移已进行 Xacro 展开、URDF/mesh 路径检查、模型静态语义比较，以及现有
robot_lab 的六项 CPU 描述测试。没有启动仿真或硬件，不能据此声称闭环验证通过。
迁移前本地原模型和脚本保存在各项目已有的忽略输出/数据目录中，并记录逐文件
SHA-256；这些备份不提交到 Git。
