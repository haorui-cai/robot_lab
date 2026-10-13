# TQBot2 中英双语训练看板

地址：[http://127.0.0.1:6006](http://127.0.0.1:6006)。指标、训练名称、状态、说明和控件同时显示中文与英文。页面读取已有 TensorBoard scalar 事件文件，支持历史记录与当前训练对比。

## 使用

- 左侧勾选训练，最多对比8组；默认选择当前从零训练与已完成续训。
- 中文、英文或原始标签均可搜索。分组包括总览、指令跟踪、足端步态、能耗关节、恢复和训练诊断。
- 原始曲线为浅色，平滑曲线为深色；平滑使用20/50/100/200个记录点的尾随均值，默认100。
- 横轴可选原始迭代编号、相对起点、记录内经过时间。拖选或“最近500轮”放大共同横轴，双击或“显示全部”复原。共同区间可能落在某个较短记录之外。
- 悬停显示原始值和平滑值；默认每20秒刷新。左侧进度由实际训练日志读取，完成状态由训练结束记录核对，运行状态核对PID及命令。
- “导出当前数据”下载当前所选记录、指标和缩放区间的原始CSV；列名同时保留中英文与原始tag，缺失值留空，不插值。平滑不会改写导出数值。

迭代编号从0开始，4500轮计划的最终编号为4499。恢复记录保留检查点编号；不同记录不会自动拼接，当前续训与从零组的初始化、摩擦不同，不作单因素归因。

## 数据含义

课程是所有环境的平均等级，包括平地，并非最高通过等级或坡面单独等级；现有日志缺少分地形数据，页面不推测填充。误差是Isaac命令时间归一化指标，不是固定工况RMSE。分项奖励已包含权重和回合时长归一化，恢复奖励受门控和倒地时间影响；能耗为代理指标，不能视为实测电池消耗。

服务只读取注册日志，绑定本机127.0.0.1，不启动、停止或修改训练/控制器。不加载远程脚本或CDN，图表由本地Canvas绘制。已确认的数据范围是scalar曲线；未实现官方TensorBoard的图像、计算图或剖析器插件。

## 启动与验证

使用已有IsaacLab环境，无新增安装：

```bash
/home/eureka/miniconda3/envs/isaaclab/bin/python \
  /home/eureka/robot_lab/scripts/tools/TQBot2_tensorboard_zh/server.py \
  --root /home/eureka/robot_lab --port 6006
```

服务入口：[server.py](/home/eureka/robot_lab/scripts/tools/TQBot2_tensorboard_zh/server.py)。当前后台启动清单：[launch_metadata.json](/home/eureka/robot_lab/outputs/TQBot2_tensorboard_zh_20261013/launch_metadata.json)。终端前台运行可用Ctrl+C关闭看板服务，训练进程独立。

四项事件文件/HTTP集成检查通过：原始值一致、中英检索、CSV原始值与区间、路由及访问边界。实际15组训练记录、26项奖励核对通过；浏览器验证中英同时显示、训练对比、分类、平滑、横轴与缩放，实际下载CSV共5676行、10列并确认完整结束迭代。浏览器控制台未发现错误。首次测试夹具名称覆盖unittest.run导致启动失败，改名后四项通过，原失败记录保留。

[页面截图](/home/eureka/robot_lab/outputs/TQBot2_tensorboard_zh_20261013/bilingual_dashboard.jpg) · [后端核对记录](/home/eureka/robot_lab/outputs/TQBot2_tensorboard_zh_20261013/backend_checks.json) · [已下载CSV验证样本](/home/eureka/robot_lab/outputs/TQBot2_tensorboard_zh_20261013/TQBot2_export_verified.csv)
