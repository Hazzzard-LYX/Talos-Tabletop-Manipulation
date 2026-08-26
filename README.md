# TALOS Tabletop Manipulation

独立的 MJLab 研究项目，目标是在 TALOS 保持原地站立和平衡的同时，主动夹持并移动桌面物体。

## 当前状态

项目拆分、phase-0 环境骨架和首版 phase-1 抓取任务已经完成：

- 从 `WBC-For-Talos` 复制了已验证的 TALOS MJCF/STL；
- 保留真实单电机三指耦合、手指接触 mesh、头部碰撞 mesh；
- 保留腕部/踝部 F/T 传感器和真实执行器限制；
- 新增无桌腿的固定悬浮桌面、绿色取物区和红色放置区；
- 新增蓝色 60 mm 立方体和 70 mm 球体任务；
- 抓取 policy 控制双腿、躯干、双臂和右夹爪，自主维持原地平衡；
- actor 与 critic 首版直接接收物体中心真值，后续保持三维接口不变并替换为视觉估计；
- 新增右手多接触、物体抬升、base 稳定和非法桌面/地面接触判据；
- 注册 `Mjlab-Tabletop-Reaching-Talos-v0` 作为模型与接口冒烟任务。

首版可训练任务：

- `Mjlab-Tabletop-Grasp-Cube-Talos-v0`
- `Mjlab-Tabletop-Grasp-Sphere-Talos-v0`

当前只训练第一个技能策略：右手接近、形成至少两处手指/连杆接触，并将物体抬离桌面 8 cm。红色区域只作为第二个 transport/place 策略的目标标记，尚未进入抓取奖励。

## 目录

```text
src/talos_tabletop/robots/talos/  TALOS 模型、碰撞体和执行器配置
src/talos_tabletop/assets.py     桌面和自由物体实体
src/talos_tabletop/tasks/        环境、MDP 接口和任务注册
tests/                            模型与配置测试
docs/task_scope_zh.md             任务边界和阶段计划
slurm/                            后续集群训练脚本
reports/                          实验报告
```

## 本地检查

```bash
uv sync --extra cpu
uv run ruff check .
uv run pytest
```

## 设计原则

- 新项目具有独立 Git 历史，不依赖旧项目的 Python 包；
- 不复制旧 checkpoint、训练日志、托盘奖励或实验报告；
- 与集群有关的通用脚本继续由 `Shared-IAS` 管理；
- 机器人模型的上游来源和许可证保留在模型目录中。
