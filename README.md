# TALOS Tabletop Manipulation

独立的 MJLab 研究项目，目标是在 TALOS 保持原地站立和平衡的同时，主动夹持并移动桌面物体。

## 当前状态

项目拆分、phase-0 环境骨架和首版 phase-1 抓取任务已经完成：

- 从 `WBC-For-Talos` 复制了已验证的 TALOS MJCF/STL；
- 保留真实单电机三指耦合、手指接触 mesh、头部碰撞 mesh；
- 保留腕部/踝部 F/T 传感器和真实执行器限制；
- 新增顶面高度 0.86 m 的无桌腿固定悬浮桌面、绿色取物区和红色放置区；
- 新增蓝色 60 mm 立方体和 70 mm 球体任务；
- 抓取 policy 控制双腿、躯干、双臂和右夹爪，自主维持原地平衡；
- actor 与 critic 首版直接接收物体中心真值，后续保持三维接口不变并替换为视觉估计；
- 新增对向双面接触、摩擦锥安全余量、持续 pick 验证、base 稳定和非法桌面/地面接触判据；
- 注册 `Mjlab-Tabletop-Reaching-Talos-v0` 作为模型与接口冒烟任务。

首版可训练任务：

- `Mjlab-Tabletop-Grasp-Cube-Talos-v0`
- `Mjlab-Tabletop-Grasp-Sphere-Talos-v0`

当前只训练第一个技能策略：右手接近物体，在两个相反表面形成能够抵抗任务 wrench 的夹持，完成 2 cm micro-lift 验证，再将物体稳定抬离桌面 8 cm。接近、抓握质量和抬升只奖励 episode 内的新进步；成功还要求低手物相对速度并持续保持，以排除只接触、滑动和弹飞物体的投机策略。红色区域只作为第二个 transport/place 策略的目标标记。

反向课程的 Stage0 现从 8 cm 目标高度的自由方块和三指闭合 IK 接触姿态开始。它不再奖励 episode 内的历史最高高度，而是连续奖励当前有效抬升，并使用高斯目标项缩小与 8 cm 的高度误差；目标项同时要求多连杆接触和低物体速度。成功必须在目标高度 ±1.5 cm 内保持至少 5 秒并达到最低任务 wrench 质量。夹爪关节语义也已校正为 `-0.959931=闭合、0=张开`，接触力显式转换到世界坐标后才用于抓握质量计算。

首个反向课程实验注册为 `Mjlab-Tabletop-Stable-Contact-Lift-Cube-Talos-v0`：
机器人从全身 IK 生成的空中稳定抓握姿态开始，方块仍是没有固定关节的 6-DoF 自由刚体。任务先学习维持至少两个独立手指链接接触、保持站立，并把方块稳定保持在离初始桌面位置 8 cm 的目标高度。每个控制步都会按当前有效抬升高度和目标高度误差奖励，松手、弹飞或落回桌面时相关奖励立即消失。该任务保持既有 222/225 维 actor/critic 观测和 29 维动作接口，可直接初始化同架构的站立 checkpoint。

后续反向课程的初版状态分类器与课程定义位于 `reverse_curriculum.py`。它不划分 29 维原始关节网格，也不按扰动来源编号，而是把任意物理有效的站立状态压缩为手物轴向/横向误差、手腕误差、夹爪开度、关节极限风险、base 倾斜与速度、手物相对速度等任务坐标。严格且动态稳定的抓握属于 Stage 0；其他状态以最差归一化误差作为主要难度，再用较小权重叠加其余误差的平均值，然后映射到 Stage 1--20。该融合不会让平均值稀释最危险的误差，但能区分“单项困难”和“多项同时困难”。课程 1--19 的 reset 状态以 30%/60%/10% 从前一、当前和后一难度状态库采样，Stage 20 以 30%/70% 从 Stage 19/20 采样。每一级只根据当前难度初始化样本的成功率升级：机器人必须抓住物体并在 8 cm 目标高度连续保持 5 秒，满 4096 个当前难度 episode 后成功率严格超过 80% 才开放下一级，避免简单回放样本虚高升级指标。初始权重、边界和评估窗口可在后续用 rollout 数据校准；当前不随机物体尺寸、质量或摩擦，也尚未把状态库接入 reset 或启用升级奖励。

IK 锚点构建采用“候选生成后物理筛选”：`build_reverse_curriculum_anchors.py` 固定稳定站立腿部姿态，只用全身 IK 调节躯干和右臂，并沿抓取轴、横向平面、三个手腕旋转轴及夹爪开度生成分散候选；速度型状态留给真实 rollout 扩展。`validate_reverse_curriculum_anchors.py` 将候选完整恢复到包含桌面和自由物体的 MJLab 环境，以训练 Stage0 时实际使用的稳定站立 actor 控制，重新计算真实接触和难度，并要求最终连续稳定站立 5 秒。每个 Stage 最终只保留 10 个经验证且最分散的永久锚点。生成的 `.pt` 状态库属于实验数据，不提交 Git。

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

更完整的研究目标、训练路线、奖励定义与当前完成度见
[`docs/project_overview_zh.md`](docs/project_overview_zh.md)。

## 本地检查

```bash
uv sync --extra cpu
uv run ruff check .
uv run pytest
```

## 集群小规模训练与可视化

项目使用 IAS Cluster 已验证的共享 MJLab 环境，不构建新镜像。项目专用脚本位于 `slurm/`：

- `mjlab-grasp-train.sbatch`：默认在学生分区的一张可用 GPU 上训练 256 个立方体抓取环境；
- `mjlab-grasp-livestream.sbatch`：等待第 25 个 checkpoint，并在独立 CPU 作业中打开一个 Viser 环境；
- `train_grasp.py`：从源码 checkout 显式注册项目任务后进入 MJLab 训练 CLI；
- `play_grasp_livestream.py`：加载 actor checkpoint 的可视化入口。
- `mjlab-build-anchors.sbatch`：生成每级候选并用稳定站立模型筛选，每级最终保留 10 个永久锚点。

两个作业都通过 SLURM 运行，livestream 默认监听计算节点的 `18080` 端口，需从本地经登录节点建立 SSH tunnel。

## 设计原则

- 新项目具有独立 Git 历史，不依赖旧项目的 Python 包；
- 不复制旧 checkpoint、训练日志、托盘奖励或实验报告；
- 与集群有关的通用脚本继续由 `Shared-IAS` 管理；
- 机器人模型的上游来源和许可证保留在模型目录中。
