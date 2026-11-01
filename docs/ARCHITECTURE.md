# Project C 结构说明

更新：2026-11-01（M12）。每新增一个文件，就在这里补上它的职责。

## 目录树

```
projectC_learning_assisted_control/
├── README.md                    英文项目首页：问题、结果、如何运行和复现（对外）
├── reproduce.py                 一条命令按顺序重跑全部实验、图、表、报告和视频
├── pyproject.toml               包定义和依赖
├── .gitignore
├── docs/
│   ├── RESEARCH_PLAN.md         原中文 README：研究问题、方法、实验规则、参考论文（内部）
│   ├── REPORT_OUTLINE.md        报告大纲（内部）
│   ├── PLAN.md                  排期、里程碑验收标准、砍范围顺序
│   ├── PROGRESS.md              每一步做了什么、结果、下一步
│   ├── TECHSTACK.md             工具、版本、选择理由、常用命令
│   └── ARCHITECTURE.md          本文件
├── src/lac/
│   ├── __init__.py
│   ├── assets/
│   │   └── two_link_arm.xml     MuJoCo 模型：两连杆、负载、目标点、电机、相机
│   ├── arm.py                   真实机械臂一侧：仿真模型的读写、三种世界、带摩擦的单步推进
│   ├── model.py                 控制器一侧：手写运动方程（含质心偏移和摩擦）、Belief
│   ├── predictors.py            控制器一侧：可替换的预测部件
│   ├── control.py               控制器一侧：逆运动学、PD
│   ├── mpc.py                   控制器一侧：采样式 MPC
│   ├── data.py                  运动数据采集、按段划分、预测误差的度量
│   ├── estimate.py              C1：最小二乘估计负载质量
│   ├── learned.py               C2、C3：残差神经网络的训练、NumPy 推理、预测部件；集成预测部件
│   ├── methods.py               五种方法和集成的组装（读缓存的网络）、测试任务生成
│   ├── demos.py                 M8：空载手臂上录专家示范（到达和纠偏两种回合）
│   ├── policy.py                M8：模仿策略（CFDP、最近邻）和把它们接到机械臂上的策略控制器
│   ├── diffusion.py             M9b：标准 Diffusion Policy（一维卷积 U-Net、DDIM 采样），接口同 CFDP
│   ├── monitor.py               M9：监测器的信号（残差、动作块一致性）、CUSUM、conformal 阈值、测量噪声
│   ├── handover.py              M10：在线检测器、报警后交给自适应 MPC 的接管控制器
│   ├── panda.py                 M11：Panda 平台（真实机器人、控制器的 MuJoCo 模型、三种世界、到达任务）
│   ├── panda_control.py         M11：Panda 上规划目标位置的 MPC、在线负载估计、C0/C1/参考上界
│   ├── panda_view.py            M11：Panda 的 3D 交互查看器和 C0/C1 对比图
│   ├── panda_policy.py          M12：Panda 上的示范、CFDP 的观测和动作、策略控制器
│   ├── panda_monitor.py         M12：Panda 上的监测器（物理、学习、上界、一致性）和报警后接管
│   ├── simulate.py              把控制器接到机械臂上跑一次并记录
│   └── view.py                  交互面板入口
├── experiments/
│   ├── m1_pd_reach.py           M1 实验：三种控制器 × 四档负载
│   ├── m2a_prediction.py        M2a 实验：模型预测对比实际，负载给对和给错
│   ├── m2b_mpc_unloaded.py      M2b 实验：空载下 MPC 对比 PD，改预测时长和候选数量
│   ├── m3_c0_baseline.py        M3 实验：模型按空载算的 MPC 对比告知真实负载的 MPC
│   ├── m4_worlds_and_c1.py      M4 实验：三种世界的差距检查；C1 的估计值和预测误差
│   ├── m5_dev_choices.py        M5 开发：只用开发数据选网络设置，结果冻结
│   ├── m5a_prediction.py        M5a 实验：五种方法的预测对比，训练好的网络存入缓存
│   ├── m5b_control.py           M5b 实验：五种方法在新测试任务上的控制对比；单独测计算耗时
│   ├── m6a_outside.py           M6a 实验：训练区域内外的控制对比、预测误差和乐观程度、最差轨迹
│   ├── m6b_dev_spread.py        M6b 开发：只用开发材料选分歧惩罚力度，结果冻结
│   ├── m6b_ensemble.py          M6b 实验：五网络集成（只取平均、加惩罚）对比单个网络
│   ├── m8_policy.py             M8 实验：空载上学的策略在负载下、区域内外的表现，对照 M6a 的 MPC
│   ├── m9_monitors.py           M9 实验：四种监测器在同一误报率下的检出率和报警时间
│   ├── m9b_train.py             M9b：在 M8 的示范上训练 Diffusion Policy（每个世界一个）
│   ├── m9b_diffusion.py         M9b 实验：开发检查；用 M9 的分析代码重跑整套监测对比
│   ├── m10_handover.py          M10 实验：策略、自适应 MPC、参考上界和五种触发的接管
│   ├── m11a_platform.py         M11a 测量：Panda 的下垂、预测是否精确、推演速度、场景截图
│   ├── m11c_compare.py          M11c 实验：Panda 上 C0、C1、参考上界，区域内外、三种世界、三档负载；`--gains soft` 是 M12a
│   ├── m12_demos.py             M12：录 Panda 的示范（两个机器人 × 两档刚度）
│   ├── m12c_monitors.py         M12c 实验：Panda 上的 CFDP 和四种监测器；`--dev` 开发检查
│   └── m12c_handover.py         M12c 实验：策略、自适应 MPC、参考上界和三种触发的接管
├── tests/
│   ├── test_arm.py              仿真模型的几何和静力检查
│   ├── test_model.py            手写模型逐项对仿真器
│   ├── test_control.py          逆运动学、PD 行为检查
│   ├── test_mpc.py              MPC 能到达、守力矩上限、按周期重算、可重复
│   ├── test_data.py             数据采集、按段划分、C1 估计
│   ├── test_learned.py          网络输入、NumPy 与 PyTorch 一致、训练有效、保存读取
│   ├── test_methods.py          测试任务的区域和可重复性、缺网络时报错
│   ├── test_policy.py           训练样本、目标加权、两个策略、策略控制器、示范录制
│   ├── test_monitor.py          残差、一致性、CUSUM、阈值、测量噪声
│   ├── test_diffusion.py        噪声表、缩放、网络形状、训练和缓存、输出范围、接进策略控制器
│   ├── test_handover.py         在线与离线分数一致、接管、自适应 MPC、在线负载估计
│   ├── test_panda.py            两连杆上 gravcomp 等于手写重力、告知真实时预测精确、下垂、批量与多线程一致、任务
│   ├── test_panda_control.py    Panda 上的 C0/C1/参考上界、负载估计精确、目标位置不超关节范围、可重复
│   └── test_panda_monitor.py    Panda 监测残差、监测网络、策略控制器、在线与离线分数一致、接管、校准
├── results/
│   ├── m0/                      M0 面板截图
│   ├── m1/                      M1 结果表、误差曲线、面板截图
│   ├── m2a/                     M2a 结果表、预测误差图
│   ├── m2b/                     M2b 逐次运行表、汇总表、误差曲线、设置对比图
│   ├── m3/                      M3 逐次运行表、汇总表、误差随负载和随时间的图
│   ├── m4/                      M4 差距检查、估计值、预测误差表和图
│   ├── m5/                      M5 设置选择记录、选中的设置、预测和控制对比表和图、计算耗时
│   ├── m6/                      M6 区域内外对比、最差轨迹、惩罚力度选择、集成对比、计算耗时
│   ├── m8/                      M8 逐次运行表、汇总表（含 M6a 的 MPC 对照）、对比图
│   ├── m9/                      M9 逐次报警表、汇总表、检出率图（含噪声放大 3 倍的版本）
│   ├── m9b/                     M9b 训练损失、开发检查、逐次报警表、汇总表、DP 与 CFDP 对比图
│   ├── m10/                     M10 逐次运行表、汇总表、接管对比图
│   ├── m12a/                    M12a soft 刚度下的逐次运行表、汇总表和图
│   ├── m12c/                    M12c 开发检查、监测的逐次表和汇总表、检出率图；接管的逐次表、汇总表和图
│   └── m11/                     M11 平台测量表、场景截图、C0/C1 对比图、对比实验的逐次运行表、汇总表和图
├── report/
│   ├── main.tex, main.pdf       技术报告（IEEE 双栏，5 页）
│   ├── references.bib           参考文献（逐篇核对过）
│   ├── figures.py               从保存的结果表生成报告用图
│   ├── readme_figures.py        README 的两段动图（重放真实 Panda 测试运行）和结果总览图
│   ├── figures/                 报告图（PDF 和 PNG）
│   ├── tables.py                从保存的结果表生成报告用表
│   ├── generated/               生成的 LaTeX 表
│   ├── video.py                 重放真实运行并渲染演示视频
│   └── demo.mp4                 演示视频（55 秒）
├── third_party/
│   └── franka_emika_panda/      MuJoCo Menagerie 的 Panda（不带夹爪的 XML、网格、LICENSE、README、MENAGERIE_COMMIT）
├── cache/                       采过的数据和训练好的网络（可重新生成，不入库）
│   ├── nets/                    M5a 的 54 个网络，M6b 的 216 个集成成员，以及开发用的 60 个
│   ├── demos_v3_*.npz           M8 的示范，每个世界一份
│   ├── m9_runs.pkl              M9 的全部运行（每周期的状态、力矩、动作块），供重新分析
│   ├── dp/                      M9b 训练好的 Diffusion Policy 网络
│   ├── m9b_runs.pkl             M9b 的全部运行
│   ├── panda_demos_*.npz        M12 的 Panda 示范（W1、W3 × stiff、soft）
│   ├── panda_nets/              M12 学习残差监测器的网络
│   ├── m12c_runs.pkl, m12c_clean.pkl  M12c 的全部运行；其中的开发和校准运行
│   └── m10_clean.pkl            M10 拟合检测器用的空载策略运行
└── .venv/                       虚拟环境，不入库
```

## 各文件职责

| 文件 | 职责 | 对外提供 |
|---|---|---|
| `assets/two_link_arm.xml` | 定义「真实机械臂」：连杆尺寸和质量、关节、电机力矩上限、末端负载体、目标点、侧视相机 | 由 `arm.load_arm` 读取 |
| `arm.py` | 对仿真模型的全部读写都经过这里。推进仿真必须用 `step`，它负责加库仑摩擦 | `load_arm`、`set_world`、`WORLDS`、`step`、`friction_torque`、`set_payload`、`get_payload`、`set_joint_angles`、`set_target`、`tip_position`、`target_position`、`forward_kinematics`、`gravity_torque` |
| `model.py` | 控制器对机械臂的认识：空载参数和运动方程。不确定的部分由 `Belief` 给出（负载质量、质心偏移、额外摩擦）；只给一个数时表示质点在末端、没有额外摩擦。不读仿真器 | `Belief`、`as_belief`、`mass_matrix`、`velocity_terms`、`gravity`、`damping`、`coulomb_friction`、`acceleration`、`predict`、`rollout`，以及各参数常量 |
| `predictors.py` | 「状态加上保持一段时间的力矩，之后会到哪」。MPC 只通过它预测，各方法的区别就是交给 MPC 的预测部件不同 | `PhysicsPredictor`（`step`、`hold_torque`） |
| `control.py` | 控制算法。重力补偿引用 `model.gravity`。不读仿真器 | `inverse_kinematics`、`nearest_solution`、`gravity_model`（即 `model.gravity`）、`PDController` |
| `mpc.py` | `show_plan` 控制是否预测整段计划（只为画图，实验里关掉）；`spread_weight` 惩罚预测部件的分歧（M6b）。采样式 MPC。每个控制周期生成候选力矩序列，用预测部件批量推演，打分后加权合成。带随机种子，可重复。不读仿真器 | `MPCController`（`start`、`retarget`、`torque`、`act`，属性 `predictor`、`planned_tip`、`solve_seconds`）、`tip_of` |
| `data.py` | 用不修正的 MPC 加随机扰动采集运动数据；缓存到 `cache/`；按整段划分；衡量一个预测部件对记录数据的解释程度 | `collect`、`load_or_collect`、`Transitions`、`split_by_episode`、`random_target`、`REGIONS`、`one_period_velocity_error`、`horizon_tip_errors` |
| `estimate.py` | C1：高斯-牛顿最小二乘估计负载质量；M10 加了从一次运行自己的周期（逐步力矩）估计的版本 | `payload_mass`、`payload_mass_from_periods` |
| `methods.py` | `build_ensembles` 组装 C2-E、C3-E（成员 0 是 M5 的网络，成员 1 到 4 只换随机初始值）。按世界、负载、数据量、数据种子组装 C0、C1、C2、C3、参考上界的预测部件；网络只从缓存读，缺了报错（除非明确允许训练）；生成指定区域内的到达任务 | `build_predictors`、`make_tasks`、`frozen_setting`、`net_path`、`METHODS` |
| `learned.py` | `EnsemblePredictor`：多个残差网络叠成一次批量计算，预测取平均，每步后在 `last_spread` 里记下末端位置的分歧。残差网络：输入角度的正弦余弦、角速度、力矩，输出物理模型对 0.02 秒后状态的预测偏差。PyTorch 训练（提前停止），NumPy 推理。`ResidualPredictor` 把物理预测部件和网络合成一个新的预测部件 | `features`、`residual_targets`、`train_residual`、`ResidualNet`（`save`、`load`）、`ResidualPredictor` |
| `demos.py` | 专家是知道真实世界的参考 MPC，手臂空载。到达回合（3 秒）和纠偏回合（1.5 秒，起点在目标 0.5–8 厘米外），每周期记录关节角、角速度、目标、力矩；并行录制并缓存 | `Demos`、`record`、`load_or_record`、`EPISODE_SECONDS`、`SETTLE_OFFSET` |
| `policy.py` | 观测是每周期 8 个数（角度的正弦余弦、角速度、目标减末端，后两个乘 `target_weight`），动作是未来 8 个周期相对当前的关节位置。`NearestNeighborPolicy` 抄最近示范的动作块；`CFDP` 用示范直接算去噪方向，积分反向常微分方程。`PolicyController` 接口同 MPC：每 `n_exec` 个周期生成一块，底层 PD 加重力补偿（按 `assumed_payload`，默认 0）跟踪；记录每块（`chunks`、`chunk_periods`）和每周期的命令位置（`setpoints`） | `frame`、`training_pairs`、`Normalizer`、`ChunkIndex`、`local_mahalanobis_metric`、`NearestNeighborPolicy`、`CFDP`、`PolicyController` |
| `diffusion.py` | Chi 等 2023 的 CNN 版 Diffusion Policy：`ConditionalUNet1D`（FiLM 条件的残差块）、平方余弦噪声表、`Scaler`（缩放到 [−1, 1]）、`DiffusionPolicy`（`fit` 训练或读缓存，`predict` 10 步 DDIM）。观测和动作复用 `policy.training_pairs` | `DiffusionPolicy`、`ConditionalUNet1D`、`Scaler`、`cosine_alpha_bar`、`CACHE` |
| `experiments/m9b_diffusion.py` | 导入 `m9_monitors` 的任务、分析和汇总代码；自己的 `run_group` 按世界、组别、负载分任务并行（每进程 PyTorch 单线程）；CFDP 的数字从 `results/m9/summary.csv` 读。`--dev` 开发检查，`--analyse-only` 只重新分析 | 写入 `results/m9b/` |
| `experiments/m8_policy.py` | 每个世界录（或读缓存）示范；NN 1 次、CFDP 3 个种子，在 M6a 的区域内外任务上、空载和 1 kg 下运行；记录最终误差、稳定时间、命令与实际的差、生成耗时；把 M6a 的 C0、C3（90 秒）、参考上界抄进汇总表。`--plot-only` 只重画图 | 并行运行，写入 `results/m8/` |
| `tests/test_policy.py` | 训练样本里动作块是未来位置减当前；目标加权只改后两维；NN 对示范的观测返回该示范的动作块；CFDP 同种子结果相同、噪声小时接近示范；空载时控制器能稳住；按空载补偿时 1 kg 会下垂，补偿对了就不会；示范的两种回合长度和起点距离 | 命令 `python -m pytest` |
| `monitor.py` | 监测器只看机器人自己有的东西：关节角、角速度、发出的力矩、策略生成的动作块。`periods` 把一次运行切成控制周期；`measured` 加测量噪声；`predict_period_speeds` 从实测的周期起点按逐步力矩积分（残差网络的修正用周期平均力矩）；`residual_signal`、`chunk_disagreement` 每次检查一个数；`Cusum` 是 B 的累积方式；`conformal_threshold`、`episode_max`、`first_alarm` 是 B 的校准和报警 | `Periods`、`periods`、`measured`、`predict_period_speeds`、`period_residuals`、`residual_signal`、`chunk_disagreement`、`Cusum`、`conformal_threshold`、`episode_max`、`first_alarm` |
| `experiments/m9_monitors.py` | 先仿真（开发、校准、区域内、区域外四组）并存进缓存；再加噪声、算信号、在开发组上拟合标准化、在校准组上定阈值、评估测试组；还算「残差或一致性」各占 α/2 的合用。`--analyse-only`、`--noise-scale k` | 并行运行，写入 `results/m9/` |
| `tests/test_monitor.py` | 模型说对时残差为零、不知道摩擦时不为零；负载在残差里可见；网络输出为零时等于物理残差；检查之间取平均；噪声大小和可复现；一致性只比重叠部分；CUSUM 在变化后增长；阈值的分位数和数据不够时为无穷；首次报警 | 命令 `python -m pytest` |
| `handover.py` | `Detector` 是拟合、校准好的监测器，`alarms` 只看最近一次检查；`fit_detector` 按 M9 的做法拟合；`HandoverController` 接口同 MPC：策略阶段记录带噪声的测量和每步力矩，每次策略生成动作块时问检测器，第一次报警就建一个 MPC，负载按已测到的周期估计并每 10 个周期更新。不给策略时就是自适应 MPC | `Detector`、`fit_detector`、`residual`、`consistency`、`HandoverController`（属性 `handover_period`、`alarmed_by`、`mass_estimates`） |
| `panda.py` | Panda 平台。`Robot` 是真实机器人：关节阻抗（Franka 示例增益，弹簧部分受力矩上限）加按配置负载（`configure_load`，即控制器的认识）算的重力补偿；`step` 返回实际加的力矩。`Predictor` 是控制器的模型：同一 MuJoCo 模型的第二份，负载和摩擦按 `Belief`，用 `gravcomp` 自己补偿重力，`rollout` 用 MuJoCo 多线程批量推演目标位置序列，返回每周期末的末端和关节速度。两边都用 Euler 积分，告知真实时逐步一致。`run_reach` 每步向控制器要目标位置（`setpoint`）并按控制器的 `load` 配置机器人 | `Robot`、`Predictor`、`Belief`、`WORLDS`、`world_belief`、`build`、`set_payload`、`Rollout`、`run_reach`、`HoldController`、`make_tasks`、`REGIONS`、常量 `STIFFNESS`、`DAMPING`、`TORQUE_LIMITS`、`HOME`、`PERIOD`、`HOLD` |
| `panda_control.py` | `PandaMPC`：两连杆 MPC 的做法，规划每周期的关节目标位置（不超关节范围），用 `panda.Predictor` 批量推演；机器人配置的负载就是它的认识（`load`）。`PayloadEstimator`：逆动力学（开 `invdiscrete`）对负载质量线性，逐步累加最小二乘。`AdaptiveMPC`：每次到达从 0 kg 开始，每 10 个周期重新估计；通过 `observe` 接收机器人报告的力矩 | `PandaMPC`（`start`、`retarget`、`setpoint`、`set_belief`、`load`、`solve_seconds`）、`PayloadEstimator`（`add`、`mass`、`reset`）、`AdaptiveMPC`（`observe`、`mass_estimates`）、`make_controller` |
| `panda_view.py` | MuJoCo 自带的 3D 窗口（`launch_passive`），自己推进仿真并按实时节奏显示；按键换方法、负载、任务；左上角读数；末端轨迹点。`--snapshot` 渲染 C0 和 C1 在同一任务 2 秒时的并排图 | `python -m lac.panda_view` |
| `panda_policy.py` | Panda 示范（专家 = 告知真实的 MPC，空载；到达和纠偏回合；W2 用 W1 的），观测（24 个数）和动作（专家这一周期起 8 个周期的目标位置减当前关节角），`fit_policy` 把 CFDP/NN 接上，`PandaPolicyController`（按空载配置机器人负载，记录动作块） | `PandaDemos`、`record`、`load_or_record`、`demo_world`、`training_pairs`、`fit_policy`、`frame`、`tip_of`、`PandaPolicyController` |
| `panda_monitor.py` | `Periods`（周期边界的角度、速度、每周期的目标位置）、测量噪声、`MonitorModel`（Predictor 加可选残差网络，按每个周期的起点批量推演一个周期）、监测网络的训练和缓存、残差和一致性信号、`fit_detector`；`PandaHandover`：策略执行、检查、报警后交给 MPC，质量从第一步起估计 | `periods`、`measured`、`features`、`MonitorModel`、`period_residuals`、`residual_signal`、`train_monitor_net`、`load_or_train_net`、`monitor_models`、`residual`、`consistency`、`fit_detector`、`PandaHandover` |
| `experiments/m12c_monitors.py` | 先训练 4 个监测网络；按刚度、世界、组别、负载分任务并行跑 CFDP；按 M9 的做法拟合检测器并评估；`--dev` 只跑开发检查，`--analyse-only` 只重新分析 | 写入 `results/m12c/` 和 `cache/m12c_*.pkl` |
| `experiments/m12c_handover.py` | 每个工作进程用 m12c 的开发和校准运行重新拟合检测器；6 种控制方式在 m12c 的测试任务上运行，MPC 组先跑 | 写入 `results/m12c/handover_*` |
| `experiments/m11c_compare.py` | 864 次运行，每进程单线程、全部核并行；记录最终误差、稳定时间、C1 最后的质量估计、每轮规划耗时；`--plot-only` 只重画图 | 写入 `results/m11/` |
| `experiments/m11a_platform.py` | 20 个起点姿态上测配置负载为 0 时的下垂（三种世界、0/1/2.5 kg）、告知真实时的预测误差、1 和 16 线程的推演耗时；渲染场景截图 | 写入 `results/m11/` |
| `experiments/m10_handover.py` | 先跑空载策略运行（开发 20、校准 100）存进缓存；每个工作进程按世界拟合检测器；8 种控制方式在 M9 的测试任务上运行。`--plot-only` 只重画图 | 并行运行，写入 `results/m10/` |
| `tests/test_handover.py` | 零噪声时在线分数与离线逐位相同；报警后接管并估对负载；没有策略时第 0、10、20 周期估计；从策略的周期估负载；校准数据不够时阈值为无穷 | 命令 `python -m pytest` |
| `simulate.py` | 设置负载和起始姿势，每个仿真步向控制器要力矩，记录全过程。PD 和 MPC 都能接 | `run_reach`、`Rollout` |
| `view.py` | 交互面板：摆姿势、选控制方法（无、PD、PD＋重力补偿、MPC）、调增益、运行中改负载和目标；MPC 时画出预测轨迹；也能存截图或打开自带查看器 | 命令 `python -m lac.view` |
| `experiments/m2b_mpc_unloaded.py` | 6 个任务 × 5 个种子的空载对比，加预测时长和候选数量两组对比。任务 A 是调参用的，其余没参与调参 | 写入 `results/m2b/` |
| `experiments/m3_c0_baseline.py` | 6 个任务 × 4 档负载 × 5 个种子。C0（模型按空载算）、参考上界（告知真实负载）、PD＋重力补偿。同时记录 MPC 的单周期预测误差和顶在力矩上限的时间 | 写入 `results/m3/` |
| `experiments/m4_worlds_and_c1.py` | 每种世界里 C0 对比参考上界；每种世界、多档数据量和负载下 C1 的估计值，以及 C0、C1、参考上界在区域内外验证数据上的预测误差。`--plot-only` 只重画图 | 写入 `results/m4/` |
| `experiments/m5_dev_choices.py` | 12 组网络设置，只在开发数据（数据种子 4、验证种子 500）上评判，最好的一组写入 `results/m5/dev_choice.json` | 并行运行 |
| `experiments/m6a_outside.py` | M5b 的区域内任务加 8 个区域外新任务（种子 6000）；记录 MPC 每轮的预期并算预测误差和乐观程度；检查区域内结果与 M5b 逐一相同；重放最差三次画轨迹 | 并行运行，写入 `results/m6/` |
| `experiments/m6b_dev_spread.py` | 只用开发数据（种子 4）和开发任务（种子 5500、6500）选分歧惩罚力度 | 写入 `results/m6/spread_choice.json` |
| `experiments/m6b_ensemble.py` | 训练集成成员；在 M6a 同样的任务和种子上对比只取平均、加惩罚和单个网络；单独测耗时 | 并行运行，写入 `results/m6/` |
| `experiments/m5b_control.py` | 8 个新测试任务（种子 5000，训练区域内）上，五种方法交给同一个 MPC 比控制；另外单独测每轮计算耗时。`--plot-only` 只重画图 | 并行运行，写入 `results/m5/` |
| `tests/test_methods.py` | 任务起点和目标都在指定区域、相距至少 15 厘米、同种子可重复；缓存里没有的网络会报错 | 命令 `python -m pytest` |
| `experiments/m5a_prediction.py` | 读取冻结的设置；三种世界 × 三档数据量 × 3 个数据种子，训练 C2、C3 并和 C0、C1、参考上界比预测误差（区域内、区域外） | 写入 `results/m5/`，网络存入 `cache/nets/` |
| `tests/test_learned.py` | 角度转一整圈输入不变；网络输出 0 时等于物理预测；拒绝没训练过的步长；NumPy 与 PyTorch 一致；训练降低验证误差；保存读取不变 | 命令 `python -m pytest` |
| `tests/test_data.py` | 每段数据连续、可重复、划分不重叠；告知全部真相时预测误差为 0；W1 里 C1 精确估出质量；W2 里 C1 修不完 | 命令 `python -m pytest` |
| `tests/test_mpc.py` | 空载到达误差小于 0.5 厘米、力矩不超限且周期内保持不变、预测轨迹从当前末端出发、同种子结果相同 | 命令 `python -m pytest` |
| `experiments/m1_pd_reach.py` | 固定起点、目标和增益，比较三种控制器在四档负载下的误差 | 写入 `results/m1/` |
| `experiments/m2a_prediction.py` | 把真实运动中的力矩喂给模型，比较不同预测时长下的末端误差 | 写入 `results/m2a/` |
| `tests/test_arm.py` | 末端位置对公式、保持力矩对手算、松手下落、负载设置 | 命令 `python -m pytest` |
| `tests/test_model.py` | 模型参数和模型文件一致；惯性矩阵、速度项加重力项、加速度、单步预测都和仿真器一致；支持批量 | 同上 |
| `tests/test_control.py` | 逆运动学两组解都到位、重力模型和仿真器一致、PD 的稳态偏差随负载变化、力矩截断、记录的末端和关节角对应 | 同上 |

## 数据怎么流动（M1）

```
                  目标点 (x, z)
                       │
                       ▼
              control.nearest_solution          ← 逆运动学
                       │ 目标关节角
                       ▼
 关节角、角速度 ──► PDController.torque ──► 力矩（已截断）
        ▲                                        │
        │                                        ▼
        └──────────  MuJoCo 仿真一步  ◄──────────┘
                    （arm.py，带真实负载）
```

`simulate.run_reach` 和面板的运行模式都按这个回路走，每个仿真步（0.002 秒）向控制器要一次力矩。控制器只收到关节角、角速度和目标点。

MPC 接在同一个回路里，替换掉 PD 那一格。它每 10 个仿真步（0.02 秒）才重新计算一轮，中间返回同一个力矩：

```
关节角、角速度 ──► 生成 256 组候选力矩序列
                        │
                        ▼
           预测部件批量推演 0.2 秒   ← 默认是 PhysicsPredictor，用假设的负载
                        │
                        ▼
              按末端离目标的距离和速度打分
                        │
                        ▼
              加权合成新计划 ──► 执行第一步，其余留给下一轮
```

## 策略怎么接到机械臂上（M8）

```
关节角、角速度、目标 ──► frame：8 个数 ──► 策略（NN 或 CFDP）
                                            │ 每 4 个周期一次
                                            ▼
                         未来 8 个周期的关节位置（相对当前，加上当前变成绝对位置）
                                            │ 每个周期取一个作为命令位置
                                            ▼
关节角、角速度 ──► PD + 重力补偿（按以为的负载 0 算）──► 力矩 ──► MuJoCo 仿真一步
```

示范是知道真实世界的参考 MPC 在空载手臂上录的。负载出现后，重力补偿不够，手臂跟不上命令位置：这个「命令与实际之差」是 M9 物理残差监测要看的。

## 必须守住的约束

- **真实机械臂和控制器的模型分开。** `arm.py` 和它加载的 MuJoCo 模型代表真实机械臂；`model.py`、`control.py` 和 `mpc.py` 代表控制器的认识。这几个文件只从 `arm.py` 导入连杆长度这个几何常量，不调用 `arm.get_payload`、`arm.gravity_torque`，也不读仿真里的质量。
- **空载参数写了两份，靠检查保持一致。** 一份在 `two_link_arm.xml`（真实机械臂），一份在 `model.py`（控制器的认识）。`tests/test_model.py` 的第一项会在两者不一致时报错。改其中一份时必须想清楚：是真实机械臂变了，还是控制器的认识变了。
- **推进真实机械臂一律用 `arm.step`，不直接调 `mujoco.mj_step`。** 否则 W3 的库仑摩擦会被漏掉。
- **世界设置也是真相。** `arm.WORLDS` 只允许用于设置真实机械臂和构造参考上界，C0、C1、C2 不能读它。
- **仿真器走完一步后，缓存的末端位置还是上一步的。** 记录或显示末端位置时，用 `arm.forward_kinematics(当前关节角)`，或先调用 `mj_forward`。
- **真实负载只通过实验脚本和 `run_reach` 的 `payload` 参数进入仿真。** 把它传给控制器的 `assumed_payload` 只允许用于标明为「参考上界」的方法，目前有三处：`m1_pd_reach.py` 里的 `pd_gc_true`、`m3_c0_baseline.py` 里的 `oracle`、面板里的「MPC（告知真实负载，参考）」。
- **测试任务只用于最终对比。** M5b 的 8 个任务（种子 5000）和 M6a 的 8 个区域外任务（种子 6000）不能拿来调任何设置；调设置只能用开发任务（A、B，种子 5500、6500）和开发数据（种子 4、500）。M8 的策略设置也只在开发任务（种子 5500）上选。
- **接管只用已经发生的数据。** 检测器在策略生成动作块的那一刻判断，用的是到此为止测到的周期和动作块；接手的 MPC 的负载估计也只用这次运行测到的数据。参考上界是唯一知道真实负载的控制方式。
- **监测器不读负载和世界设置。** 物理上界（oracle）监测器的模型知道摩擦，只作上界，和参考上界同理。监测器的模型都按空载构造。开发组（种子 5500）拟合标准化，校准组（种子 7100）定阈值，测试组只用于评估。
- **示范的专家可以读 `arm.WORLDS`。** 它就是参考上界那个 MPC，只在空载手臂上录示范。策略本身和策略控制器不读世界设置，也不知道负载。
- **网络设置只在开发数据上选，选完冻结。** 开发数据是数据种子 4 和验证种子 500；报告用的是数据种子 1 到 3 和验证种子 1000。两者不能混用。
- **所有方法共用同一个 MPC 和同一组设置，只有模型里的负载不同。** 设置只能在任务 A 上、用告知真实负载的模型来选，不能看着 C0、C1、C2 的结果去调。
- **`arm.gravity_torque` 只用于面板读数和检查。** 它读的是真实模型，给控制器用就等于泄漏了真实负载。
- 代码和模型文件随里程碑逐步创建，不提前生成空壳模块。
