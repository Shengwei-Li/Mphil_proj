# Project C 技术栈

更新：2026-10-29（M8–M10 没有新依赖；CFDP、最近邻、监测器和接管只用 NumPy）。记录项目实际用到的工具、版本和选择理由。新增依赖时先在这里登记，再写进 `pyproject.toml`。

## 运行环境

| 项 | 取值 |
|---|---|
| 操作系统 | Windows 11 |
| Python | 3.11.9，项目内独立虚拟环境 `.venv/` |
| 硬件假设 | 笔记本 CPU 即可；目前没有用到显卡 |
| 包名 | `lac`（learning-assisted control），源码在 `src/lac/` |

## 已使用

| 工具 | 版本 | 用来做什么 | 为什么选它 |
|---|---|---|---|
| MuJoCo | 3.14.0 | 物理仿真，扮演「真实机械臂」；离屏渲染画面 | 机器人研究里通用的仿真器，官方 Python 接口，CPU 上足够快 |
| NumPy | 2.4.6 | 数组和数值计算 | 基础依赖 |
| Matplotlib | 3.11.2 | 交互面板（滑块、按钮、读数）和后续结果图 | 已是依赖，能显示中文读数；避免为 M0 引入额外界面库 |
| PyTorch（CPU 版） | 2.14.1 | 训练 C2、C3 的残差网络（M5a 起） | 具身智能和机器人学习行业的标准工具；能自动求导；小网络在 CPU 上训练很快。不用 GPU：网络和数据都很小，搬数据到显卡的开销比计算还大 |
| imageio、imageio-ffmpeg | 2.38.1、0.6.0 | 把渲染的帧写成 MP4（M7c 起，可选依赖组 `report`） | imageio-ffmpeg 自带 ffmpeg，不用另外安装 |
| TeX Live 2026（本机已有） | pdfTeX 1.40.29 | 编译报告：`IEEEtran`、`newtx`（Times 字体）、`latexmk` | 本机已安装，不需要再装 Tectonic |
| pytest | 9.1.1 | 自动检查 | 和项目 A、B 一致 |
| setuptools | 构建后端 | 以可编辑方式安装 `lac` | 和项目 A、B 一致 |

## 自己实现、没有用现成库的部分

| 部分 | 位置 | 做法 |
|---|---|---|
| 预测模型 | `model.py` | 手写两连杆运动方程，只用 NumPy，支持批量计算 |
| MPC | `mpc.py` | 采样式，按分数加权合成（MPPI）；256 个候选、0.2 秒预测。模型每轮 10 小步积分时约 11 毫秒，2 小步时约 2.5 毫秒（M4 起默认 10 小步）；加上残差网络后约 19.7 毫秒，95% 分位数约 21 毫秒，超过 20 毫秒的控制周期（M5b 实测） |
| 负载估计 | `estimate.py` | 一维高斯-牛顿最小二乘，只用 NumPy |
| 集成推理 | `learned.py` 的 `EnsemblePredictor` | 5 个网络的权重叠成三维数组，一次批量矩阵乘法算完；单独测每轮约 28 毫秒（C3-E），超出 20 毫秒周期 |
| 模仿策略 | `policy.py` | CFDP 和最近邻从 B 移植，只用 NumPy：近邻用 `argpartition` 全量查找（约 4.5 万条，不需要 scipy 的 KD 树），局部马氏距离，反向常微分方程 20 步。生成一个动作块 NN 约 3 毫秒，CFDP 约 34 毫秒，都在 80 毫秒的预算内 |
| 网络推理 | `learned.py` 的 `ResidualNet` | 训练用 PyTorch，训练完权重复制成 NumPy 数组，MPC 里用矩阵乘法算。MPC 每轮调用 10 次、每次 256 条，这么小的调用 PyTorch 的固定开销比计算本身还大。有检查保证两者输出一致 |

## 计划使用，尚未引入

到对应里程碑再决定并登记，现在不安装。

| 里程碑 | 候选 | 用途 | 待定的事 |
|---|---|---|---|
| M5 | pandas | 逐回合结果表 | 结果量不大时可以只用 CSV 加 NumPy |
| M9b | 自己写的小型 Diffusion Policy（PyTorch）；参考 Chi 等的官方实现和 LeRobot | 第二个模仿策略 | 状态只有 8 维，网络很小，先试 CPU；不够快再决定是否装 GPU 版 PyTorch（约 2–3 GB，需你同意）。不直接装 LeRobot，依赖太多，也不利于讲清每一步 |

## GPU（M6 时评估过，不用）

这台电脑有 RTX 5060 Laptop（8 GB）。实测 MPC 每轮的时间花在约 200 次小计算的调用开销上，每次只算 256 条数据；GPU 上每个小计算的启动开销更大，反而更慢。只有把整轮推演编译成一个整体（例如 JAX）且候选涨到几万条时 GPU 才划算，那需要重写 MPC，JAX 的 GPU 版在 Windows 上还需要 WSL。留到 M8 三维机械臂再考虑。CPU 上改用的两处提速见 PROGRESS 的 M6.0。

## 考虑过但不用的学习方法

| 方法 | 不用的原因 |
|---|---|
| 高斯过程 | 数据少时很强，还自带不确定性（Carron 等 2019 在机械臂上用过）。但 MPC 每轮要查询几千次，它的计算量随数据量增长太快。M6 用网络集成来获得不确定性 |
| 线性回归加手工特征 | 能修掉不少误差，但等于把物理知识手工写进特征，失去「通用学习模型」的意义。可以在报告里作为讨论点 |
| scikit-learn 的神经网络 | 太受限，M6 做集成和自定义打分时不好改 |
| JAX | GPU 版在 Windows 上需要 WSL，现在没必要 |

## 明确不用

| 不用 | 原因 |
|---|---|
| PyTorch Lightning、TensorBoard、wandb | 网络很小、训练几秒，用不上 |
| Gymnasium 环境封装 | 首版不做强化学习，直接调 MuJoCo 更容易看清每一步 |
| MuJoCo MPC（DeepMind 的 C++ 框架）、CasADi、do-mpc | 自己写一个小的采样式 MPC 才讲得清楚；这些只作阅读参考 |
| ROS、真机驱动 | 没有真机 |
| 相机、视觉模型、语言模型 | 不在研究问题内 |

## 常用命令

都在 `projectC_learning_assisted_control/` 目录下运行。

| 目的 | 命令 |
|---|---|
| 首次安装 | `python -m venv .venv`；先装 CPU 版 PyTorch：`.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`；再 `.venv\Scripts\python -m pip install -e ".[dev,report]"` |
| 打开交互面板 | `.venv\Scripts\python -m lac.view` |
| 指定初始姿势和负载 | `.venv\Scripts\python -m lac.view --q1 0 --q2 0 --payload 1.5` |
| 预选控制方法和增益 | `.venv\Scripts\python -m lac.view --method 2 --kp 100 --kd 8`（0 无，1 PD，2 PD＋重力补偿，3 MPC） |
| 重跑 M2b 实验（约 8 分钟） | `.venv\Scripts\python experiments\m2b_mpc_unloaded.py` |
| 重跑 M3 实验（约 10 分钟） | `.venv\Scripts\python experiments\m3_c0_baseline.py` |
| 重跑 M4 实验（约 6 分钟） | `.venv\Scripts\python experiments\m4_worlds_and_c1.py` |
| 只重画 M4 的图 | `.venv\Scripts\python experiments\m4_worlds_and_c1.py --plot-only` |
| M5 选网络设置（并行，约 12 分钟） | `.venv\Scripts\python experiments\m5_dev_choices.py` |
| M5a 预测对比（并行，约 4 分钟；训练好的网络存进 `cache/nets/`） | `.venv\Scripts\python experiments\m5a_prediction.py` |
| M5b 控制对比（并行，5 到 10 分钟；最后单独测计算耗时） | `.venv\Scripts\python experiments\m5b_control.py` |
| M6a 区域内外对比（并行，约 5 分钟） | `.venv\Scripts\python experiments\m6a_outside.py` |
| M6b 选惩罚力度（并行，约 10 分钟） | `.venv\Scripts\python experiments\m6b_dev_spread.py` |
| M6b 集成对比（并行，约 25 分钟，含训练 216 个网络） | `.venv\Scripts\python experiments\m6b_ensemble.py` |
| M8 策略对比（并行，约 3 分钟，首次含录示范约 1.5 分钟） | `.venv\Scripts\python experiments\m8_policy.py`（`--plot-only` 只重画图） |
| M10 接管对比（并行，约 15 分钟） | `.venv\Scripts\python experiments\m10_handover.py`（`--plot-only` 只重画图） |
| M9 监测器对比（并行，约 10 分钟，首次含训练 3 个空载残差网络） | `.venv\Scripts\python experiments\m9_monitors.py`（`--analyse-only` 只重新分析；`--analyse-only --noise-scale 3` 噪声放大 3 倍） |
| 重画报告图（约 1 分钟） | `.venv\Scripts\python reportigures.py` |
| 生成报告结果表 | `.venv\Scripts\python report	ables.py` |
| 编译报告 | `latexmk -pdf -cd report\main.tex` |
| 生成演示视频（约 30 秒） | `.venv\Scripts\python reportideo.py` |
| 一条命令复现全部（约 1.5 到 2 小时） | `.venv\Scripts\python reproduce.py`（`--list` 看步骤，`--outputs-only` 只重建图表、报告和视频） |
| 面板选真实世界 | `.venv\Scripts\python -m lac.view --world W3 --payload 1 --method 4` |
| 面板直接选参考上界 | `.venv\Scripts\python -m lac.view --method 4 --payload 1`（3 是模型按空载算的 MPC） |
| 重跑 M1 实验 | `.venv\Scripts\python experiments\m1_pd_reach.py` |
| 重跑 M2a 实验 | `.venv\Scripts\python experiments\m2a_prediction.py` |
| 只存一张截图 | `.venv\Scripts\python -m lac.view --snapshot results/m0/pose.png` |
| MuJoCo 自带三维查看器 | `.venv\Scripts\python -m lac.view --native` |
| 运行检查 | `.venv\Scripts\python -m pytest -q` |

## 版本管理

| 项 | 取值 |
|---|---|
| 远端仓库 | `git@github.com:Shengwei-Li/Mphil_proj.git`（私有），`main` 分支，仓库根目录就是本项目文件夹 |
| 不入库 | `.venv/`、`cache/`（采集的数据和训练好的网络，可由 `reproduce.py` 重新生成）、LaTeX 中间文件 |
| 提交署名 | 本机 git 配置的 Shengwei-Li；不加 AI 署名（对外材料不提 AI 的决定） |
| 注意 | 本文件夹在 Windows 上属于另一个本机账户，已加入 git 全局可信列表（2026-10-22，你同意的） |

## 约定

- **单位：** 长度米，质量千克，力矩牛·米，时间秒。代码内部角度用弧度，界面和命令行用度。
- **坐标：** 机械臂在 x-z 平面，x 水平向右，z 竖直向上，重力沿 -z。
- **语言：** 代码、注释、文件名用英文；`docs/` 和面板读数用中文。
- **可重复：** 之后凡是带随机性的实验，都要保存随机种子和运行配置。
