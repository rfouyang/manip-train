# manip-train 计划

更新日期：2026-09-29。

## 1. 目标

manip-data-platform 输出的 LeRobot v3 数据集 → 训练 ACT / Diffusion Policy / π0.5 → ClearML 报告 →
导出与注册 → 在 4090 笔记本上作为策略服务运行（Jetson Thor 暂缓，用户要求 2026-09-29）。第一个跑通的目标：天轶 2.0 + Inspire RH56DFX 灵巧手（与 RoboMIND 2.0 天轶子集同一构型；2026-09-29 由 Robotiq 改为 Inspire，用户确认）。

## 2. 关键决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 模型实现 | lerobot 0.6.1 的 `act` / `diffusion` / `pi05` | 都是官方 PyTorch 实现；π0.5 用 `lerobot/pi05_base`（由 openpi 转换） |
| 训练循环 | 直接调用 `lerobot_train.train()`，把其中的 `WandBLogger` 换成 ClearML 适配器 | 续训、多卡（accelerate/FSDP）、PEFT、验证 loss 都用官方实现，不复制约 600 行训练循环 |
| lerobot 隔离 | 只有 `util/lerobot_helper.py` 直接 import lerobot | 升级 lerobot 时只改一个文件；有架构测试 |
| 配置 | `lerobot:` 段的键与 `lerobot-train` 命令行参数一一对应 | 看配置就知道等价的 lerobot 命令，官方文档可以直接对照 |
| 数据集 | 只接受 v3.0，训练前核对 fps、维度名称与顺序、相机、统计量 | 维度相同但顺序不同会静默训练出错误的策略 |
| 归一化 | state/action 的 std 设下限（机器人配置 `normalization.min_std`，默认 0.01） | 双臂做单臂任务时，另一只手臂的 std 约 1e-4，噪声被放大上千倍（RoboMIND 样本上实测 L1 loss 从 45 降到 0.25） |
| 深度 | 暂不作为策略输入（机器人配置里 `cameras` 只列 RGB） | ACT/DP/π0.5 都以 RGB 为主；深度留在数据集中 |
| 追踪 | ClearML 云端；远程训练用 `task.execute_remotely(queue)` 交给 H200 上的 clearml-agent | 需求指定 |
| 部署 | 导出包（pretrained_model + manifest）→ ClearML OutputModel → WebSocket + msgpack 策略服务 | 与 openpi serve_policy 相近，机器人端客户端很薄 |

### π0.6

官方没有开源：openpi 只有 π0、π0-FAST、π0.5；询问 π0.6 的 issue（#791，2025-11）至今没有回复；
lerobot 0.6.1 也没有 pi06。第三方复现 OpenPIE-0.6（带 RECAP）的权重不是官方的，放在 P5 评估。

### RoboMIND 2.0（2026-09-29 调研）

- 天轶子集 `X-Humanoid/RoboMIND2.0-Tianyi`（ModelScope）：36 个任务、7145 条、约 1.9 TB；另有 mobile 子集（10 个任务、1779 条）。
- 末端是 Inspire RH56DFX 五指灵巧手（样本画面确认），每只手只记 1 维开合度；没有 Robotiq 数据（Robotiq 只在 UR5e / Franka 上）。
- H5 布局与 manip-data-platform 的 `config/x_humanoid.json` 一致：1 条样本经平台原样转换为 v3，通过 `tianyi2_inspire` 检查，ACT 训练正常。
- 只有一路头部相机 camera_top（D435if），没有腕部相机。
- 首个正式任务：`place_cup_in_box_with_right_hand`（152 条，H5 约 21 GB）。数据平台里已有的
  `place_red_pepper_on_red_tray`（294 条，约 83 GB）也可直接用。
- 许可证：论文写 CC BY 4.0，ModelScope 写 Apache-2.0，商用前要确认。

## 3. 资源

| 模型 | 4090 笔记本 16GB | 正式训练 |
|---|---|---|
| ACT | 可训练；640×480、batch 8 约 2 GB 显存、0.095 s/步（6 万步 1 小时 42 分钟） | 本机 |
| Diffusion Policy | 可训练；缩放到 240×320、batch 8 约 5 GB 显存、0.11 s/步（batch 16 显存不足） | 本机或 H200 |
| π0.5（约 3B） | 显存被其他服务占用约 7.6 GB 时无法冒烟；空闲时可尝试只训练 action expert、batch 1 | H200 |

推理（4090 笔记本）：ACT 每帧重新推理平均 12.7 ms；DP DDIM 10 步一次约 110 ms。详见 current_state.md。

## 4. 阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 骨架 | uv 环境、分层与架构测试、配置合并、数据集检查、合成数据集 | 已完成，已验证 |
| P1 ACT + DP | lerobot 训练接入、续训、离线评估（含对照基准）、导出、本地报告 | 已完成；RoboMIND 天轶 152 条真实数据上训练完成（2026-09-30） |
| P2 π0.5 | 配置构造已验证；本机显存不足（与其他 GPU 服务共用），需停服务或上 H200 | 进行中 |
| P2 H200 | 训练镜像、多卡 accelerate（ClearML 队列暂缓） | 未开始 |
| P3 部署 | 4090 笔记本上的策略服务、客户端、延迟测试已完成；闭环评估未开始（需要 Inspire 手的仿真模型或真机） | 部分完成 |
| P4 自动化 | 数据平台发布新数据集 → 训练 → 评估 → 导出 | 未开始 |
| P5 研究 | OpenPIE-0.6 / RECAP；更多 RoboMIND 天轶任务联合训练 | 未开始 |

## 4a. 下一步建议

1. 闭环评估：tianyi-manip 的 MuJoCo 场景换上 Inspire RH56DFX 手模型，或者直接上真机测试（需要先评审执行边界）。
2. 更多数据：同样的分批工具转换其他右手任务（例如 `place_peach_in_blue_plate_with_right_hand` 296 条、
   `place_red_apple_in_basket_with_right_hand` 147 条），做多任务训练；π0.5 可以把 36 个任务都用上。
3. π0.5：停掉本机其他 GPU 服务后冒烟，或者直接在 H200 上训练。
4. ACT 的固定偏差：可以试相对动作（action − state），lerobot 的 π0.5 已有 `use_relative_actions`。

## 5. 待确认

1. ClearML：用户要求暂缓（2026-09-29）。报告先写本地日志；接入时再要凭证和 H200 队列名（先假设 `h200`）。
2. Inspire 手部署映射：1 维开合度 → RH56DFX 6 个手指角度的规则（先假设各手指同步、拇指旋转固定）。
3. 是否会有腕部相机数据；有的话在 `config/robot/tianyi2_inspire.yaml` 的 `cameras` 中加上。
4. Jetson Thor：用户要求暂缓（2026-09-29）。
5. RoboMIND 2.0 的许可证。
