# 当前进度

更新日期：2026-09-30 04:20。

## 夜间结果（2026-09-29 22:50 – 09-30 04:20，用户要求自行跑完、自行检查）

目标构型：天轶 2.0 + Inspire RH56DFX 灵巧手（`config/robot/tianyi2_inspire.yaml`）。
RoboMIND 2.0 天轶子集中没有 Robotiq 数据（样本画面确认末端是五指灵巧手）。

### 1. 数据：RoboMIND 2.0 天轶 `place_cup_in_box_with_right_hand`

- 用 manip-data-platform 的 `tools/robomind_batch_convert.py` 分批处理：每批约 10 GB H5 → v3 → 验证通过后删除 H5 → 合并。
  三批 65 / 76 / 11 个 episode，每批 22 项验证通过、无跳过；合并验证通过。用时约 1 小时 40 分钟。
- 结果：`data/robomind2_tianyi_place_cup_in_box_with_right_hand`，152 个 episode、63,369 帧、7.5 GB
  （原始 H5 21.3 GB；RGB 视频很小，大部分是无损深度视频）。通过 `manip-train check`。
- 分批中间结果 `data/robomind_parts/`（7.5 GB）在合并数据集验证通过、并抽查批次边界帧可读后已删除；同时删除合成数据 DP 冒烟的训练输出（3 GB）。清理后磁盘剩余 42 GB（96%）。

### 2. 训练（4090 笔记本，GPU 与 face / yolo / GraspGenX 等服务共用）

切分：训练 136 / 验证 16 个 episode（每个任务最后 10%）。

| | ACT | Diffusion Policy |
|---|---|---|
| 步数 / batch | 60,000 / 8 | 40,000 / 8（batch 16 显存不足） |
| 用时 | 1 小时 42 分钟 | 1 小时 36 分钟 |
| 验证 loss | 0.154（5k）→ 0.13–0.14（25k 起持平） | 0.042（5k）→ 0.018（40k） |
| 导出包 | `output/models/tianyi2-inspire-place-cup-in-box-act-20260930-021658` | `output/models/tianyi2-inspire-place-cup-in-box-diffusion-20260930-035840` |

- state/action 的 std 下限作用在第 1、7 维（左臂 shoulder_roll、左手；右手任务中左侧基本不动）。
- DP 图像缩放到 240×320 再随机裁剪 95%；推理 DDIM 10 步。

### 3. 离线评估（5 个验证 episode、1919 帧，开环：观测来自录制数据）

| 模型 / 推理方式 | action MAE（rad；手为 0~1） | RMSE | 对照：保持不动的 MAE |
|---|---|---|---|
| ACT，chunk 开环 50 帧（lerobot 默认） | 0.0223 | 0.0553 | 保持 chunk 起点 state 50 帧：0.0278 |
| ACT，时间集成（coeff 0.01） | 0.0206 | 0.0437 | — |
| ACT，每帧重新推理（`n_action_steps=1`） | **0.0078** | **0.0135** | 保持当前 state：0.0013 |
| DP，每 8 帧推理（DDIM 10） | 0.0083 | 0.0189 | 保持 chunk 起点 state 8 帧：0.0058 |

ACT 2 万 / 4 万 / 6 万步的 checkpoint（chunk 50）：MAE 0.0256 / 0.0221 / 0.0223。
曲线图与逐关节表在 `output/train/<实验名>_report/eval_*/`。

- 右臂 7 个关节和右手：两个模型都跟住了录制轨迹（抬臂、手闭合约第 220 帧、张开约第 300 帧）。
  ACT 每帧重新推理最平滑，但有约 0.03–0.05 rad 的固定偏差（例如右腕 pitch）；DP 偏差小，但有采样抖动，
  episode 137 中右手在第 205 帧附近有一次提前闭合后又张开。
- ACT chunk 开环执行时，静止段每次重新预测都会提前“猜”动作开始，形成锯齿；这是开环执行 chunk 的问题，
  每帧重新推理后消失。所以 ACT 实验改为 `evaluation.policy_overrides: {n_action_steps: 1}`，导出包会记录这个设置，
  部署时默认采用（已导出的 ACT 包是改动前导出的，需要用 `--set n_action_steps=1` 覆盖）。
- **离线指标不能代表任务成功率。** 在 30 Hz 下，关节每帧只动约 0.001 rad，“保持不动”这种不能完成任务的基准
  在很短的时间窗口内误差反而更小。需要闭环评估（仿真或真机）才能判断能不能把杯子放进盒子。

### 4. 部署延迟（4090 笔记本 GPU，`manip-train bench`，200 步）

| | 平均 | p95 | 最大 |
|---|---|---|---|
| ACT chunk 50 | 10.1 ms | 24.4 ms | 29.9 ms |
| ACT 每帧重新推理 | 12.7 ms | 23.2 ms | 40.7 ms |
| ACT 时间集成 | 17.6 ms | 37.2 ms | 71.5 ms |
| DP（每 8 帧一次推理约 110 ms） | 13.9 ms | 106 ms | 119 ms |

30 Hz 每帧 33 ms：ACT 的几种方式平均都满足；DP 的一次推理会卡 3–4 帧，需要异步推理或减少 DDIM 步数。

### 5. π0.5

- 配置构造已验证（从 `lerobot/pi05_base` 加载，输入覆盖为 16 维 state + camera_top）。
- 训练冒烟两次都显存不足：第二次（04:02）其他服务约占 5 GB，本进程在把模型搬上 GPU 时占到 8.2 GB 后失败，
  还没开始训练。本机需要先停掉其他 GPU 服务，否则只能在 H200 上跑。

### 6. 夜间对代码的修改

- 目标构型改为 Inspire（机器人配置、实验模板、合成数据、测试）。
- 离线模式下报告写本地：`ClearMLTracker(report_dir=...)` 保存图（PNG）、表（CSV）、标量（jsonl）、artifact（JSON），
  目录为 `output/train/<实验名>_report/`；`manip-train eval --report-dir`。
- 离线评估加对照基准（保持当前 state、保持 chunk 起点 state）和推理参数覆盖（`evaluation.policy_overrides`）。
- `serve` / `bench` 支持 `--set` 覆盖推理参数；导出包 manifest 记录 `inference_overrides`。
- DP 默认配置：图像缩放 240×320 + 裁剪 95%，DDIM 10 步（合成数据上实测一次推理约 1.1 s → 约 0.1 s）。
- `PYTHONPATH= uv run pytest`：12 项通过（04:05）。

## 此前已验证（2026-09-29）

- uv 环境（torch 2.10.0+cu128，lerobot 0.6.1，clearml 2.1.12），CLI `check / train / eval / serve / bench`。
- 合成数据集与 RoboMIND 单条样本上的 ACT / DP 冒烟；续训（第 10 步 → 第 20 步，数据顺序与 loss 连续）；
  导出包 + 策略服务往返；图像尺寸不一致时服务返回错误。

## 暂缓（用户要求，2026-09-29）

- ClearML 云端上报：代码已写，`.env` 中 `MANIP_TRAIN_CLEARML=false`，报告写本地。
- `--queue h200` 远程执行依赖 ClearML，一并暂缓；H200 上可以直接运行 `uv run manip-train train ...`。
- Jetson Thor 部署。

## 已知问题

- 离线评估是开环比较，不是成功率。tianyi-manip 的 MuJoCo 场景是 Robotiq 夹爪，闭环评估需要 Inspire 手的模型。
- Inspire 手部署映射未定：数据中每只手只有 1 维开合度，RH56DFX 有 6 个自由度（`config/robot/tianyi2_inspire.yaml` 标注 ASSUMED）。
- 代码已推送（2026-09-30）：manip-train 的 main；manip-data-platform 的分批工具在 `robomind-batch-convert` 分支，待合并。数据集不进 Git，H200 上用分批工具重新转换（README“在 H200 上准备”）。
- 多卡训练未验证：`TrainRunner` 训练后的离线评估与导出目前每个进程都会执行，需要改成只在主进程执行。
- 磁盘紧张（剩余 42 GB）：DP 训练输出 12 GB（4 个 checkpoint 含优化器状态）、π0.5 权重缓存在 ~/.cache/huggingface。需要空间时可以只保留导出包（output/models）。
- 本机 `../tianyi/cherry` 与 `../TianYiRobot/cherry` 的 parquet 已损坏；`~/workspace/data/trajectory.hdf5` 是旧版 H5，数据平台读不了。
