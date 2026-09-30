# manip-train

manip-data-platform 的下游：用 LeRobot v3 数据集训练 ACT / Diffusion Policy / π0.5，报告写到 ClearML，
导出模型并通过 WebSocket 策略服务部署（4090 笔记本）。

## 环境

Python 3.12，uv，torch 2.10 + CUDA 12.8（与 st-lerobot 相同）。

```bash
uv sync
cp .env.example .env   # 填 ClearML 凭证；不填时报告只写本地日志
```

## 用法

```bash
# 冒烟：合成数据集（与天轶 2.0 + Inspire 布局相同（RoboMIND 2.0 天轶经数据平台转换后的布局））
uv run python tests/fixtures/make_synthetic_v3.py
uv run manip-train check config/experiment/synthetic_act_smoke.yaml
uv run manip-train train config/experiment/synthetic_act_smoke.yaml

# 覆盖参数、续训、提交到 H200 队列
uv run manip-train train config/experiment/tianyi2_inspire_place_cup_in_box_act.yaml --set lerobot.steps=50000
uv run manip-train train config/experiment/tianyi2_inspire_place_cup_in_box_act.yaml --resume output/train/tianyi2-inspire-place-cup-in-box-act
uv run manip-train train config/experiment/tianyi2_inspire_place_cup_in_box_pi05.yaml --queue h200

# 离线评估（报告写到 --report-dir：曲线图 PNG、逐关节 CSV、metrics.json）
uv run manip-train eval config/experiment/tianyi2_inspire_place_cup_in_box_act.yaml \
    output/train/<实验名>/checkpoints/last/pretrained_model --no-clearml --set "evaluation.policy_overrides={n_action_steps: 1}"

# 部署（--set 覆盖推理参数，例如 ACT 每帧重新推理；不给时用导出包 manifest 的 inference_overrides）
uv run manip-train bench output/models/<bundle> --set n_action_steps=1
uv run manip-train serve output/models/<bundle> --port 8765 --set n_action_steps=1
```

RoboMIND 2.0 天轶数据：用 manip-data-platform 的 `tools/robomind_batch_convert.py` 分批下载、转换、合并，
输出放到 `data/`（`.env` 的 `MANIP_TRAIN_DATA_DIR`）。

机器人端使用 `component/serving/policy_client.py` 的 `PolicyClient`（只依赖 numpy、msgpack、websockets）。

## 在 H200 上准备

数据集不放在 Git 里，在 H200 上用 manip-data-platform 的分批转换工具直接从 ModelScope 下载并转换：

```bash
git clone git@github.com:rfouyang/manip-train.git
git clone -b robomind-batch-convert git@github.com:rfouyang/manip-data-platform.git   # 工具合并到 main 之前用这个分支

# 1. 数据：RoboMIND 2.0 天轶 place_cup_in_box_with_right_hand（152 条，H5 21.3 GB → v3 7.5 GB）
cd manip-data-platform && uv sync --locked
MANIP_OUTPUT_DIR=/data/manip uv run python tools/robomind_batch_convert.py   # 任务列表在 main() 中

# 2. 训练环境
cd ../manip-train && uv sync && cp .env.example .env   # .env 中 MANIP_TRAIN_DATA_DIR=/data/manip
PYTHONPATH= uv run pytest
uv run manip-train check config/experiment/tianyi2_inspire_place_cup_in_box_pi05.yaml
uv run manip-train train config/experiment/tianyi2_inspire_place_cup_in_box_pi05.yaml
```

未验证：多卡（`accelerate launch`）。当前 `TrainRunner` 在训练后做离线评估和导出，多卡时每个进程都会执行，需要先改成只在主进程执行。

## 输出

- `output/train/<实验名>/`：lerobot checkpoint（`checkpoints/<step>/pretrained_model` + `training_state`）、`result.json`。
- `output/train/<实验名>_report/`：本地报告（数据集检查、episode 切分、标量 jsonl、评估图与表）。
- `output/models/<实验名>-<时间>/`：导出包 = pretrained_model + `manifest.json`（数据集、机器人、评估指标、git commit）。
- ClearML：超参、train/eval 曲线、逐关节误差表、预测与录制 action 曲线图、数据集检查与 episode 切分、导出包（OutputModel）。

## 测试

```bash
PYTHONPATH= uv run pytest
```

代码约定见 [AGENTS.md](AGENTS.md)，计划见 [doc/plan.md](doc/plan.md)，进度见 [doc/current_state.md](doc/current_state.md)。
