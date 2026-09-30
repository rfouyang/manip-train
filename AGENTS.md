# 项目协作与代码风格

与 manip-data-platform 相同的约定（见其 AGENTS.md），这里只写本项目特有的部分。

## 目录与依赖

- 依赖方向：`app → component → util`；`tests/test_architecture.py` 检查。
- `app/cli.py` 是命令行入口（`uv run manip-train ...`），`app/context.py` 的 `AppContext` 持有配置与组件。
- `component/`：data（数据集检查）、training（一次实验的完整流程）、evaluation（离线评估）、tracking（ClearML）、
  export（导出包）、serving（推理 runtime、WebSocket 服务、客户端）。
- **只有 `util/lerobot_helper.py` 直接 import lerobot**（测试夹具除外）。lerobot 每个版本的 API 都在变，升级时只改这里。
- 训练循环用 lerobot 官方的 `lerobot_train.train()`；ClearML 通过替换其中的 `WandBLogger` 接入，不复制训练循环。

## 配置

- `config/robot/*.yaml`：机器人的 state/action 名称与顺序、策略输入相机、归一化 std 下限。所有 ASSUMED 值在这里标注。
- `config/policy/*.yaml`：各模型默认参数，`lerobot:` 段的键与 `lerobot-train` 命令行参数一一对应。
- `config/experiment/*.yaml`：一次实验 = 数据集 + 机器人 + 模型 + 覆盖参数。合并顺序 policy ← experiment ← `--set`。
- 数据集路径用 `${MANIP_TRAIN_DATA_DIR}` 引用，远程机器（H200）按自己的 `.env` 展开。

## 数据

- 数据集只读；只接受 LeRobot v3.0（由 manip-data-platform 转换）。训练前 `check_dataset` 核对版本、fps、维度名称与顺序、相机、统计量。
- 不静默缩放图像、补零或重排维度；推理时图像尺寸与训练数据不一致直接报错。

## Demo 与测试

- 每个 Python 文件提供 `main()`，IDE 右键可运行；`main()` 不用 argparse（`app/cli.py` 除外）。
- 运行测试：`PYTHONPATH= uv run pytest`（本机 ROS 2 Humble 的 PYTHONPATH 会让 pytest 加载 launch_testing 插件而报错）。
- `tests/test_pipeline.py` 端到端：训练 → 续训 → 评估 → 导出 → 服务往返，约 15 秒（GPU）。

## 安全边界

- 部署目前只到策略服务与仿真。给真机发指令需要单独评审执行边界（速度/工作空间限制、急停、操作员确认），与 tianyi-manip 一致。

## 文档

- 计划：`doc/plan.md`；当前进度：`doc/current_state.md`（只保留这一份）。区分已实现、已验证、计划中和未确认事项。
