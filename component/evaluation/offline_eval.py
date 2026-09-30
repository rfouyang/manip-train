"""离线评估：在验证集 episode 上逐帧运行策略（与部署时相同的 select_action + action chunk），和录制的 action 比较。

这不是闭环成功率：观测来自录制数据，不受策略动作影响。闭环评估见计划中的 MuJoCo 仿真评估。
"""

import time

import numpy as np
import pandas as pd
import torch
from loguru import logger

from util.lerobot_helper import frame_to_batch, load_policy, open_dataset


def run_episode(policy, preprocessor, postprocessor, dataset, max_frames):
    policy.reset()
    predicted, recorded, states, latency = [], [], [], []
    count = min(len(dataset), max_frames)
    for index in range(count):
        frame = dataset[index]
        started = time.perf_counter()
        with torch.inference_mode():
            batch = preprocessor(frame_to_batch(frame))
            action = postprocessor(policy.select_action(batch))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency.append((time.perf_counter() - started) * 1000)
        predicted.append(action.squeeze(0).float().cpu().numpy())
        recorded.append(frame["action"].numpy())
        states.append(frame["observation.state"].numpy())
    result = {"predicted": np.stack(predicted), "recorded": np.stack(recorded), "state": np.stack(states),
              "latency_ms": np.array(latency)}
    return result


def joint_table(runs, names, chunk):
    errors = np.concatenate([run["predicted"] - run["recorded"] for run in runs])
    # 基准：输出当前 state（不动）。action 是下一帧的关节位置，这个基准的误差就是每帧的运动量；
    # 策略的 MAE 要和它比才有意义（开环执行 chunk 时误差会累积，比基准高不一定代表没学到）。
    hold = np.concatenate([run["state"] - run["recorded"] for run in runs])
    # 公平基准：每 n_action_steps 帧取一次 state 并保持不动，与策略开环执行 chunk 的信息量相同。
    hold_chunk = np.concatenate([run["state"][(np.arange(len(run["state"])) // chunk) * chunk] - run["recorded"]
                                 for run in runs])
    table = pd.DataFrame({
        "joint": names,
        "mae": np.abs(errors).mean(axis=0),
        "rmse": np.sqrt((errors ** 2).mean(axis=0)),
        "max_abs": np.abs(errors).max(axis=0),
        "hold_state_mae": np.abs(hold).mean(axis=0),
        "hold_chunk_start_mae": np.abs(hold_chunk).mean(axis=0),
    })
    return table


def episode_figure(run, names, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 4
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.2 * rows), sharex=True, squeeze=False)
    for index, name in enumerate(names):
        ax = axes[index // columns][index % columns]
        ax.plot(run["recorded"][:, index], label="recorded", linewidth=1.2)
        ax.plot(run["predicted"][:, index], label="predicted", linewidth=1.0, linestyle="--")
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
    for index in range(len(names), rows * columns):
        axes[index // columns][index % columns].axis("off")
    axes[0][0].legend(fontsize=7)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure


def evaluate(pretrained_dir, dataset_root, repo_id, episodes, tracker, **kwargs):
    """返回汇总指标；逐关节表格和每个 episode 的曲线图写入 tracker。"""
    max_episodes = kwargs.get("max_episodes", 5)
    max_frames = kwargs.get("max_frames", 600)
    step = kwargs.get("step", 0)
    overrides = kwargs.get("policy_overrides")
    policy, preprocessor, postprocessor = load_policy(pretrained_dir, overrides=overrides)
    chunk = getattr(policy.config, "n_action_steps", 1)
    names = None
    runs = []
    for episode in episodes[:max_episodes]:
        dataset = open_dataset(dataset_root, repo_id, episodes=[episode])
        names = names or dataset.meta.features["action"].get("names") or [
            f"a{i}" for i in range(dataset.meta.features["action"]["shape"][0])]
        run = run_episode(policy, preprocessor, postprocessor, dataset, max_frames)
        runs.append(run)
        figure = episode_figure(run, names, f"episode {episode}: recorded vs predicted action")
        tracker.log_figure("offline_eval/actions", f"episode_{episode}", figure, step)
        logger.info("Episode {}: {} frames, MAE {:.4f}", episode, len(run["recorded"]),
                    float(np.abs(run["predicted"] - run["recorded"]).mean()))

    table = joint_table(runs, names, chunk)
    latency = np.concatenate([run["latency_ms"][1:] if len(run["latency_ms"]) > 1 else run["latency_ms"]
                              for run in runs])
    metrics = {
        "action_mae": float(table["mae"].mean()),
        "action_rmse": float(np.sqrt((table["rmse"] ** 2).mean())),
        "hold_state_mae": float(table["hold_state_mae"].mean()),
        "hold_chunk_start_mae": float(table["hold_chunk_start_mae"].mean()),
        "n_action_steps": int(chunk),
        "latency_ms_mean": float(latency.mean()),
        "latency_ms_p95": float(np.percentile(latency, 95)),
        # select_action 每 n_action_steps 帧才真正推理一次，其余帧只是出队；max 约等于一次 chunk 推理的耗时。
        "latency_ms_max": float(latency.max()),
        "episodes": len(runs),
        "frames": int(sum(len(run["recorded"]) for run in runs)),
    }
    tracker.log_table("offline_eval", "per_joint", table, step)
    tracker.log_scalars(step, "offline_eval", {k: v for k, v in metrics.items() if isinstance(v, float)})
    metrics["policy_overrides"] = overrides
    result = {"metrics": metrics, "per_joint": table.to_dict(orient="records")}
    return result


def main():
    from pathlib import Path

    from component.tracking.clearml_tracker import ClearMLTracker

    root = Path(__file__).resolve().parents[2]
    # 用 app.cli train 跑完冒烟实验后，这里指向它的最后一个 checkpoint。
    checkpoint = root / "output/train/synthetic-act-smoke/checkpoints/last/pretrained_model"
    tracker = ClearMLTracker("manip-train/demo", "offline-eval-demo", enabled=False)
    result = evaluate(checkpoint, root / "data/synthetic_tianyi2_inspire", "local/synthetic_tianyi2_inspire", [5],
                      tracker, max_frames=60)
    logger.info("Metrics: {}", result["metrics"])


if __name__ == "__main__":
    main()
