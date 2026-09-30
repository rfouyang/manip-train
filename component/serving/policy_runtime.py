"""加载导出包并逐帧推理。输入是机器人端的原始观测（HWC uint8 图像、float32 state、任务描述），输出一帧 action。

action chunk 由 lerobot 的 select_action 内部排队：每 n_action_steps 帧推理一次，其余帧直接出队。
"""

import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from component.export.model_bundle import read_manifest
from util.lerobot_helper import load_policy


class PolicyRuntime:
    def __init__(self, bundle_dir, device=None, overrides=None):
        """overrides 只改推理行为；默认用导出包 manifest 中记录的 inference_overrides（评估时用的推理设置）。"""
        self.bundle_dir = Path(bundle_dir)
        self.manifest = read_manifest(self.bundle_dir)
        overrides = overrides if overrides is not None else self.manifest.get("inference_overrides")
        self.policy, self.preprocessor, self.postprocessor = load_policy(self.bundle_dir, device, overrides)
        self.default_task = None
        logger.info("Loaded {} ({}) for {}", self.manifest["name"], self.manifest["policy"], self.manifest["robot"])

    def reset(self):
        """每个 episode 开始时调用：清空 action 队列。"""
        self.policy.reset()

    def to_batch(self, observation):
        batch = {}
        state_key = self.manifest["state_feature"]
        batch[state_key] = torch.as_tensor(np.asarray(observation[state_key], dtype=np.float32)).unsqueeze(0)
        for key, shape in self.manifest["cameras"].items():
            image = np.asarray(observation[key])
            # 分辨率必须与训练数据一致；不在这里静默缩放，避免相机配置变了还继续跑。
            if list(image.shape) != list(shape):
                raise ValueError(f"{key} 尺寸 {list(image.shape)} 与训练数据 {shape} 不一致")
            batch[key] = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        batch["task"] = [observation.get("task") or self.default_task or ""]
        return batch

    def infer(self, observation):
        started = time.perf_counter()
        with torch.inference_mode():
            batch = self.preprocessor(self.to_batch(observation))
            action = self.postprocessor(self.policy.select_action(batch))
        action = action.squeeze(0).float().cpu().numpy()
        result = {"action": action, "latency_ms": (time.perf_counter() - started) * 1000}
        return result


def dummy_observation(manifest):
    observation = {manifest["state_feature"]: np.zeros(manifest["state_dims"], dtype=np.float32)}
    for key, shape in manifest["cameras"].items():
        observation[key] = np.zeros(shape, dtype=np.uint8)
    observation["task"] = "pick up the bottle with the right hand"
    return observation


def benchmark(runtime, steps=100):
    runtime.reset()
    observation = dummy_observation(runtime.manifest)
    latencies = [runtime.infer(observation)["latency_ms"] for _ in range(steps)]
    latencies = np.array(latencies[1:])
    result = {"mean_ms": float(latencies.mean()), "p95_ms": float(np.percentile(latencies, 95)),
              "max_ms": float(latencies.max())}
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    bundles = sorted((root / "output").glob("**/models/*/manifest.json"))
    if not bundles:
        logger.info("No bundle yet; run component/training/train_runner.py first")
        return
    runtime = PolicyRuntime(bundles[-1].parent)
    logger.info("Benchmark: {}", benchmark(runtime))


if __name__ == "__main__":
    main()
