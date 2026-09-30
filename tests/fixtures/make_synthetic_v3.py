"""生成与天轶 2.0 + Inspire 布局相同（RoboMIND 2.0 天轶经数据平台转换后的布局）的合成 LeRobot v3 数据集（16 维 state/action、camera_top RGB + 深度、任务描述）。

特征写法与 manip-data-platform 的 util/lerobot_helper.py 一致。图像里的色块位置随右臂关节变化，
策略能学到图像与动作的关联，训练 loss 会下降；只用于跑通流程，不代表真实数据。
"""

import shutil
from pathlib import Path

import numpy as np
from loguru import logger

from lerobot.configs.video import DepthEncoderConfig
from lerobot.datasets import LeRobotDataset

JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_pitch", "wrist_roll"]
NAMES = [f"left_{j}" for j in JOINTS] + ["left_gripper"] + [f"right_{j}" for j in JOINTS] + ["right_gripper"]
TASK = "pick up the bottle with the right hand"


def features(height, width):
    result = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": NAMES},
        "action": {"dtype": "float32", "shape": (16,), "names": NAMES},
        "observation.images.camera_top": {"dtype": "video", "shape": (height, width, 3),
                                          "names": ["height", "width", "channels"]},
        "observation.images.camera_top_depth": {"dtype": "video", "shape": (height, width, 1),
                                                "names": ["height", "width", "channels"],
                                                "info": {"is_depth_map": True}},
    }
    return result


def episode_states(rng, length):
    """右臂从 home 平滑移到随机目标再抬起，夹爪中途闭合；左臂保持不动。"""
    home = np.zeros(16, dtype=np.float32)
    home[7] = home[15] = 1.0
    target = home.copy()
    target[8:15] = rng.uniform(-0.6, 0.6, size=7)
    phase = np.linspace(0, 1, length + 1)[:, None]
    smooth = 0.5 - 0.5 * np.cos(np.pi * np.clip(phase * 1.5, 0, 1))
    states = home + smooth * (target - home)
    states[:, 15] = np.where(phase[:, 0] > 0.6, 0.5, 1.0)
    states = states.astype(np.float32)
    return states


def render(state, height, width):
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    cx = int((0.5 + 0.6 * state[8]) * (width - 1))
    cy = int((0.5 + 0.6 * state[11]) * (height - 1))
    size = 6 if state[15] < 0.75 else 10
    image[max(cy - size, 0):cy + size, max(cx - size, 0):cx + size] = (220, 60, 40)
    depth = np.full((height, width, 1), 800, dtype=np.uint16)
    depth[max(cy - size, 0):cy + size, max(cx - size, 0):cx + size] = 500
    return image, depth


def make_dataset(root, episodes=6, length=60, height=96, width=128, fps=30, seed=0):
    root = Path(root)
    shutil.rmtree(root, ignore_errors=True)
    depth_encoder = DepthEncoderConfig(depth_min=0.0, depth_max=4.095, use_log=False,
                                       extra_options={"x265-params": "lossless=1:open-gop=0"})
    dataset = LeRobotDataset.create(repo_id=f"local/{root.name}", fps=fps, features=features(height, width), root=root,
                                    robot_type="tianyi2_inspire", use_videos=True, depth_encoder=depth_encoder,
                                    video_backend="pyav")
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        states = episode_states(rng, length)
        for t in range(length):
            image, depth = render(states[t], height, width)
            # action[t] = state[t+1]，与数据平台的天轶映射相同。
            dataset.add_frame({"observation.state": states[t], "action": states[t + 1], "task": TASK,
                               "observation.images.camera_top": image,
                               "observation.images.camera_top_depth": depth})
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return root


def main():
    root = make_dataset(Path(__file__).resolve().parents[2] / "data" / "synthetic_tianyi2_inspire")
    logger.info("Synthetic dataset written to {}", root)


if __name__ == "__main__":
    main()
