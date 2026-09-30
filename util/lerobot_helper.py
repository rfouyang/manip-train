"""唯一直接使用 lerobot 内部 API 的地方。lerobot 每个版本的 API 都在变，升级时只改这里（当前 0.6.1）。"""

import contextlib
import json
import math
import sys
from pathlib import Path

import torch
from loguru import logger

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from lerobot.utils.feature_utils import dataset_to_policy_features

LEROBOT_VERSION = "0.6.1"
DATASET_VERSION = "v3.0"
PRETRAINED_MODEL_DIR = "pretrained_model"


def read_info(root):
    with open(Path(root) / "meta" / "info.json", encoding="utf-8") as f:
        info = json.load(f)
    return info


def read_stats(root):
    path = Path(root) / "meta" / "stats.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)
    return stats


def open_metadata(root, repo_id):
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    return meta


def open_dataset(root, repo_id, episodes=None):
    dataset = LeRobotDataset(repo_id, root=root, episodes=episodes)
    return dataset


def split_episodes(meta, eval_split):
    """与 lerobot.datasets.factory.make_train_eval_datasets 相同：每个任务最后 ceil(n * eval_split) 个 episode 做验证。"""
    episode_tasks = meta.episodes["tasks"]
    task_to_episodes = {}
    for index in range(meta.total_episodes):
        key = episode_tasks[index][0] if episode_tasks[index] else ""
        task_to_episodes.setdefault(key, []).append(index)
    train, evaluation = [], []
    for episodes in task_to_episodes.values():
        n_eval = math.ceil(len(episodes) * eval_split)
        train.extend(episodes[: len(episodes) - n_eval])
        evaluation.extend(episodes[len(episodes) - n_eval:])
    return train, evaluation


def cli_args(flags):
    args = []
    for key, value in flags.items():
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            text = json.dumps(value)
        else:
            text = str(value)
        args.append(f"--{key}={text}")
    return args


@contextlib.contextmanager
def lerobot_argv(args):
    # TrainPipelineConfig.validate() 从 sys.argv 读 --policy.path / --config_path，构造配置期间临时替换。
    saved = sys.argv
    sys.argv = ["lerobot-train", *args]
    try:
        yield
    finally:
        sys.argv = saved


def build_train_config(flags, input_keys=None, resume_from=None):
    """flags 与 lerobot-train 命令行一一对应；input_keys 限定策略输入（例如不用深度相机）。"""
    import draccus

    args = cli_args(flags)
    if resume_from is not None:
        config_path = Path(resume_from) / PRETRAINED_MODEL_DIR / "train_config.json"
        args = [f"--config_path={config_path}", "--resume=true", *args]
        with lerobot_argv(args):
            cfg = TrainPipelineConfig.from_pretrained(config_path, cli_args=args[1:])
            cfg.validate()
        return cfg

    from lerobot.configs import parser

    with lerobot_argv(args):
        # 与 lerobot 的 parser.wrap 相同：有 --policy.path 时 --policy.* 不交给 draccus，由 validate() 作为覆盖项加载。
        parse_args = parser.filter_path_args(TrainPipelineConfig.__get_path_fields__(), args)
        cfg = draccus.parse(config_class=TrainPipelineConfig, args=parse_args)
        cfg.validate()
    if input_keys is not None:
        meta = open_metadata(cfg.dataset.root, cfg.dataset.repo_id)
        features = dataset_to_policy_features(meta.features)
        missing = [key for key in input_keys if key not in features]
        if missing:
            raise KeyError(f"数据集中没有这些输入特征：{missing}")
        cfg.policy.input_features = {key: features[key] for key in input_keys}
    return cfg


def floor_std(stats, keys, min_std):
    """std 低于 min_std 的维度抬到 min_std；min/max、分位数不变。返回被调整的 (key, 维度) 列表。

    双臂机器人做单臂任务时，另一只手臂几乎不动（std ~1e-4），MEAN_STD 归一化会把传感器噪声放大上千倍，loss 被噪声主导。
    """
    import numpy as np

    adjusted = []
    for key in keys:
        std = np.asarray(stats[key]["std"])
        low = np.flatnonzero(std < min_std)
        if low.size:
            stats[key]["std"] = np.maximum(std, min_std).astype(std.dtype)
            adjusted.append((key, low.tolist()))
    return adjusted


def run_train(cfg, logger_class, min_std=None):
    """调用 lerobot 官方训练循环；wandb logger 换成 logger_class（ClearML 适配器）。

    min_std：给 state/action 统计量的 std 设下限。统计量随 checkpoint 的前后处理器一起保存，部署时一致；源数据集不改。
    """
    import lerobot.scripts.lerobot_train as lerobot_train

    make_datasets = lerobot_train.make_train_eval_datasets

    def make_datasets_with_floor(train_cfg):
        dataset, eval_dataset = make_datasets(train_cfg)
        if min_std:
            keys = [key for key in ("observation.state", "action") if key in dataset.meta.stats]
            adjusted = floor_std(dataset.meta.stats, keys, min_std)
            logger.info("std floored to {} for {}", min_std, adjusted)
        return dataset, eval_dataset

    cfg.wandb.enable = True
    lerobot_train.WandBLogger = logger_class
    lerobot_train.make_train_eval_datasets = make_datasets_with_floor
    # train() 会再调用一次 cfg.validate()，它读 sys.argv：续训时只给 --config_path，否则给空参数，
    # 不让本进程自己的命令行参数（manip-train / pytest）被误读。
    argv = []
    if cfg.resume:
        argv = [f"--config_path={cfg.checkpoint_path / PRETRAINED_MODEL_DIR / 'train_config.json'}"]
    try:
        with lerobot_argv(argv):
            lerobot_train.train(cfg)
    finally:
        lerobot_train.make_train_eval_datasets = make_datasets


def last_checkpoint(output_dir):
    path = Path(output_dir) / "checkpoints" / "last"
    result = path.resolve() if path.exists() else None
    return result


def load_policy(pretrained_dir, device=None, overrides=None):
    """加载 pretrained_model 目录：policy + 前处理（归一化、tokenizer）+ 后处理（反归一化）。

    overrides 只改推理行为，例如 ACT 时间集成 {"temporal_ensemble_coeff": 0.01, "n_action_steps": 1}。
    """
    from lerobot.configs.policies import PreTrainedConfig

    pretrained_dir = Path(pretrained_dir)
    config = PreTrainedConfig.from_pretrained(pretrained_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    for key, value in (overrides or {}).items():
        if not hasattr(config, key):
            raise KeyError(f"{config.type} 没有参数 {key}")
        setattr(config, key, value)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(pretrained_dir, config=config)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(pretrained_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor


def frame_to_batch(frame):
    """数据集的一帧 → batch 维为 1 的策略输入：只保留 observation.* 和任务描述（π0.5 的语言输入）。"""
    batch = {}
    for key, value in frame.items():
        if key.startswith("observation.") and isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
    batch["task"] = [frame["task"]]
    return batch


def demo_split(root="../data/synthetic_tianyi2_inspire"):
    root = (Path(__file__).parent / root).resolve()
    meta = open_metadata(root, "local/synthetic_tianyi2_inspire")
    train, evaluation = split_episodes(meta, 0.1)
    logger.info("{}: {} train, eval {}", root.name, len(train), evaluation)
    logger.info("Train flags: {}", cli_args({"policy.type": "act", "steps": 10, "dataset.eval_split": 0.1}))


def main():
    demo_split()


if __name__ == "__main__":
    main()
