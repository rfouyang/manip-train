"""实验配置：experiment yaml + robot yaml + policy yaml 合并，命令行 --set 覆盖。"""

import copy
import os
import re
from pathlib import Path

import yaml
from loguru import logger

ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def expand_env(value):
    # 只展开 ${NAME}；未定义的变量直接报错，避免路径静默变成空字符串。
    def replace(match):
        name = match.group(1)
        if name not in os.environ:
            raise KeyError(f"环境变量 {name} 未定义（在 .env 中设置）")
        return os.environ[name]

    result = ENV_PATTERN.sub(replace, value)
    return result


def set_value(data, dotted_key, raw_value):
    """把 "a.b=c" 写入嵌套 dict。lerobot 段的键本身带点（policy.type），整体作为一个键。"""
    value = yaml.safe_load(raw_value) if isinstance(raw_value, str) else raw_value
    if dotted_key.startswith("lerobot."):
        data.setdefault("lerobot", {})[dotted_key.removeprefix("lerobot.")] = value
        return data
    node = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return data


def parse_overrides(items):
    overrides = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"覆盖参数应为 key=value：{item}")
        overrides[key.strip()] = value
    return overrides


def resolve_root(source, config_dir):
    root = Path(expand_env(source))
    # 相对路径以仓库根目录为基准，和 .env 中路径的约定一致。
    if not root.is_absolute():
        root = (Path(config_dir).parent / root).resolve()
    result = str(root)
    return result


def load_experiment(path, config_dir, overrides=None):
    """合并顺序：policy yaml 的 lerobot 段 ← experiment 的 lerobot 段 ← overrides。"""
    path = Path(path).resolve()
    experiment = load_yaml(path)
    for key, value in (overrides or {}).items():
        set_value(experiment, key, value)

    policy = load_yaml(Path(config_dir) / "policy" / f"{experiment['policy']}.yaml")
    robot = load_yaml(Path(config_dir) / "robot" / f"{experiment['robot']}.yaml")
    lerobot_flags = copy.deepcopy(policy.get("lerobot", {}))
    lerobot_flags.update(experiment.get("lerobot") or {})

    dataset = dict(experiment["dataset"])
    # root_source 保留未展开的写法：clearml-agent 在 H200 上重跑时按那台机器的 MANIP_TRAIN_DATA_DIR 重新展开。
    dataset["root_source"] = str(dataset["root"])
    dataset["root"] = resolve_root(dataset["root_source"], config_dir)

    result = {
        "name": experiment["name"],
        "policy": experiment["policy"],
        "robot": robot,
        "dataset": dataset,
        "clearml": experiment.get("clearml", {}),
        "evaluation": experiment.get("evaluation", {}),
        "lerobot": lerobot_flags,
        "source": str(path),
    }
    return result


def demo_load_experiment():
    root = Path(__file__).resolve().parents[1]
    experiment = load_experiment(
        root / "config/experiment/synthetic_act_smoke.yaml",
        root / "config",
        overrides=parse_overrides(["lerobot.steps=50", "evaluation.max_episodes=1"]),
    )
    logger.info("Experiment {}: dataset {}", experiment["name"], experiment["dataset"]["root"])
    logger.info("LeRobot flags: {}", experiment["lerobot"])


def main():
    demo_load_experiment()


if __name__ == "__main__":
    main()
