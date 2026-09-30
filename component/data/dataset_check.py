"""训练前检查数据集：LeRobot v3.0、fps、state/action 维度与名称、相机、统计量、任务描述。"""

from pathlib import Path

from loguru import logger

from util.lerobot_helper import DATASET_VERSION, read_info, read_stats


def expected_names(robot, part):
    spec = robot[part]
    names = spec.get("names")
    if names == "same_as_state":
        names = robot["state"].get("names")
    return names


def vector_issues(info, stats, robot, part):
    issues = []
    spec = robot[part]
    key = spec["feature"]
    feature = info["features"].get(key)
    if feature is None:
        issues.append(f"缺少 {part} 特征 {key}")
        return issues
    names = expected_names(robot, part)
    dims = len(names) if names else spec.get("dims")
    if dims is not None and feature["shape"] != [dims]:
        issues.append(f"{key} 维度为 {feature['shape']}，机器人配置要求 [{dims}]")
    # 名称写在数据集里时逐一核对顺序：维度相同但顺序不同会静默训练出错误的策略。
    if names and feature.get("names") and list(feature["names"]) != list(names):
        issues.append(f"{key} 维度名称或顺序与机器人配置不一致")
    if stats is not None and key not in stats:
        issues.append(f"meta/stats.json 缺少 {key}")
    return issues


def camera_issues(info, stats, robot):
    issues = []
    for key in robot["cameras"]:
        feature = info["features"].get(key)
        if feature is None:
            issues.append(f"缺少相机 {key}")
        elif feature["dtype"] not in ("video", "image"):
            issues.append(f"{key} 不是图像特征（dtype={feature['dtype']}）")
        elif stats is not None and key not in stats:
            issues.append(f"meta/stats.json 缺少 {key}")
    return issues


def check_dataset(root, robot, policy):
    """返回检查报告；issues 非空时不能训练。"""
    root = Path(root)
    issues = []
    if not (root / "meta" / "info.json").exists():
        report = {"ok": False, "root": str(root), "issues": [f"不是 LeRobot 数据集：{root}/meta/info.json 不存在"]}
        return report
    info = read_info(root)
    stats = read_stats(root)
    if info.get("codebase_version") != DATASET_VERSION:
        issues.append(f"数据集版本为 {info.get('codebase_version')}，需要 {DATASET_VERSION}（在 manip-data-platform 中转换）")
    if info.get("fps") != robot["fps"]:
        issues.append(f"数据集 fps={info.get('fps')}，机器人配置 fps={robot['fps']}")
    if stats is None:
        issues.append("缺少 meta/stats.json，无法归一化")
    issues.extend(vector_issues(info, stats, robot, "state"))
    issues.extend(vector_issues(info, stats, robot, "action"))
    issues.extend(camera_issues(info, stats, robot))
    if info.get("total_episodes", 0) < 2:
        issues.append("至少需要 2 个 episode（训练集与验证集）")
    # π0.5 以任务描述作为语言输入；ACT / DP 不使用。
    if policy == "pi05" and info.get("total_tasks", 0) < 1:
        issues.append("π0.5 需要任务描述（meta/tasks.parquet）")

    unused = [key for key, ft in info["features"].items()
              if ft["dtype"] in ("video", "image") and key not in robot["cameras"]]
    report = {
        "ok": not issues,
        "root": str(root),
        "issues": issues,
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "tasks": info.get("total_tasks"),
        "cameras": list(robot["cameras"]),
        "unused_images": unused,
    }
    return report


def input_keys(robot):
    keys = [robot["state"]["feature"], *robot["cameras"]]
    return keys


def demo_check_dataset():
    from config.settings import Settings
    from util.config_helper import load_yaml

    settings = Settings()
    robot = load_yaml(settings.config_dir / "robot" / "tianyi2_robotiq.yaml")
    report = check_dataset(settings.root / "data" / "synthetic_tianyi2_robotiq", robot, "act")
    logger.info("ok={} episodes={} frames={} unused images={}", report["ok"], report.get("episodes"),
                report.get("frames"), report.get("unused_images"))
    for issue in report["issues"]:
        logger.warning(issue)
    missing = check_dataset(settings.root / "data" / "missing", robot, "act")
    logger.info("missing dataset: {}", missing["issues"])


def main():
    demo_check_dataset()


if __name__ == "__main__":
    main()
