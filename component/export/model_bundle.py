"""导出包：lerobot pretrained_model 目录（权重、config、前后处理与归一化统计量）+ manifest.json。

部署端（4090 笔记本 / Jetson Thor）只需要这个目录，不需要训练输出和数据集。
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from loguru import logger

from util.lerobot_helper import LEROBOT_VERSION, read_info

MANIFEST = "manifest.json"


def git_commit(root):
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                                check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        result = None
    return result


def export_bundle(pretrained_dir, output_dir, experiment, evaluation=None):
    pretrained_dir = Path(pretrained_dir)
    info = read_info(experiment["dataset"]["root"])
    robot = experiment["robot"]
    created = datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle = Path(output_dir) / f"{experiment['name']}-{created}"
    temp = bundle.with_name(bundle.name + ".tmp")
    shutil.copytree(pretrained_dir, temp)
    manifest = {
        "name": experiment["name"],
        "policy": experiment["policy"],
        "robot": robot["name"],
        "fps": info["fps"],
        "state_feature": robot["state"]["feature"],
        "state_dims": info["features"][robot["state"]["feature"]]["shape"][0],
        "action_names": info["features"]["action"].get("names"),
        "cameras": {key: info["features"][key]["shape"] for key in robot["cameras"]},
        "dataset": {
            "root": experiment["dataset"]["root"],
            "repo_id": experiment["dataset"]["repo_id"],
            "version": experiment["dataset"].get("version"),
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
        },
        "evaluation": (evaluation or {}).get("metrics"),
        # 部署时 PolicyRuntime 默认使用与离线评估相同的推理设置。
        "inference_overrides": experiment.get("evaluation", {}).get("policy_overrides"),
        "lerobot_version": LEROBOT_VERSION,
        "git_commit": git_commit(Path(__file__).parent),
        "source_checkpoint": str(pretrained_dir),
        "created": created,
    }
    (temp / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # 写完再改名：半成品不会被当作可部署的导出包。
    temp.rename(bundle)
    logger.info("Exported bundle {}", bundle)
    return bundle


def read_manifest(bundle):
    with open(Path(bundle) / MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


def model_tags(manifest):
    tags = [f"policy:{manifest['policy']}", f"robot:{manifest['robot']}", f"dataset:{manifest['dataset']['repo_id']}"]
    if manifest["dataset"].get("version"):
        tags.append(f"dataset_version:{manifest['dataset']['version']}")
    return tags


def main():
    root = Path(__file__).resolve().parents[2]
    bundles = sorted((root / "output" / "models").glob("*/manifest.json"))
    for path in bundles:
        manifest = read_manifest(path.parent)
        logger.info("{}: {} tags={}", path.parent.name, manifest["evaluation"], model_tags(manifest))
    if not bundles:
        logger.info("No bundles yet; run: uv run python -m app.cli train config/experiment/synthetic_act_smoke.yaml")


if __name__ == "__main__":
    main()
