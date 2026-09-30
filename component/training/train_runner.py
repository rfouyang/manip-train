"""一次实验的完整流程：检查数据集 → ClearML 任务 → lerobot 训练 → 离线评估 → 导出 → 注册模型。"""

import json
import shutil
from pathlib import Path

from loguru import logger

from component.data.dataset_check import check_dataset, input_keys
from component.evaluation.offline_eval import evaluate
from component.export.model_bundle import export_bundle, model_tags, read_manifest
from component.tracking.clearml_tracker import ClearMLTracker, lerobot_logger_class
from util.config_helper import resolve_root
from util.lerobot_helper import (PRETRAINED_MODEL_DIR, build_train_config, last_checkpoint, open_metadata, run_train,
                                 split_episodes)


class TrainRunner:
    def __init__(self, settings):
        self.settings = settings

    def check(self, experiment):
        report = check_dataset(experiment["dataset"]["root"], experiment["robot"], experiment["policy"])
        return report

    def train_flags(self, experiment, output_dir):
        flags = dict(experiment["lerobot"])
        flags["dataset.root"] = experiment["dataset"]["root"]
        flags["dataset.repo_id"] = experiment["dataset"]["repo_id"]
        flags["output_dir"] = str(output_dir)
        flags["job_name"] = experiment["name"]
        return flags

    def run(self, experiment, **kwargs):
        """kwargs: queue（提交到 clearml-agent 队列远程运行）、resume（续训的训练输出目录）、clearml（覆盖开关）。"""
        report = self.check(experiment)
        if not report["ok"]:
            raise ValueError("数据集检查未通过：" + "；".join(report["issues"]))

        resume = kwargs.get("resume")
        output_dir = Path(resume) if resume else self.settings.output_dir / "train" / experiment["name"]
        if not resume and output_dir.exists():
            raise FileExistsError(f"训练输出已存在：{output_dir}（续训用 --resume，或改实验名）")

        clearml = kwargs.get("clearml", self.settings.clearml)
        clearml_cfg = experiment["clearml"]
        tracker = ClearMLTracker(clearml_cfg.get("project", "manip-train"), experiment["name"], enabled=clearml,
                                 tags=[*clearml_cfg.get("tags", []), f"policy:{experiment['policy']}",
                                       f"robot:{experiment['robot']['name']}"],
                                 report_dir=output_dir.with_name(output_dir.name + "_report"))
        experiment = tracker.connect_config(experiment)
        tracker.log_artifact("dataset_check", report)
        if kwargs.get("queue"):
            tracker.execute_remotely(kwargs["queue"])
        experiment["dataset"]["root"] = resolve_root(experiment["dataset"]["root_source"], self.settings.config_dir)

        flags = self.train_flags(experiment, output_dir)
        checkpoint = Path(resume) / "checkpoints" / "last" if resume else None
        cfg = build_train_config(flags, input_keys=input_keys(experiment["robot"]), resume_from=checkpoint)
        tracker.log_text("lerobot-train " + " ".join(f"--{k}={v}" for k, v in flags.items()))

        meta = open_metadata(experiment["dataset"]["root"], experiment["dataset"]["repo_id"])
        train_episodes, eval_episodes = split_episodes(meta, cfg.dataset.eval_split)
        split = {"train": train_episodes, "eval": eval_episodes}
        tracker.log_artifact("episode_split", split)
        logger.info("Training {} ({}): {} train / {} eval episodes -> {}", experiment["name"], experiment["policy"],
                    len(train_episodes), len(eval_episodes), output_dir)

        run_train(cfg, lerobot_logger_class(tracker), min_std=experiment["robot"].get("normalization", {}).get("min_std"))

        pretrained = last_checkpoint(output_dir) / PRETRAINED_MODEL_DIR
        evaluation = self.evaluate(experiment, pretrained, eval_episodes, tracker, step=cfg.steps)
        bundle = export_bundle(pretrained, self.settings.output_dir / "models", experiment, evaluation)
        manifest = read_manifest(bundle)
        model_id = tracker.log_model(bundle, experiment["name"], model_tags(manifest), manifest["evaluation"] or {})
        result = {
            "output_dir": str(output_dir),
            "bundle": str(bundle),
            "model_id": model_id,
            "clearml_url": tracker.url,
            "evaluation": evaluation["metrics"] if evaluation else None,
        }
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        tracker.close()
        return result

    def evaluate(self, experiment, pretrained, episodes, tracker, **kwargs):
        if not episodes:
            logger.warning("No held-out episodes (dataset.eval_split=0); offline evaluation skipped")
            return None
        options = experiment.get("evaluation", {})
        result = evaluate(pretrained, experiment["dataset"]["root"], experiment["dataset"]["repo_id"], episodes,
                          tracker, max_episodes=options.get("max_episodes", 5),
                          max_frames=options.get("max_frames", 600), step=kwargs.get("step", 0),
                          policy_overrides=options.get("policy_overrides"))
        return result


def demo_train_runner():
    from config.settings import Settings
    from util.config_helper import load_experiment, parse_overrides

    settings = Settings(output_dir="output/demo")
    shutil.rmtree(settings.output_dir / "train" / "demo-act", ignore_errors=True)
    experiment = load_experiment(settings.config_dir / "experiment/synthetic_act_smoke.yaml", settings.config_dir,
                                 overrides=parse_overrides(["name=demo-act", "lerobot.steps=40",
                                                            "lerobot.save_freq=40", "lerobot.eval_steps=20"]))
    runner = TrainRunner(settings)
    logger.info("Dataset check: {}", runner.check(experiment)["ok"])
    result = runner.run(experiment, clearml=False)
    logger.info("Result: {}", result)


def main():
    demo_train_runner()


if __name__ == "__main__":
    main()
