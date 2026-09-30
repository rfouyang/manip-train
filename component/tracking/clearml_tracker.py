"""训练报告写入 ClearML。enabled=False 时只写本地日志，接口不变（没有凭证、测试、离线时使用）。"""

import json
from pathlib import Path

from loguru import logger


class ClearMLTracker:
    def __init__(self, project, task_name, enabled=True, **kwargs):
        """kwargs：tags；report_dir（图、表、标量同时写到本地目录，不连 ClearML 时也能查看报告）。"""
        self.enabled = enabled
        self.task = None
        self.project = project
        self.task_name = task_name
        self.report_dir = Path(kwargs["report_dir"]) if kwargs.get("report_dir") else None
        if self.report_dir:
            self.report_dir.mkdir(parents=True, exist_ok=True)
        if not enabled:
            logger.info("ClearML disabled; reports go to the local log only")
            return
        from clearml import Task

        # 不让 ClearML 自动接管 torch.save：checkpoint 由 lerobot 写本地，最终模型在 log_model 中显式上传。
        self.task = Task.init(
            project_name=project,
            task_name=task_name,
            tags=kwargs.get("tags") or [],
            reuse_last_task_id=False,
            auto_connect_frameworks={"pytorch": False, "tensorboard": False, "matplotlib": False},
            auto_connect_arg_parser=False,
        )
        logger.info("ClearML task: {}", self.task.get_output_log_web_page())

    @property
    def url(self):
        result = self.task.get_output_log_web_page() if self.task else None
        return result

    def connect_config(self, config, name="experiment"):
        """返回值是 ClearML 中可能被修改过的配置：clearml-agent 克隆任务并改参数后远程运行时用到。"""
        if not self.task:
            return config
        result = self.task.connect(config, name=name)
        return result

    def execute_remotely(self, queue):
        # 本地进程在这里退出，任务进入队列，由 H200 上的 clearml-agent 重新执行同一入口。
        self.task.execute_remotely(queue_name=queue, exit_process=True)

    def save_local(self, name):
        # 文件名里不能有 "/"：offline_eval/actions → offline_eval_actions
        path = self.report_dir / name.replace("/", "_")
        return path

    def log_scalars(self, step, group, values):
        if self.report_dir:
            with open(self.save_local("scalars.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "group": group, **values}) + "\n")
        if not self.task:
            logger.info("[{}] step {}: {}", group, step, {k: round(v, 5) for k, v in values.items()})
            return
        report = self.task.get_logger()
        for key, value in values.items():
            report.report_scalar(title=group, series=key, value=value, iteration=step)

    def log_figure(self, title, series, figure, step=0):
        if self.report_dir:
            figure.savefig(self.save_local(f"{title}_{series}.png"), dpi=110)
        if not self.task:
            return
        self.task.get_logger().report_matplotlib_figure(title=title, series=series, figure=figure, iteration=step,
                                                        report_image=True)

    def log_table(self, title, series, table, step=0):
        """table 为 pandas.DataFrame。"""
        if self.report_dir:
            table.to_csv(self.save_local(f"{title}_{series}.csv"), index=False)
        if not self.task:
            logger.info("[{}] {}\n{}", title, series, table.to_string())
            return
        self.task.get_logger().report_table(title=title, series=series, iteration=step, table_plot=table)

    def log_text(self, text):
        if not self.task:
            logger.info(text)
            return
        self.task.get_logger().report_text(text)

    def log_artifact(self, name, value):
        """value 为 dict、路径或 DataFrame。"""
        if self.report_dir and isinstance(value, dict):
            self.save_local(f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        if not self.task:
            return
        self.task.upload_artifact(name=name, artifact_object=value)

    def log_video(self, title, path, step=0):
        if not self.task:
            return
        self.task.get_logger().report_media(title=title, series=Path(path).name, local_path=str(path), iteration=step)

    def log_model(self, bundle_dir, name, tags, metadata):
        """上传导出包为 ClearML OutputModel；返回模型 id（部署时按 id 或标签拉取）。"""
        if not self.task:
            logger.info("Model bundle kept locally: {}", bundle_dir)
            return None
        from clearml import OutputModel

        model = OutputModel(task=self.task, name=name, framework="PyTorch", tags=tags)
        for key, value in metadata.items():
            model.set_metadata(key, str(value))
        model.update_weights_package(weights_path=str(bundle_dir), auto_delete_file=False)
        return model.id

    def close(self):
        if self.task:
            self.task.close()


def lerobot_logger_class(tracker):
    """生成替换 lerobot WandBLogger 的类：lerobot 训练循环按 wandb 的接口回调，这里转给 tracker。"""

    class LeRobotClearMLLogger:
        def __init__(self, cfg):
            self.cfg = cfg

        def log_dict(self, d, step=None, mode="train", custom_step_key=None):
            values = {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and k != custom_step_key}
            tracker.log_scalars(step if step is not None else int(d.get(custom_step_key, 0)), mode, values)

        def log_policy(self, checkpoint_dir):
            # 中间 checkpoint 只留本地（体积大、上传慢）；训练结束后导出并上传最终模型。
            tracker.log_text(f"checkpoint saved: {checkpoint_dir}")

        def log_video(self, video_path, step, mode="train"):
            tracker.log_video(f"{mode}/video", video_path, step)

    return LeRobotClearMLLogger


def demo_offline_tracker():
    import pandas as pd

    tracker = ClearMLTracker("manip-train/demo", "offline-demo", enabled=False)
    logger_class = lerobot_logger_class(tracker)
    lerobot_logger = logger_class(cfg=None)
    lerobot_logger.log_dict({"loss": 0.52, "lr": 1e-5, "note": "ignored"}, step=10)
    tracker.log_table("eval", "per_joint", pd.DataFrame({"joint": ["a", "b"], "mae": [0.1, 0.2]}))
    tracker.close()


def main():
    demo_offline_tracker()


if __name__ == "__main__":
    main()
