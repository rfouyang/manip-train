"""命令行入口：uv run manip-train <命令>。

  check  <experiment.yaml>                  检查数据集是否可训练
  train  <experiment.yaml> [--set k=v ...]  训练 → 离线评估 → 导出 → 注册到 ClearML
         [--queue h200] [--resume DIR] [--no-clearml]
  eval   <experiment.yaml> <pretrained_dir>  只做离线评估
  serve  <bundle_dir> [--host --port]       启动策略服务
  bench  <bundle_dir>                       本地推理延迟（不经网络）
"""

import argparse
import json
from pathlib import Path

import yaml

from loguru import logger

from app.context import AppContext
from util.config_helper import load_experiment, parse_overrides
from util.log_helper import setup_logger


def load(context, args):
    experiment = load_experiment(args.experiment, context.settings.config_dir, parse_overrides(args.set))
    return experiment


def command_check(context, args):
    report = context.runner.check(load(context, args))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_train(context, args):
    options = {"queue": args.queue, "resume": args.resume}
    if args.no_clearml:
        options["clearml"] = False
    result = context.runner.run(load(context, args), **options)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_eval(context, args):
    from component.tracking.clearml_tracker import ClearMLTracker
    from util.lerobot_helper import open_metadata, split_episodes

    experiment = load(context, args)
    meta = open_metadata(experiment["dataset"]["root"], experiment["dataset"]["repo_id"])
    _, episodes = split_episodes(meta, experiment["lerobot"].get("dataset.eval_split", 0.1))
    report_dir = Path(args.report_dir) if args.report_dir else Path(args.pretrained).parent / "report_eval"
    tracker = ClearMLTracker(experiment["clearml"].get("project", "manip-train"), f"{experiment['name']}-eval",
                             enabled=context.settings.clearml and not args.no_clearml, report_dir=report_dir)
    result = context.runner.evaluate(experiment, Path(args.pretrained), episodes, tracker)
    if result:
        (report_dir / "metrics.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    tracker.close()
    print(json.dumps(result["metrics"] if result else None, indent=2))
    logger.info("Report written to {}", report_dir)


def command_serve(context, args):
    from component.serving.policy_runtime import PolicyRuntime
    from component.serving.policy_server import PolicyServer

    runtime = PolicyRuntime(args.bundle, device=args.device, overrides=inference_overrides(args))
    PolicyServer(runtime, host=args.host, port=args.port).run()


def command_bench(context, args):
    from component.serving.policy_runtime import PolicyRuntime, benchmark

    result = benchmark(PolicyRuntime(args.bundle, device=args.device, overrides=inference_overrides(args)),
                       steps=args.steps)
    print(json.dumps(result, indent=2))


def inference_overrides(args):
    """--set n_action_steps=1 等推理参数；不给时用导出包 manifest 里的 inference_overrides。"""
    if not args.set:
        return None
    result = {key: yaml.safe_load(value) for key, value in parse_overrides(args.set).items()}
    return result


def build_parser():
    parser = argparse.ArgumentParser(prog="manip-train", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("check", command_check), ("train", command_train), ("eval", command_eval)):
        sub = commands.add_parser(name)
        sub.add_argument("experiment")
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help="覆盖实验参数，例如 lerobot.steps=1000 或 dataset.root=/data/x")
        sub.set_defaults(handler=handler)
        if name == "train":
            sub.add_argument("--queue", help="提交到 clearml-agent 队列远程训练（例如 h200）")
            sub.add_argument("--resume", help="续训：已有的训练输出目录")
        if name == "eval":
            sub.add_argument("pretrained", help="checkpoint 的 pretrained_model 目录或导出包")
            sub.add_argument("--report-dir", help="本地报告目录（默认为 pretrained 同级的 report_eval/）")
        if name in ("train", "eval"):
            sub.add_argument("--no-clearml", action="store_true")

    for name, handler in (("serve", command_serve), ("bench", command_bench)):
        sub = commands.add_parser(name)
        sub.add_argument("bundle")
        sub.add_argument("--device", default=None)
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help="推理参数，例如 n_action_steps=1（每帧重新推理）或 temporal_ensemble_coeff=0.01")
        sub.set_defaults(handler=handler)
        if name == "serve":
            sub.add_argument("--host", default="0.0.0.0")
            sub.add_argument("--port", type=int, default=8765)
        else:
            sub.add_argument("--steps", type=int, default=100)
    return parser


@logger.catch(reraise=True)
def main():
    args = build_parser().parse_args()
    context = AppContext()
    setup_logger(context.settings.log_dir, context.settings.log_level)
    args.handler(context, args)


if __name__ == "__main__":
    main()
