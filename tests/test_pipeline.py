"""端到端：ACT 训练几步 → 续训 → 离线评估 → 导出 → 策略服务往返。约 1 分钟（GPU）。"""

import asyncio
import threading

import numpy as np

from component.serving.policy_client import PolicyClient
from component.serving.policy_runtime import PolicyRuntime, dummy_observation
from component.serving.policy_server import PolicyServer
from component.training.train_runner import TrainRunner
from config.settings import Settings
from tests.conftest import ROOT
from util.config_helper import load_experiment, parse_overrides


def experiment_for(dataset, steps):
    overrides = parse_overrides([f"dataset.root={dataset}", "name=test-act", f"lerobot.steps={steps}",
                                 "lerobot.save_freq=10", "lerobot.eval_steps=10", "lerobot.log_freq=5",
                                 "lerobot.num_workers=0", "lerobot.dataset.eval_split=0.25",
                                 "evaluation.max_frames=20"])
    experiment = load_experiment(ROOT / "config/experiment/synthetic_act_smoke.yaml", ROOT / "config", overrides)
    return experiment


def test_train_resume_export_serve(synthetic_dataset, tmp_path):
    runner = TrainRunner(Settings(output_dir=str(tmp_path / "output")))
    result = runner.run(experiment_for(synthetic_dataset, 10), clearml=False)
    assert result["evaluation"]["frames"] == 20

    resumed = runner.run(experiment_for(synthetic_dataset, 20), clearml=False, resume=result["output_dir"])
    assert (tmp_path / "output/train/test-act/checkpoints/000020").exists()

    runtime = PolicyRuntime(resumed["bundle"])
    assert runtime.manifest["state_dims"] == 16
    server = PolicyServer(runtime, host="127.0.0.1", port=18765)
    ready = threading.Event()
    threading.Thread(target=lambda: asyncio.run(server.serve_forever(ready)), daemon=True).start()
    assert ready.wait(10)

    client = PolicyClient("ws://127.0.0.1:18765")
    client.reset()
    reply = client.infer(dummy_observation(client.manifest))
    assert reply["action"].shape == (16,) and np.isfinite(reply["action"]).all()
    bad = dummy_observation(client.manifest)
    bad["observation.images.camera_top"] = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        client.infer(bad)
        raise AssertionError("wrong image size should be rejected")
    except RuntimeError as error:
        assert "尺寸" in str(error)
    client.close()
