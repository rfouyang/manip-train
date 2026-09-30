import pytest

from tests.conftest import ROOT
from util.config_helper import expand_env, load_experiment, parse_overrides


def test_overrides_and_merge_order():
    experiment = load_experiment(ROOT / "config/experiment/synthetic_act_smoke.yaml", ROOT / "config",
                                 parse_overrides(["lerobot.steps=7", "evaluation.max_frames=5", "lerobot.policy.chunk_size=20"]))
    flags = experiment["lerobot"]
    assert flags["policy.type"] == "act"          # policy yaml
    assert flags["batch_size"] == 8               # experiment 覆盖 policy yaml
    assert flags["steps"] == 7                    # 命令行覆盖 experiment
    assert flags["policy.chunk_size"] == 20
    assert experiment["evaluation"]["max_frames"] == 5
    assert experiment["dataset"]["root"].startswith(str(ROOT))


def test_undefined_env_is_an_error(monkeypatch):
    monkeypatch.delenv("MANIP_TRAIN_UNDEFINED", raising=False)
    with pytest.raises(KeyError):
        expand_env("${MANIP_TRAIN_UNDEFINED}/x")


def test_bad_override():
    with pytest.raises(ValueError):
        parse_overrides(["lerobot.steps"])
