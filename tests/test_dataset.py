import copy
import json
import math

import numpy as np

from component.data.dataset_check import check_dataset
from tests.conftest import ROOT
from util.config_helper import load_yaml
from util.lerobot_helper import floor_std, open_metadata, split_episodes

ROBOT = load_yaml(ROOT / "config/robot/tianyi2_inspire.yaml")


def test_synthetic_dataset_passes(synthetic_dataset):
    report = check_dataset(synthetic_dataset, ROBOT, "pi05")
    assert report["ok"], report["issues"]
    assert report["unused_images"] == ["observation.images.camera_top_depth"]


def test_wrong_names_and_version_are_reported(synthetic_dataset, tmp_path):
    robot = copy.deepcopy(ROBOT)
    robot["state"]["names"] = list(reversed(robot["state"]["names"]))
    robot["fps"] = 15
    report = check_dataset(synthetic_dataset, robot, "act")
    assert not report["ok"]
    assert any("顺序" in issue for issue in report["issues"])
    assert any("fps" in issue for issue in report["issues"])

    fake = tmp_path / "v21" / "meta"
    fake.mkdir(parents=True)
    info = json.loads((synthetic_dataset / "meta/info.json").read_text())
    info["codebase_version"] = "v2.1"
    (fake / "info.json").write_text(json.dumps(info))
    report = check_dataset(fake.parent, ROBOT, "act")
    assert any("v2.1" in issue for issue in report["issues"])


def test_split_matches_lerobot_rule(synthetic_dataset):
    meta = open_metadata(synthetic_dataset, "local/synthetic")
    train, evaluation = split_episodes(meta, 0.2)
    n_eval = math.ceil(meta.total_episodes * 0.2)
    assert evaluation == list(range(meta.total_episodes - n_eval, meta.total_episodes))
    assert sorted(train + evaluation) == list(range(meta.total_episodes))


def test_floor_std_only_raises_small_dims():
    stats = {"action": {"std": np.array([1e-4, 0.5, 0.0], dtype=np.float32), "min": np.zeros(3)}}
    adjusted = floor_std(stats, ["action"], 0.01)
    assert adjusted == [("action", [0, 2])]
    np.testing.assert_allclose(stats["action"]["std"], [0.01, 0.5, 0.01])
