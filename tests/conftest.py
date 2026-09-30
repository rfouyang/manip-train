from pathlib import Path

import pytest

from tests.fixtures.make_synthetic_v3 import make_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    root = make_dataset(tmp_path_factory.mktemp("data") / "synthetic", episodes=4, length=30)
    return root
