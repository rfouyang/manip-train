"""依赖方向 app → component → util；只有 util/lerobot_helper.py 直接 import lerobot（测试夹具除外）。"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def sources(package):
    result = sorted((ROOT / package).rglob("*.py"))
    return result


def test_util_does_not_import_upper_layers():
    for path in sources("util"):
        assert not imports(path) & {"app", "component"}, path


def test_component_does_not_import_app():
    for path in sources("component"):
        assert "app" not in imports(path), path


def test_only_lerobot_helper_imports_lerobot():
    for package in ("app", "component", "util", "config"):
        for path in sources(package):
            if path.name == "lerobot_helper.py":
                continue
            assert "lerobot" not in imports(path), path
