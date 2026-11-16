"""patch_stella against the pinned stella_vslam 0.7.0 sources: applies once, is idempotent, and refuses other sources."""
import importlib.util
import os
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("patch_stella", os.path.join(REPO, "tracking", "patch_stella.py"))
patch_stella = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_stella)


def _fake_tree(tmp_path):
    for rel, edits in patch_stella.EDITS.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("// head\n" + "\n// --\n".join(old for old, _ in edits) + "\n// tail\n", encoding="utf-8")
    return tmp_path


def test_applies_then_is_idempotent(tmp_path):
    root = _fake_tree(tmp_path)
    assert patch_stella.apply(root) == "patched"
    assert patch_stella.apply(root) == "already patched"
    top = (root / "CMakeLists.txt").read_text()
    assert "find_package(OpenCV QUIET" in top and "3.3.1" not in top


def test_reapplies_after_sources_are_restored(tmp_path):
    root = _fake_tree(tmp_path)
    patch_stella.apply(root)
    _fake_tree(tmp_path)                       # what git checkout --force does to tracked files
    assert patch_stella.apply(root) == "patched"


def test_refuses_unexpected_sources(tmp_path):
    root = _fake_tree(tmp_path)
    (root / "CMakeLists.txt").write_text("find_package(OpenCV 9 QUIET)\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        patch_stella.apply(root)

