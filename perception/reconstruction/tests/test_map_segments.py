import os, shutil, subprocess
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CXX = shutil.which("g++") or shutil.which("clang++")


@pytest.mark.skipif(CXX is None, reason="needs a desktop C++ compiler (g++ or clang++)")
def test_map_restart_policy(tmp_path):
    src = os.path.join(HERE, "..", "tracking", "test_map_segments.cc")
    exe = str(tmp_path / "tms")
    subprocess.run([CXX, "-std=c++17", "-O1", "-o", exe, src], check=True)
    r = subprocess.run([exe], capture_output=True, text=True)
    assert r.returncode == 0 and "map_segments: ok" in r.stdout, r.stderr
