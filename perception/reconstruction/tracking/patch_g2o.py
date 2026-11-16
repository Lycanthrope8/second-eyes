"""Build fix for g2o on Android: its package config always asks for desktop OpenGL.

g2o's cmake_modules/Config.cmake.in runs find_dependency(OpenGL) unconditionally, so every project that uses g2o
(stella_vslam, qtrack) fails to configure where there is no desktop OpenGL, as on Android. We build g2o without
OpenGL (G2O_USE_OPENGL=OFF: no opengl_helper library, nothing links OpenGL), so the line is dropped.
Idempotent, and it stops if the template is not the pinned one."""
import sys
from pathlib import Path

TEMPLATE = "cmake_modules/Config.cmake.in"
OLD = "find_dependency(Eigen3)\nfind_dependency(OpenGL)\n"
NEW = "find_dependency(Eigen3)\n"


def apply(root):
    p = Path(root) / TEMPLATE
    s = p.read_text(encoding="utf-8")
    if s.count(OLD) == 1:
        p.write_text(s.replace(OLD, NEW), encoding="utf-8")
        return "patched"
    if "find_dependency(OpenGL)" not in s and NEW in s:
        return "already patched"
    raise SystemExit(f"patch_g2o: {TEMPLATE} is not the pinned g2o template; stopping.")


if __name__ == "__main__":
    print(apply(sys.argv[1]))
