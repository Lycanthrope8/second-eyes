#!/usr/bin/env python3
"""Build libse_llama and its test program with llama.cpp at the pinned release (A1.8b, D55).

    python tools/build_llama.py fetch
    python tools/build_llama.py android [--ndk PATH] [--arm-arch armv8.2-a+dotprod+fp16]
    python tools/build_llama.py host

fetch clones llama.cpp once into third_party/llama.cpp at the release in native/llama.cpp.pin and checks its commit.
android cross-compiles native/ for the headset (arm64-v8a) with an Android NDK: --ndk, else ANDROID_NDK_HOME,
ANDROID_NDK_ROOT or ANDROID_NDK, else the newest NDK inside a Unity install (Unity's Android Build Support includes
one). llama.cpp's CPU code is built for --arm-arch (dot-product and half-precision instructions, which the Quest 3's
Arm cores have); the rest keeps Android's portable baseline, as llama.cpp's docs/android.md advises. host builds for
this PC, for tests, with llama.cpp's native flags: it runs only on a processor like the one that built it. Both need CMake and Ninja (pip install cmake ninja). The files go to native/out/<target>/, with
build.json recording the release, the NDK, the flags and each file's SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "native"
LLAMA = ROOT / "third_party" / "llama.cpp"
URL = "https://github.com/ggml-org/llama.cpp"
TARGETS = {"android": ["se_llama", "se_llama_cli", "llama-bench"], "host": ["se_llama", "se_llama_cli"]}
OUTPUTS = {"android": ["libse_llama.so", "se_llama_cli", "llama-bench"],
           "host": ["libse_llama.so", "se_llama_cli"] if os.name != "nt" else ["se_llama.dll", "se_llama_cli.exe"]}


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


def pinned() -> tuple:
    tag, commit = (NATIVE / "llama.cpp.pin").read_text(encoding="utf-8").split()[:2]
    return tag, commit


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        raise ToolError(f"{Path(str(cmd[0])).name} stopped with code {r.returncode}; see its output above.")
    return r


def fetch() -> Path:
    tag, commit = pinned()
    if not (LLAMA / "include" / "llama.h").is_file():
        if shutil.which("git") is None:
            raise ToolError("git was not found on your PATH.")
        LLAMA.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching llama.cpp {tag} into third_party/llama.cpp ...")
        run(["git", "clone", "--depth", "1", "--branch", tag, URL, LLAMA])
    head = subprocess.run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip() != commit:
        raise ToolError(f"third_party/llama.cpp is at {head.stdout.strip()[:8] or '?'}, but native/llama.cpp.pin says "
                        f"{tag} ({commit[:8]}). Delete the folder and fetch again.")
    print(f"llama.cpp {tag} ({commit[:8]}) is in third_party/llama.cpp.")
    return LLAMA


def version_key(name: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", name)) or (0,)


def find_ndk(given) -> Path:
    candidates = [Path(given)] if given else []
    candidates += [Path(os.environ[v]) for v in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "ANDROID_NDK") if os.environ.get(v)]
    hubs = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Unity" / "Hub" / "Editor",
            Path("/Applications/Unity/Hub/Editor"), Path.home() / "Unity" / "Hub" / "Editor"]
    for hub in hubs:
        if hub.is_dir():
            for editor in sorted(hub.iterdir(), key=lambda p: version_key(p.name), reverse=True):
                for sub in ("Editor/Data/PlaybackEngines/AndroidPlayer/NDK", "PlaybackEngines/AndroidPlayer/NDK"):
                    candidates.append(editor / sub)
    for ndk in candidates:
        if (ndk / "build" / "cmake" / "android.toolchain.cmake").is_file():
            return ndk
    raise ToolError("No Android NDK found. Install Unity's Android Build Support, or pass --ndk <the NDK folder>.")


def ndk_revision(ndk: Path) -> str:
    props = ndk / "source.properties"
    if props.is_file():
        m = re.search(r"Pkg\.Revision\s*=\s*(\S+)", props.read_text(encoding="utf-8", errors="replace"))
        if m:
            return m.group(1)
    return "unknown"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(target: str, args) -> int:
    for tool in ("cmake", "ninja"):
        if shutil.which(tool) is None:
            raise ToolError(f"{tool} was not found. Install both with: pip install cmake ninja")
    llama = fetch()
    build_dir = ROOT / "build" / ("android-arm64" if target == "android" else "host")
    out_dir = NATIVE / "out" / ("android-arm64" if target == "android" else "host")
    config = ["cmake", "-S", NATIVE, "-B", build_dir, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={shutil.which('ninja')}",
              "-DCMAKE_BUILD_TYPE=Release", f"-DLLAMA_DIR={llama}", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"]
    record = {"target": target}
    if target == "android":
        ndk = find_ndk(args.ndk)
        record["ndk"] = {"path": str(ndk), "revision": ndk_revision(ndk)}
        record["arm_arch"] = args.arm_arch
        print(f"NDK {record['ndk']['revision']} at {ndk}")
        config += [f"-DCMAKE_TOOLCHAIN_FILE={ndk / 'build' / 'cmake' / 'android.toolchain.cmake'}",
                   "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-28", "-DGGML_NATIVE=OFF",
                   f"-DGGML_CPU_ARM_ARCH={args.arm_arch}", "-DSE_BUILD_BENCH=ON"]
    run(config)
    run(["cmake", "--build", build_dir, "--target", *TARGETS[target]])
    out_dir.mkdir(parents=True, exist_ok=True)
    record["files"] = {}
    for name in OUTPUTS[target]:
        found = next((p for p in build_dir.rglob(name) if p.is_file()), None)
        if found is None:
            raise ToolError(f"The build finished, but {name} isn't in {build_dir.relative_to(ROOT).as_posix()}/.")
        shutil.copy2(found, out_dir / name)
        record["files"][name] = {"bytes": found.stat().st_size, "sha256": sha256_of(found)}
    tag, commit = pinned()
    record["llama_cpp"] = f"{tag} ({commit[:8]})"
    record["cmake"] = subprocess.run(["cmake", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    record["host"] = f"{platform.system()} {platform.machine()}"
    (out_dir / "build.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Built into {out_dir.relative_to(ROOT).as_posix()}/: " +
          ", ".join(f"{n} ({f['bytes'] / 1e6:.1f} MB)" for n, f in record["files"].items()))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/build_llama.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="clone llama.cpp at the pinned release into third_party/")
    a = sub.add_parser("android", help="cross-compile for the headset")
    a.add_argument("--ndk", help="the Android NDK folder (default: found automatically)")
    a.add_argument("--arm-arch", default="armv8.2-a+dotprod+fp16", help="instruction set for llama.cpp's CPU code")
    sub.add_parser("host", help="build for this PC, for tests")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            fetch()
            return 0
        return build(args.command, args)
    except ToolError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
