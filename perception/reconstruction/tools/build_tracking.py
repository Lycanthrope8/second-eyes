#!/usr/bin/env python3
"""Cross-build the Quest reconstruction runtime for the headset (arm64-v8a): the tracker (stella_vslam + qtrack) and
the reconstruction program (questpipe/qrecon_quest, with ONNX Runtime for the depth model).

    python tools/build_tracking.py [--ndk PATH] [--jobs N]

Same conventions as the Second Eyes repo's tools/build_llama.py: an NDK from --ndk, ANDROID_NDK_HOME / ANDROID_NDK_ROOT /
ANDROID_NDK, or Unity's Android Build Support; CMake and Ninja from pip (pip install cmake ninja); stripped outputs; a
build.json with every pin, the NDK revision and each file's SHA-256. Sources are fetched at the commits in
tracking/pins.json. Downloads without a pinned SHA-256 are recorded on first use (pin them from build.json afterwards).

Output: tracking/out/android-arm64/{bin/qtrack, bin/qrecon_quest, lib/*.so, share/orb_vocab.fbow, licenses/, build.json}"""
import argparse, hashlib, json, os, platform, re, shutil, subprocess, sys, tarfile, urllib.request, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent            # quest_recon/
TRK = HERE / "tracking"
TP = HERE / "third_party" / "tracking"
BUILD = HERE / "build" / "tracking-android-arm64"
PREFIX = BUILD / "prefix"
OUT = TRK / "out" / "android-arm64"
PINS = json.loads((TRK / "pins.json").read_text())
sys.path.insert(0, str(TRK))
import patch_stella  # noqa: E402
import patch_g2o  # noqa: E402


class ToolError(Exception):
    pass


def run(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        raise ToolError(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd[:3])} ...")
    return r


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_ndk(given):
    cands = [Path(given)] if given else []
    cands += [Path(os.environ[v]) for v in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "ANDROID_NDK") if os.environ.get(v)]
    roots = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Unity" / "Hub" / "Editor",
             Path.home() / "Unity" / "Hub" / "Editor", Path("/Applications/Unity/Hub/Editor")]
    for r in roots:
        if r.is_dir():
            for ed in sorted(r.iterdir(), reverse=True):
                for sub in ("Editor/Data/PlaybackEngines/AndroidPlayer/NDK", "PlaybackEngines/AndroidPlayer/NDK"):
                    cands.append(ed / sub)
    for ndk in cands:
        if (ndk / "build" / "cmake" / "android.toolchain.cmake").is_file():
            return ndk
    raise ToolError("No Android NDK found. Install Unity's Android Build Support, or pass --ndk <the NDK folder>.")


# Libraries every Android device provides (the NDK's stable system libraries)
ANDROID_SYSTEM_LIBS = {"libc.so", "libm.so", "libdl.so", "liblog.so", "libz.so", "libandroid.so", "libEGL.so",
                       "libGLESv1_CM.so", "libGLESv2.so", "libGLESv3.so", "libjnigraphics.so", "libOpenSLES.so",
                       "libOpenMAXAL.so", "libmediandk.so", "libcamera2ndk.so", "libvulkan.so", "libnativewindow.so",
                       "libneuralnetworks.so", "libsync.so", "libbinder_ndk.so", "libaaudio.so", "libamidi.so",
                       "libstdc++.so"}


def needed_of(readelf_output):
    """Shared libraries listed as NEEDED in `llvm-readelf -d` output."""
    return [line.split("[", 1)[1].split("]", 1)[0] for line in readelf_output.splitlines()
            if "(NEEDED)" in line and "[" in line]


def missing_libraries(readelf, out):
    """NEEDED libraries of the staged programs and libraries that are neither staged nor part of Android."""
    staged = {f.name for f in (out / "lib").iterdir()}
    missing = {}
    for f in list((out / "bin").iterdir()) + list((out / "lib").iterdir()):
        r = subprocess.run([str(readelf), "-d", str(f)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  (could not read the dependencies of {f.name}; skipped)")
            continue
        for lib in needed_of(r.stdout):
            if lib not in staged and lib not in ANDROID_SYSTEM_LIBS:
                missing.setdefault(lib, []).append(f.name)
    return missing


def ndk_revision(ndk):
    p = ndk / "source.properties"
    m = re.search(r"Pkg\.Revision\s*=\s*(\S+)", p.read_text()) if p.is_file() else None
    return m.group(1) if m else "unknown"


def find_tool(ndk, name):
    found = sorted((ndk / "toolchains" / "llvm" / "prebuilt").glob(f"*/bin/{name}*"))
    if not found:
        raise ToolError(f"{name} isn't in {ndk}/toolchains/llvm/prebuilt/*/bin/.")
    return found[0]


def git_fetch(name):
    pin, dst = PINS[name], TP / name
    if not (dst / ".git").is_dir():
        run(["git", "clone", "--recurse-submodules", pin["git"], dst])
    head = subprocess.check_output(["git", "-C", str(dst), "rev-parse", "HEAD"]).decode().strip()
    if head == pin["commit"]:
        # Already at the pinned commit: no forced checkout, which would restore every patched file and make the build
        # recompile the whole library. The patches below are idempotent and leave already-patched files untouched.
        print(f"  {name}: at the pinned commit {head[:8]}")
        return dst
    run(["git", "-C", dst, "fetch", "--tags", "origin"])
    run(["git", "-C", dst, "checkout", "--force", pin["commit"]])
    run(["git", "-C", dst, "submodule", "update", "--init", "--recursive"])
    head = subprocess.check_output(["git", "-C", str(dst), "rev-parse", "HEAD"]).decode().strip()
    if head != pin["commit"]:
        raise ToolError(f"{name}: HEAD is {head}, pinned {pin['commit']}")
    return dst


def json_licence(st):
    """stella bundles nlohmann/json's headers without its licence file: fetch LICENSE.MIT for the bundled version."""
    hdr = st / "3rd" / "json" / "include" / "nlohmann" / "json.hpp"
    dest = BUILD / "licences" / "nlohmann_json_LICENSE.MIT"
    if dest.is_file():
        return dest
    try:
        text = hdr.read_text(encoding="utf-8", errors="replace")
        v = [re.search(rf"#define NLOHMANN_JSON_VERSION_{k} (\d+)", text).group(1) for k in ("MAJOR", "MINOR", "PATCH")]
        url = f"https://raw.githubusercontent.com/nlohmann/json/v{'.'.join(v)}/LICENSE.MIT"
        try:
            import ssl, certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = None
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, context=ctx, timeout=30) as r:
            dest.write_bytes(r.read())
        print(f"  nlohmann_json {'.'.join(v)}: licence fetched from {url}")
    except Exception as e:                          # offline, or the header has no version: report it, don't fail the build
        print(f"  note: could not fetch nlohmann_json's licence ({e})")
    return dest


def download(name, record):
    pin = PINS[name]
    TP.mkdir(parents=True, exist_ok=True)
    arc = TP / Path(pin["url"]).name
    if not arc.is_file():
        print(f"  downloading {pin['url']}", flush=True)
        try:
            ctx = None
            try:   # certifi's roots: Python on Windows can lack the one some sites use
                import ssl, certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                pass
            with urllib.request.urlopen(pin["url"], context=ctx) as r, open(arc, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception as e:
            if arc.exists():
                arc.unlink()
            raise ToolError(f"{name}: download failed ({e}). Open the URL in a browser; if that version is gone, "
                            "tell me which versions the site lists rather than editing tracking/pins.json.")
    digest = sha256_of(arc)
    if pin.get("sha256") and pin["sha256"] != digest:
        raise ToolError(f"{name}: SHA-256 {digest} does not match the pin {pin['sha256']}")
    record["downloads"][name] = {"url": pin["url"], "sha256": digest, "pinned": bool(pin.get("sha256"))}
    dst = TP / name
    if not dst.is_dir():
        if arc.suffix in (".zip", ".aar"):
            zipfile.ZipFile(arc).extractall(dst)
        else:
            with tarfile.open(arc) as t:
                t.extractall(dst, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(dst)
    inner = [p for p in dst.iterdir() if p.is_dir()]
    return inner[0] if len(inner) == 1 else dst


def cmake_build(src, bdir, extra, base, target=None):
    run(["cmake", "-S", src, "-B", bdir, *base, *extra])
    run(["cmake", "--build", bdir, "--parallel", str(base_jobs[0])] + (["--target", target] if target else []))
    run(["cmake", "--install", bdir])


base_jobs = [4]


def build(args):
    for tool in ("cmake", "ninja", "git"):
        if not shutil.which(tool):
            raise ToolError(f"{tool} was not found. Install CMake and Ninja with: pip install cmake ninja (and Git for Windows)")
    base_jobs[0] = args.jobs
    ndk = find_ndk(args.ndk)
    if os.name == "nt" and " " in str(ndk):
        # With a space in the path, Ninja calls clang++ by its 8.3 short name (CLANG_~1.EXE); clang picks C or
        # C++ mode from its own name, so every C++ link would miss the C++ library.
        raise ToolError(f"The NDK path has a space in it ({ndk}). Make a link without one and pass it:\n"
                        f'  New-Item -ItemType Junction -Path "$HOME\\android-ndk-unity" -Target "{ndk}"\n'
                        f'  python tools\\build_tracking.py --ndk "$HOME\\android-ndk-unity"')
    print(f"NDK {ndk_revision(ndk)} at {ndk}")
    record = {"pins": PINS, "ndk": {"path": str(ndk), "revision": ndk_revision(ndk)}, "downloads": {},
              "host": f"{platform.system()} {platform.machine()}", "abi": "arm64-v8a", "platform": "android-28", "stl": "c++_shared"}
    PREFIX.mkdir(parents=True, exist_ok=True)
    base = ["-G", "Ninja", "-Wno-dev", "-Wno-deprecated", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
            "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",       # plain libX.so names, as Android loads them
            f"-DCMAKE_MAKE_PROGRAM={shutil.which('ninja')}",
            f"-DCMAKE_TOOLCHAIN_FILE={ndk / 'build' / 'cmake' / 'android.toolchain.cmake'}",
            "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-28", "-DANDROID_STL=c++_shared",
            "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={PREFIX}",
            f"-DCMAKE_PREFIX_PATH={PREFIX}", f"-DCMAKE_FIND_ROOT_PATH={PREFIX}"]

    print("== Eigen 3.4.0 (headers)")
    eigen = download("eigen", record)
    cmake_build(eigen, BUILD / "eigen", ["-DBUILD_TESTING=OFF", "-DEIGEN_BUILD_DOC=OFF", "-DEIGEN_BUILD_PKGCONFIG=OFF"], base)

    print("== SQLite (static)")
    sq = download("sqlite", record)
    (BUILD / "sqlite_src").mkdir(parents=True, exist_ok=True)
    (BUILD / "sqlite_src" / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(sqlite3 C)\n"
        f"add_library(sqlite3 STATIC \"{(sq / 'sqlite3.c').as_posix()}\")\n"
        "target_compile_definitions(sqlite3 PRIVATE SQLITE_THREADSAFE=1 SQLITE_OMIT_LOAD_EXTENSION)\n"
        "install(TARGETS sqlite3 ARCHIVE DESTINATION lib)\n"
        f"install(FILES \"{(sq / 'sqlite3.h').as_posix()}\" DESTINATION include)\n")
    cmake_build(BUILD / "sqlite_src", BUILD / "sqlite", [], base)

    print("== yaml-cpp 0.7.0 (static)")
    yc = git_fetch("yaml_cpp")
    cmake_build(yc, BUILD / "yaml_cpp", ["-DYAML_CPP_BUILD_TESTS=OFF", "-DYAML_CPP_BUILD_TOOLS=OFF", "-DYAML_CPP_BUILD_CONTRIB=OFF",
                                         "-DYAML_BUILD_SHARED_LIBS=OFF", "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"], base)

    print("== OpenCV Android SDK 4.10.0 (prebuilt)")
    ocv = download("opencv_android_sdk", record)
    ocv_dir = next((p.parent for p in ocv.rglob("OpenCVConfig.cmake") if "jni" in p.parts), None)
    if ocv_dir is None:
        raise ToolError("OpenCVConfig.cmake not found in the OpenCV Android SDK")
    common = [f"-DOpenCV_DIR={ocv_dir}", f"-DEigen3_DIR={PREFIX / 'share' / 'eigen3' / 'cmake'}"]

    print("== g2o (BSD parts only: no CSparse, no CHOLMOD, no viewer)")
    g2o = git_fetch("g2o")
    print("  " + patch_g2o.apply(g2o))
    cmake_build(g2o, BUILD / "g2o", common + ["-DBUILD_SHARED_LIBS=ON", "-DBUILD_UNITTESTS=OFF", "-DG2O_USE_CHOLMOD=OFF",
                                             "-DG2O_USE_CSPARSE=OFF", "-DG2O_USE_LGPL_LIBS=OFF", "-DG2O_USE_OPENGL=OFF",
                                             "-DG2O_USE_OPENMP=OFF", "-DG2O_BUILD_APPS=OFF", "-DG2O_BUILD_EXAMPLES=OFF",
                                             "-DG2O_BUILD_LINKED_APPS=OFF", "-DG2O_BUILD_SLAM2D_TYPES=OFF",
                                             "-DG2O_BUILD_SLAM2D_ADDON_TYPES=OFF", "-DG2O_BUILD_DATA_TYPES=OFF",
                                             "-DG2O_BUILD_SCLAM2D_TYPES=OFF", "-DG2O_BUILD_ICP_TYPES=OFF",
                                             "-DG2O_BUILD_SLAM3D_ADDON_TYPES=OFF"], base)

    print("== stella_vslam 0.7.0, patched (D132)")
    st = git_fetch("stella_vslam")
    print("  " + patch_stella.apply(st))
    cmake_build(st, BUILD / "stella", common + [f"-Dg2o_DIR={PREFIX / 'lib' / 'cmake' / 'g2o'}",
                                                f"-Dyaml-cpp_DIR={PREFIX / 'share' / 'cmake' / 'yaml-cpp'}",
                                                "-DBUILD_SHARED_LIBS=ON", "-DBUILD_TESTS=OFF", "-DUSE_ARUCO=OFF", "-DUSE_GTSAM=OFF",
                                                "-DBOW_FRAMEWORK=FBoW", "-DUSE_CCACHE=OFF", "-DUSE_SANITIZER=OFF",
                                                "-DBUILD_WITH_MARCH_NATIVE=OFF"], base)

    print("== qtrack")
    cmake_build(TRK, BUILD / "qtrack", common + [f"-Dstella_vslam_DIR={PREFIX / 'lib' / 'cmake' / 'stella_vslam'}",
                                                 f"-Dg2o_DIR={PREFIX / 'lib' / 'cmake' / 'g2o'}",
                                                 f"-Dyaml-cpp_DIR={PREFIX / 'share' / 'cmake' / 'yaml-cpp'}"], base)

    print("== ONNX Runtime for Android (prebuilt)")
    ort = download("onnxruntime_android", record)
    if not (ort / "headers").is_dir() or not (ort / "jni" / "arm64-v8a" / "libonnxruntime.so").is_file():
        raise ToolError(f"unexpected ONNX Runtime package layout in {ort}")

    print("== qrecon_quest")
    cmake_build(HERE / "questpipe", BUILD / "qrecon_quest", common + [f"-DNLOHMANN_JSON_INCLUDE={st / '3rd' / 'json' / 'include'}",
                                                                      f"-DORT_ROOT={ort}"], base)

    print("== collect")
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("bin", "lib", "share", "licenses"):
        (OUT / d).mkdir(parents=True)
    strip = find_tool(ndk, "llvm-strip")
    shutil.copy2(PREFIX / "bin" / "qtrack", OUT / "bin" / "qtrack")
    shutil.copy2(PREFIX / "bin" / "qrecon_quest", OUT / "bin" / "qrecon_quest")
    shutil.copy2(ort / "jni" / "arm64-v8a" / "libonnxruntime.so", OUT / "lib" / "libonnxruntime.so")
    for so in list(PREFIX.glob("lib/*.so")) + list(PREFIX.glob("lib64/*.so")):
        shutil.copy2(so, OUT / "lib" / so.name)
    cxx = next(iter(sorted((ndk / "toolchains" / "llvm" / "prebuilt").glob("*/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"))), None)
    if cxx is None:
        raise ToolError("libc++_shared.so for aarch64 not found in the NDK")
    shutil.copy2(cxx, OUT / "lib" / "libc++_shared.so")
    omp = next(iter(sorted((ndk / "toolchains" / "llvm" / "prebuilt").glob("*/lib/clang/*/lib/linux/aarch64/libomp.so"))), None)
    if omp is not None:                                   # stella links OpenMP; clang links the shared runtime
        shutil.copy2(omp, OUT / "lib" / "libomp.so")
    missing = missing_libraries(find_tool(ndk, "llvm-readelf"), OUT)
    if missing:
        raise ToolError("these libraries are needed on the headset but are not in the output: "
                        + ", ".join(f"{lib} (by {', '.join(users)})" for lib, users in sorted(missing.items())))
    for f in list((OUT / "bin").iterdir()) + list((OUT / "lib").iterdir()):
        if f.name != "libonnxruntime.so":            # prebuilt and already stripped
            run([strip, "--strip-unneeded", f])
    vocab = git_fetch("orb_vocab")
    shutil.copy2(vocab / "orb_vocab.fbow", OUT / "share" / "orb_vocab.fbow")
    lic = {"stella_vslam": st / "LICENSE", "stella_vslam_fork": st / "LICENSE.fork", "stella_vslam_original": st / "LICENSE.original",
           "g2o": g2o / "doc" / "license-bsd.txt", "yaml-cpp": yc / "LICENSE",
           "orb_vocab": vocab / "LICENSE", "eigen_MPL2": eigen / "COPYING.MPL2", "fbow": st / "3rd" / "FBoW" / "LICENSE",
           "nlohmann_json": st / "3rd" / "json" / "LICENSE.MIT"}
    record["licences_without_file"] = {"onnxruntime": "MIT (Microsoft); the Android package carries no licence file",
                                       "opencv_android_sdk": "Apache-2.0", "sqlite": "public domain"}
    if not lic["nlohmann_json"].is_file():
        lic["nlohmann_json"] = json_licence(st)
    for name, p in lic.items():
        if p.is_file():
            shutil.copy2(p, OUT / "licenses" / f"{name}.txt")
        else:
            print(f"  note: licence file for {name} not found at {p}; copy it by hand")
    record["files"] = {str(p.relative_to(OUT)).replace("\\", "/"): sha256_of(p) for p in OUT.rglob("*") if p.is_file()}
    (OUT / "build.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"done -> {OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ndk")
    ap.add_argument("--jobs", type=int, default=4)
    try:
        build(ap.parse_args())
    except ToolError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
