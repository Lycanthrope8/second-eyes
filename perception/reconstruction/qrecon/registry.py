"""Run registry: every number traces back to a run folder with its manifest."""
import datetime
import json
import os
import subprocess
from . import config as C


def _git_commit(cwd):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _registry_settings():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "registry.yaml")
    return C.load(p) if os.path.exists(p) else {}


def _git_dirty(cwd):
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL).decode()
        return any(l and not l[3:].startswith(("runs/", "notes/", "docs/")) for l in out.splitlines())
    except Exception:
        return None


def new_run(root, kind, manifest):
    """Run IDs follow the repository's registry (D5: YYYYMMDD_<phase>_r<NNN>, counted per phase, never reset) once
    configs/registry.yaml names a phase. Until then they are provisional (qrecon_YYYYMMDD_rNNN) and marked so."""
    os.makedirs(root, exist_ok=True)
    now = datetime.datetime.now()
    reg = _registry_settings()
    phase = reg.get("phase")
    if phase:
        n = 1 + sum(1 for d in os.listdir(root) if f"_{phase}_r" in d)
        run_id = f"{now:%Y%m%d}_{phase}_r{n:03d}"
    else:
        day = now.strftime("%Y%m%d")
        n = 1 + sum(1 for d in os.listdir(root) if d.startswith(f"qrecon_{day}_r"))
        run_id = f"qrecon_{day}_r{n:03d}"
    path = os.path.join(root, run_id)
    os.makedirs(path)
    here = os.path.dirname(os.path.abspath(__file__))
    manifest = dict(manifest, config_version=1, run_id=run_id, phase=phase, provisional_run_id=not bool(phase), kind=kind,
                    created=now.isoformat(timespec="seconds"), operator=reg.get("operator"),
                    purpose=manifest.get("purpose", kind), git_commit=_git_commit(here), git_dirty=_git_dirty(here))
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return run_id, path


def save_json(path, name, obj):
    with open(os.path.join(path, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path, name):
    with open(os.path.join(path, name), "r", encoding="utf-8") as f:
        return json.load(f)


def freeze_state(repo_root):
    p = os.path.join(repo_root, "configs", "freeze.json")
    return json.load(open(p, "r", encoding="utf-8")) if os.path.exists(p) else None


def check_frozen(repo_root, cfg_paths, spec_path):
    """True only if a freeze record exists and still matches the spec and configs."""
    st = freeze_state(repo_root)
    if not st:
        return False, "no freeze record"
    for p in list(cfg_paths) + [spec_path]:
        rel = os.path.relpath(p, repo_root).replace("\\", "/")
        if st["files"].get(rel) != C.file_sha256(p):
            return False, f"{rel} changed after the freeze"
    return True, "frozen"
