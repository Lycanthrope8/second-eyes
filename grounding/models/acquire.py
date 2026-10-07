"""Download a pinned Hugging Face model's files and prove they are the pinned ones (A2.3d).

    python -m grounding.models.acquire --description grounding/models/qwen2.5-7b-instruct.json --out DIR [--only tokenizer]

Every file named in the description is fetched with huggingface_hub at the pinned revision into DIR (the local-folder
mode, which writes .cache/huggingface/download/<file>.metadata with the commit and ETag), or kept if DIR already holds a
copy that verifies. Then every file is checked:
- its size equals the description's;
- a weight file's SHA-256 equals the description's pin (published by Hugging Face's API on 7 October 2026);
- the download metadata names the pinned commit, and its ETag matches the file (SHA-256 for a large file, the Git blob
  SHA-1 for a small one), which is how A2.3a's checkpoint check ties files to a revision;
- config.json's architecture equals the description's.
The record, `acquisition.json` in DIR, lists every file's size, SHA-256, Git blob SHA-1 and metadata. Tokenizer and
config files have no published SHA-256, so their hashes are first recorded here and are frozen by A2.3d's request
preparation; both machines' copies must then agree.

`--only tokenizer` fetches the tokenizer files, config.json and the license (for the laptop, which prepares requests and
never loads the weights). Exit 0: verified. 2: an identity or description problem (nothing is deleted). 3: a download or
write failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

KINDS = ("weights", "config", "tokenizer", "license")


class AcquisitionError(Exception):
    def __init__(self, problems, code=2):
        super().__init__("; ".join(problems))
        self.problems, self.code = list(problems), code


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % p.stat().st_size)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_description(path) -> dict:
    p = Path(path)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise AcquisitionError([f"{p}: cannot read the description: {e}"]) from None
    bad = []
    if d.get("format") != 2:
        bad.append("format must be 2 (a description with file pins)")
    for k in ("name", "hf_repo", "hf_revision", "files", "license", "architecture"):
        if k not in d:
            bad.append(f"missing {k}")
    rev = d.get("hf_revision", "")
    if not (isinstance(rev, str) and len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)):
        bad.append("hf_revision must be a full 40-character commit hash")
    for name, f in (d.get("files") or {}).items():
        if not isinstance(f, dict) or f.get("kind") not in KINDS or type(f.get("bytes")) is not int or f["bytes"] < 0:
            bad.append(f"{name}: needs a kind in {KINDS} and a plain integer size")
        elif f["kind"] == "weights" and not (isinstance(f.get("sha256"), str) and len(f["sha256"]) == 64):
            bad.append(f"{name}: a weight file needs its published SHA-256")
    if bad:
        raise AcquisitionError([f"{p}: {m}" for m in bad])
    return d


def selected(desc, only) -> list:
    names = list(desc["files"])
    if only == "tokenizer":
        return [n for n in names if desc["files"][n]["kind"] in ("tokenizer", "license") or n == "config.json"]
    return names


def metadata_of(out: Path, name: str):
    m = out / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    if not m.is_file():
        return None
    lines = m.read_text(encoding="utf-8").splitlines()
    return {"commit": lines[0] if lines else None, "etag": (lines[1] if len(lines) > 1 else "").strip('"') or None}


def verify_file(out: Path, name: str, spec: dict, revision: str) -> tuple:
    p, problems = out / name, []
    if not p.is_file():
        return None, [f"{name}: missing"]
    rec = {"bytes": p.stat().st_size, "sha256": sha256_file(p), "git_blob_sha1": git_blob_sha1(p),
           "kind": spec["kind"], "metadata": metadata_of(out, name)}
    if rec["bytes"] != spec["bytes"]:
        problems.append(f"{name}: {rec['bytes']} bytes, the description says {spec['bytes']}")
    if spec["kind"] == "weights" and rec["sha256"] != spec["sha256"]:
        problems.append(f"{name}: SHA-256 {rec['sha256']} is not the pinned {spec['sha256']}")
    md = rec["metadata"]
    if md is None:
        problems.append(f"{name}: no download metadata, so nothing ties it to the pinned commit")
    else:
        if md["commit"] != revision:
            problems.append(f"{name}: the download metadata names commit {md['commit']}, not {revision}")
        etag = md["etag"] or ""
        if not (etag == rec["sha256"] if len(etag) == 64 else etag == rec["git_blob_sha1"]):
            problems.append(f"{name}: its contents do not match the download metadata's ETag")
    return rec, problems


def check_config(out: Path, arch: dict) -> list:
    p = out / "config.json"
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"config.json is unreadable: {e}"]
    bad = []
    for k, want in arch.items():
        if k == "min_max_position_embeddings":
            if cfg.get("max_position_embeddings", 0) < want:
                bad.append(f"config.json allows {cfg.get('max_position_embeddings')} positions, fewer than {want}")
        elif cfg.get(k) != want:
            bad.append(f"config.json has {k} = {cfg.get(k)!r}, the description says {want!r}")
    return bad


def _hub_download(repo_id, filename, revision, local_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=str(local_dir))


def acquire(*, description, out, only="all", downloader=None, progress=print) -> dict:
    desc = load_description(description)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names, rev = selected(desc, only), desc["hf_revision"]
    fetch = downloader or _hub_download
    fetched, kept = [], []
    for name in names:
        rec, problems = (verify_file(out, name, desc["files"][name], rev) if (out / name).is_file() else (None, ["missing"]))
        if rec is not None and not problems:
            kept.append(name)
            continue
        progress(f"  fetching {name} ({desc['files'][name]['bytes']:,} bytes)")
        try:
            fetch(desc["hf_repo"], name, rev, out)
        except Exception as e:  # noqa: BLE001
            raise AcquisitionError([f"{name}: download failed: {type(e).__name__}: {e}"], code=3) from e
        fetched.append(name)
    files, problems = {}, []
    for name in names:
        rec, bad = verify_file(out, name, desc["files"][name], rev)
        files[name] = rec
        problems += bad
    if "config.json" in names:
        problems += check_config(out, desc["architecture"])
    lic = desc["license"]["file"]
    if lic in names and files.get(lic) is None:
        problems.append("the license file is missing")
    if problems:
        raise AcquisitionError(problems)
    try:
        import huggingface_hub
        hub = huggingface_hub.__version__
    except ImportError:
        hub = None
    record = {"format_version": 1, "record_type": "model_acquisition", "name": desc["name"], "hf_repo": desc["hf_repo"],
              "hf_revision": rev, "only": only, "license_card": desc["license"]["card"],
              "license_file_sha256": files[lic]["sha256"] if lic in files else None,
              "description_sha256": hashlib.sha256(Path(description).read_bytes()).hexdigest(),
              "files": files, "fetched": fetched, "kept": kept,
              "acquired_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "runtime": {"python": platform.python_version(), "platform": platform.platform(), "huggingface_hub": hub}}
    try:
        (out / "acquisition.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as e:
        raise AcquisitionError([f"cannot write acquisition.json: {e}"], code=3) from e
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.models.acquire")
    ap.add_argument("--description", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=("all", "tokenizer"), default="all")
    a = ap.parse_args(argv)
    try:
        r = acquire(description=a.description, out=a.out, only=a.only)
    except AcquisitionError as e:
        for m in e.problems:
            print(f"problem: {m}", file=sys.stderr)
        return e.code
    print(f"{r['name']} at {r['hf_revision']}: {len(r['files'])} files verified "
          f"({len(r['fetched'])} fetched, {len(r['kept'])} already present); license {r['license_card']}")
    for name, f in sorted(r["files"].items()):
        print(f"  {f['sha256']}  {f['bytes']:>12,}  {name}")
    print(f"record: {Path(a.out) / 'acquisition.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
