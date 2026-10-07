"""Checks for grounding.models.acquire (A2.3d): pinned downloads verified by size, SHA-256, commit and ETag.

Run directly (python grounding/tests/test_model_acquire.py) or as a module. A labelled fake downloader stands in for
Hugging Face; no network is used.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []
REV = "a" * 40
CONFIG = {"num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "hidden_size": 8,
          "vocab_size": 50, "tie_word_embeddings": False, "max_position_embeddings": 32768}
CONTENT = {"config.json": json.dumps(CONFIG).encode(), "model-00001-of-00002.safetensors": b"W" * 100,
           "model-00002-of-00002.safetensors": b"V" * 70, "tokenizer.json": b'{"t": 1}', "vocab.json": b'{"a": 0}',
           "merges.txt": b"#version\n", "tokenizer_config.json": b"{}", "LICENSE": b"Apache License 2.0\n"}


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def description(tmp: Path, **over) -> Path:
    files = {}
    for n, b in CONTENT.items():
        kind = "weights" if n.endswith(".safetensors") else "license" if n == "LICENSE" else \
            "config" if n == "config.json" else "tokenizer"
        files[n] = {"bytes": len(b), "kind": kind}
        if kind == "weights":
            files[n]["sha256"] = hashlib.sha256(b).hexdigest()
    arch = {k: v for k, v in CONFIG.items() if k != "max_position_embeddings"}
    arch["min_max_position_embeddings"] = 8192
    d = {"format": 2, "name": "fake-model", "hf_repo": "fake/model", "hf_revision": REV, "license": {"card": "apache-2.0",
         "file": "LICENSE"}, "architecture": arch, "files": files}
    for k, v in over.items():
        d[k] = v
    p = tmp / "desc.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


class FakeHub:
    """Writes the file and Hugging Face's local-folder metadata (commit, ETag, time), with optional faults."""

    def __init__(self, content=None, commit=REV, bad_etag=(), fail=()):
        self.content, self.commit, self.bad_etag, self.fail, self.calls = dict(content or CONTENT), commit, bad_etag, fail, []

    def __call__(self, repo_id, filename, revision, local_dir):
        self.calls.append(filename)
        if filename in self.fail:
            raise ConnectionError("simulated network failure")
        out = Path(local_dir)
        b = self.content[filename]
        (out / filename).write_bytes(b)
        etag = hashlib.sha256(b).hexdigest() if filename.endswith(".safetensors") else blob(b)
        if filename in self.bad_etag:
            etag = "0" * len(etag)
        meta = out / ".cache" / "huggingface" / "download"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / f"{filename}.metadata").write_text(f"{self.commit}\n{etag}\n1700000000.0\n", encoding="utf-8")


def outcome(fn):
    A = importlib.import_module("grounding.models.acquire")
    try:
        return ("ok", fn())
    except A.AcquisitionError as e:
        return ("error", e.code, e.problems)


def main() -> int:
    A = importlib.import_module("grounding.models.acquire")
    print("-- grounding.models.acquire")
    check("the Git blob SHA-1 of 'hello\\n' is ce013625... (git hash-object)", blob(b"hello\n") ==
          "ce013625030ba8dba906f756967f9e9ca394464a")
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        p = t / "f"; p.write_bytes(b"hello\n")
        check("the module's blob hash agrees", A.git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a")
        d = description(t)
        hub = FakeHub()
        r = outcome(lambda: A.acquire(description=d, out=t / "m", downloader=hub, progress=lambda m: None))
        check("all files fetched and verified; the record lists every file", r[0] == "ok" and len(r[1]["files"]) == 8
              and sorted(hub.calls) == sorted(CONTENT) and (t / "m" / "acquisition.json").is_file(), str(r[:2]))
        hub2 = FakeHub()
        r2 = outcome(lambda: A.acquire(description=d, out=t / "m", downloader=hub2, progress=lambda m: None))
        check("a second run keeps verified files and downloads nothing", r2[0] == "ok" and hub2.calls == []
              and len(r2[1]["kept"]) == 8)
        hub3 = FakeHub()
        r3 = outcome(lambda: A.acquire(description=d, out=t / "tok", only="tokenizer", downloader=hub3, progress=lambda m: None))
        check("--only tokenizer fetches the tokenizer files, config.json and the license, never weights",
              r3[0] == "ok" and sorted(hub3.calls) == sorted(["config.json", "tokenizer.json", "vocab.json", "merges.txt",
                                                              "tokenizer_config.json", "LICENSE"]))
        bad = dict(CONTENT); bad["model-00002-of-00002.safetensors"] = b"X" * 70
        r4 = outcome(lambda: A.acquire(description=d, out=t / "w", downloader=FakeHub(content=bad), progress=lambda m: None))
        check("a weight file with another SHA-256 is refused (exit 2)", r4[0] == "error" and r4[1] == 2
              and any("not the pinned" in m for m in r4[2]), str(r4[:3]))
        r5 = outcome(lambda: A.acquire(description=d, out=t / "c", downloader=FakeHub(commit="b" * 40), progress=lambda m: None))
        check("download metadata naming another commit is refused", r5[0] == "error" and any("names commit" in m for m in r5[2]))
        r6 = outcome(lambda: A.acquire(description=d, out=t / "e", downloader=FakeHub(bad_etag=("vocab.json",)),
                                       progress=lambda m: None))
        check("a small file whose contents do not match its ETag is refused", r6[0] == "error"
              and any(m.startswith("vocab.json") and "ETag" in m for m in r6[2]))
        r7 = outcome(lambda: A.acquire(description=d, out=t / "n", downloader=FakeHub(fail=("tokenizer.json",)),
                                       progress=lambda m: None))
        check("a download failure exits 3", r7[0] == "error" and r7[1] == 3)
        d2 = description(t, architecture=dict(json.loads(d.read_text())["architecture"], num_key_value_heads=2))
        r8 = outcome(lambda: A.acquire(description=d2, out=t / "a", downloader=FakeHub(), progress=lambda m: None))
        check("config.json that differs from the description's architecture is refused", r8[0] == "error"
              and any("num_key_value_heads" in m for m in r8[2]))
        files = json.loads(d.read_text())["files"]; files["merges.txt"]["bytes"] += 1
        d3 = description(t, files=files)
        r9 = outcome(lambda: A.acquire(description=d3, out=t / "s", downloader=FakeHub(), progress=lambda m: None))
        check("a size that differs from the description is refused", r9[0] == "error" and any("bytes" in m for m in r9[2]))
        for label, over in (("format 1", {"format": 1}), ("a short revision", {"hf_revision": "a09a3545"})):
            r10 = outcome(lambda: A.acquire(description=description(t, **over), out=t / "x", downloader=FakeHub(),
                                            progress=lambda m: None))
            check(f"a description with {label} is refused before any download", r10[0] == "error" and r10[1] == 2)
    real = A.load_description(REPO / "grounding" / "models" / "qwen2.5-7b-instruct.json")
    w = [n for n, f in real["files"].items() if f["kind"] == "weights"]
    check("the shipped 7B description loads: revision a09a3545..., four pinned shards, Apache-2.0",
          real["hf_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28" and len(w) == 4
          and real["license"]["card"] == "apache-2.0")
    check("its tokenizer subset is the four tokenizer files, config.json and LICENSE",
          sorted(A.selected(real, "tokenizer")) == sorted(["config.json", "tokenizer.json", "tokenizer_config.json",
                                                           "vocab.json", "merges.txt", "LICENSE"]))
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
