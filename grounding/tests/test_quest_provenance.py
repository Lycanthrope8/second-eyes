"""A2.5 delivery 1: the read-only provenance collector (ChatGPT's request of 9 October).

Run directly (python grounding/tests/test_quest_provenance.py) or as a module. It uses a fixture model folder with a
Hugging Face snapshot, download metadata, an export record, and a hand-written GGUF header carrying the qwen2 keys, plus a
fake r006 run with checkpoint evidence. Expectations are written by hand.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []
REV = json.loads((REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct.json").read_text(encoding="utf-8"))["hf_revision"]


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def gguf(path, meta, tensors):
    """A GGUF v3 header: scalar metadata (str, u32, f32) and tensor entries; no tensor data (only the header is read)."""
    def s(x):
        b = x.encode("utf-8")
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        if isinstance(v, str):
            out += s(k) + struct.pack("<I", 8) + s(v)
        elif isinstance(v, float):
            out += s(k) + struct.pack("<If", 6, v)
        else:
            out += s(k) + struct.pack("<II", 4, v)
    for name, shape, typ in tensors:
        out += s(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape) + struct.pack("<IQ", typ, 0)
    Path(path).write_bytes(out)


CONFIG = {"model_type": "qwen2", "hidden_size": 896, "intermediate_size": 4864, "num_attention_heads": 14, "num_key_value_heads": 2,
          "num_hidden_layers": 24, "max_position_embeddings": 32768, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
          "vocab_size": 151936, "tie_word_embeddings": True, "rope_scaling": None, "torch_dtype": "bfloat16"}
META = {"general.architecture": "qwen2", "general.file_type": 7, "qwen2.embedding_length": 896, "qwen2.feed_forward_length": 4864,
        "qwen2.attention.head_count": 14, "qwen2.attention.head_count_kv": 2, "qwen2.block_count": 24, "qwen2.context_length": 32768,
        "qwen2.attention.layer_norm_rms_epsilon": 1e-06, "qwen2.rope.freq_base": 1000000.0}
TENSORS = [("token_embd.weight", [896, 151936], 8), ("output_norm.weight", [896], 0)]


def fixture(root, config=None, meta=None, with_export=True, weights=b"labelled fake weights"):
    md = root / "model"
    hf = md / "hf"
    (hf / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (hf / "config.json").write_text(json.dumps(config or CONFIG), encoding="utf-8")
    (hf / "generation_config.json").write_text("{}", encoding="utf-8")
    (hf / "model.safetensors").write_bytes(weights)
    for n in ("config.json", "model.safetensors"):
        (hf / ".cache" / "huggingface" / "download" / f"{n}.metadata").write_text(f"{REV}\netag-{n}\n1.0\n", encoding="utf-8")
    (md / "gguf").mkdir()
    g = md / "gguf" / "qwen2.5-0.5b-instruct-q8_0.gguf"
    gguf(g, meta or META, TENSORS)
    if with_export:
        (md / "export.json").write_text(json.dumps({"variants": {"q8_0": {"folder": "gguf", "files": {g.name: {
            "sha256": hashlib.sha256(g.read_bytes()).hexdigest(), "bytes": g.stat().st_size}}, "llama_cpp": "b11277 (eae11d22)",
            "converter": "convert_hf_to_gguf.py"}}}), encoding="utf-8")
    return md, g


def r006(root, weights_sha):
    run = root / "data" / "run-small"
    run.mkdir(parents=True)
    ev = {"revision": REV, "tie": "every file's download metadata names the revision",
          "files": {"model.safetensors": {"sha256": weights_sha, "bytes": 21}, "config.json": {"sha256": "x", "bytes": 1}}}
    (run / "summary.json").write_text(json.dumps({"sessions": [{"evidence": ev}]}), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"record_type": "fake run manifest"}), encoding="utf-8")
    return run, ("20261007_A2_r006", hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest())


def main() -> int:
    PV = importlib.import_module("grounding.quest.provenance")
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        md, g = fixture(tmp / "a")
        wsha = hashlib.sha256((md / "hf" / "model.safetensors").read_bytes()).hexdigest()
        cfg_sha = hashlib.sha256((md / "hf" / "config.json").read_bytes()).hexdigest()
        run, pin = r006(tmp / "a", wsha)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted((tmp / "a").rglob("*")) if p.is_file()}
        s = PV.collect(model_dir=md, out=tmp / "p1", search=[tmp / "a" / "data"], gguf=g, r006_pin=pin)
        rows = {r["item"]: r for r in s["comparison"]}
        print("-- a consistent fixture")
        for item in ("config hidden_size against GGUF qwen2.embedding_length", "config rope_theta against GGUF qwen2.rope.freq_base",
                     "config rms_norm_eps against GGUF qwen2.attention.layer_norm_rms_epsilon",
                     "config num_key_value_heads against GGUF qwen2.attention.head_count_kv",
                     "config tie_word_embeddings against the GGUF's output.weight (absent when tied)",
                     "config vocab_size and hidden_size against token_embd.weight's shape",
                     "config rope_scaling against the GGUF's rope scaling keys", "config model_type against the GGUF architecture",
                     "snapshot download metadata names the description's revision",
                     "r006 checkpoint evidence names the description's revision",
                     "model.safetensors: r006's recorded hash against the laptop snapshot's today",
                     "export.json's llama.cpp release is the pinned one"):
            check(f"agrees: {item}", rows.get(item, {}).get("status") == "agrees", str(rows.get(item)))
        check("r006's config.json hash is recorded as differing from today's file, not hidden",
              rows["config.json: r006's recorded hash against the laptop snapshot's today"]["status"] == "differs"
              and rows["config.json: r006's recorded hash against the laptop snapshot's today"]["today"]["sha256"] == cfg_sha)
        check("the fixture GGUF is not the accepted one, and says so", rows["today's GGUF is the accepted one"]["status"] == "differs"
              and rows["export.json's GGUF hash is the accepted one"]["status"] == "differs")
        orig = tmp / "p1" / "originals"
        check("historical records are copied byte for byte",
              (orig / "model" / "export.json").read_bytes() == (md / "export.json").read_bytes()
              and (orig / "model" / "hf" / "config.json").read_bytes() == (md / "hf" / "config.json").read_bytes()
              and (orig / "r006" / "summary.json").read_bytes() == (run / "summary.json").read_bytes()
              and (orig / "model" / "hf" / ".cache" / "huggingface" / "download" / "model.safetensors.metadata").is_file())
        obs = json.loads((tmp / "p1" / "observed.json").read_text(encoding="utf-8"))
        check("today's observations hold the weights' and GGUF's hashes and the GGUF's metadata, labelled with a date",
              obs["snapshot_files_sha256"]["model.safetensors"]["sha256"] == wsha and obs["gguf"]["metadata"]["qwen2.block_count"] == 24
              and obs["gguf"]["has_output_weight"] is False and obs["made_utc"])
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted((tmp / "a").rglob("*")) if p.is_file()}
        check("read-only: every source file is byte-identical, with the same modification time", before == after)
        man = json.loads((tmp / "p1" / "manifest.json").read_text(encoding="utf-8"))
        check("the manifest hashes every file it published",
              all(hashlib.sha256((tmp / "p1" / k).read_bytes()).hexdigest() == v for k, v in man["files"].items()))
        print("-- differences and gaps are reported, never filled in")
        md2, g2 = fixture(tmp / "b", config=dict(CONFIG, rope_theta=10000.0, tie_word_embeddings=False), with_export=False)
        s = PV.collect(model_dir=md2, out=tmp / "p2", search=[tmp / "b"], gguf=g2, r006_pin=pin)
        rows = {r["item"]: r for r in s["comparison"]}
        check("a rope_theta of 10,000 against the GGUF's 1,000,000 differs",
              rows["config rope_theta against GGUF qwen2.rope.freq_base"]["status"] == "differs")
        check("untied embeddings in config against a GGUF without output.weight differ",
              rows["config tie_word_embeddings against the GGUF's output.weight (absent when tied)"]["status"] == "differs")
        check("no export.json: the row and the missing list say so",
              rows["export.json records the accepted GGUF"]["status"] == "missing" and any("export.json" in m for m in s["missing"]))
        check("r006 not found under the searched folders: its evidence is missing, not assumed",
              rows["r006 checkpoint evidence"]["status"] == "missing" and any("not found" in m for m in s["missing"]))
        md3, g3 = fixture(tmp / "c", weights=b"other weights")
        run3, pin3 = r006(tmp / "c", wsha)
        s = PV.collect(model_dir=md3, out=tmp / "p3", search=[tmp / "c" / "data"], gguf=g3, r006_pin=pin3)
        rows = {r["item"]: r for r in s["comparison"]}
        check("other weights today than r006 recorded: differs",
              rows["model.safetensors: r006's recorded hash against the laptop snapshot's today"]["status"] == "differs")
        try:
            PV.collect(model_dir=md, out=tmp / "p1", search=[tmp / "a" / "data"], gguf=g, r006_pin=pin)
            refused = False
        except EI:
            refused = True
        check("an existing destination is refused", refused)
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
