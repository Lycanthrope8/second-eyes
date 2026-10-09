"""A2.5 delivery 1: read-only provenance of the conversion and the configuration (ChatGPT's request of 9 October).

It changes nothing, downloads nothing, loads no model and runs no inference. It writes only a new folder.

**Historical records**, copied byte for byte into `originals/`:

- the model folder's `export.json`, the GGUF export record;
- the Hugging Face snapshot's `config.json`, `generation_config.json` and download metadata
  (`.cache/huggingface/download/*.metadata`);
- the JSON records of run `20261007_A2_r006`, whose float32 rows are the replay's references, found by their pinned
  manifest hash.

**Today's observations**, labelled as such:

- the SHA-256 of every copied file, of the snapshot's weight files, and of the GGUF;
- the GGUF header's metadata and tensor shapes (only the header is read);
- the SHA-256 of the converter script in the pinned llama.cpp checkout, if present.

**The comparison** reports each item as agreeing, differing or missing. Missing items are never filled in.

- **Checkpoint:** the model description's revision; the snapshot's download metadata; r006's checkpoint evidence
  (revision, tie and per-file hashes); today's hashes of the laptop's snapshot.
- **Conversion:** `export.json`'s GGUF hash and llama.cpp release, against the accepted GGUF and the pinned release.
- **Configuration:** the snapshot's `config.json`, against the GGUF's `qwen2.*` keys and tensor layout. That covers the
  architecture and RoPE settings the source review could not check.
"""
from __future__ import annotations

import datetime
import json
import shutil
import struct
import tempfile
from pathlib import Path

from ..evaluation.iref_vla import output
from ..evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError, encode_json, issue, runtime, sha256
from .binaries import Unsupported, gguf_header
from .publish import publish
from .runtime_identity import REPO, file_sha256

ACCEPTED_GGUF_SHA256 = "dd753cd62f163c8baa8d2e598e3b61385f31cd46ca04488cd88ba01a9c83eb18"
ACCEPTED_LLAMA_CPP = "b11277 (eae11d22)"
R006 = ("20261007_A2_r006", "f06cb6c6f2bc0eca9f89bebc156c2c315434dc5307d191b03267d43d67e5baf6")
DESCRIPTION = REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct.json"
GGUF_NAME = "qwen2.5-0.5b-instruct-q8_0.gguf"
CONFIG_TO_GGUF = (   # config.json key, GGUF key, how the GGUF stores it
    ("hidden_size", "qwen2.embedding_length", "int"), ("intermediate_size", "qwen2.feed_forward_length", "int"),
    ("num_attention_heads", "qwen2.attention.head_count", "int"), ("num_key_value_heads", "qwen2.attention.head_count_kv", "int"),
    ("num_hidden_layers", "qwen2.block_count", "int"), ("max_position_embeddings", "qwen2.context_length", "int"),
    ("rms_norm_eps", "qwen2.attention.layer_norm_rms_epsilon", "f32"), ("rope_theta", "qwen2.rope.freq_base", "f32"))


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def find_run(manifest_sha256, roots):
    """The folder whose manifest.json has the pinned SHA-256, searched read-only under the given roots."""
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for m in root.rglob("manifest.json"):
            try:
                if file_sha256(m) == manifest_sha256:
                    return m.parent
            except OSError:
                continue
    return None


def _evidence_in(obj):
    """Checkpoint evidence dicts (revision, tie, files) found anywhere in a JSON value."""
    found = []
    if isinstance(obj, dict):
        if {"revision", "tie", "files"} <= set(obj):
            found.append(obj)
        for v in obj.values():
            found += _evidence_in(v)
    elif isinstance(obj, list):
        for v in obj:
            found += _evidence_in(v)
    return found


def collect(*, model_dir, out, search=(), gguf=None, r006_pin=R006) -> dict:
    out = output.refuse_existing(out)
    md = Path(model_dir)
    hf, gguf = md / "hf", Path(gguf) if gguf is not None else md / "gguf" / GGUF_NAME
    originals, observed, missing = {}, {"made_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, []

    def keep(src, rel):
        if src.is_file():
            originals[rel] = src.read_bytes()
            return True
        missing.append(f"historical record {rel}: no file at {src}")
        return False

    keep(md / "export.json", "model/export.json")
    keep(hf / "config.json", "model/hf/config.json")
    keep(hf / "generation_config.json", "model/hf/generation_config.json")
    meta_dir = hf / ".cache" / "huggingface" / "download"
    metas = sorted(meta_dir.glob("*.metadata")) if meta_dir.is_dir() else []
    if not metas:
        missing.append(f"historical record: no Hugging Face download metadata under {meta_dir}")
    for m in metas:
        keep(m, f"model/hf/.cache/huggingface/download/{m.name}")
    r006 = find_run(r006_pin[1], list(search) + [REPO / "runs" / r006_pin[0]])
    if r006 is None:
        missing.append(f"historical record: run {r006_pin[0]}'s folder (manifest SHA-256 {r006_pin[1][:8]}...) was not found under the "
                       "searched folders")
    else:
        for p in sorted(r006.glob("*.json")) + sorted(r006.glob("*.jsonl")):
            keep(p, f"r006/{p.name}")
        observed["r006_folder"] = str(r006)
    # today's observations
    observed["originals_sha256"] = {k: sha256(v) for k, v in originals.items()}
    weights = sorted(hf.glob("*.safetensors")) if hf.is_dir() else []
    observed["snapshot_files_sha256"] = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                                         for p in weights + [hf / "config.json"] if p.is_file()}
    if not weights:
        missing.append(f"observation: no weight files (*.safetensors) under {hf}")
    if gguf.is_file():
        try:
            g = gguf_header(gguf, details=True)
            observed["gguf"] = {"path": str(gguf), "sha256": file_sha256(gguf), "bytes": gguf.stat().st_size,
                                "summary": {k: v for k, v in g.items() if k not in ("metadata", "tensor_shapes")},
                                "metadata": {k: v for k, v in g["metadata"].items() if not k.startswith("tokenizer.ggml.")
                                             or not isinstance(v, (list, dict))},
                                "tensor_shapes": {k: v for k, v in g["tensor_shapes"].items()
                                                  if k in ("token_embd.weight", "output.weight", "output_norm.weight")},
                                "tensor_names_count": len(g["tensor_shapes"]), "has_output_weight": "output.weight" in g["tensor_shapes"]}
        except (Unsupported, OSError) as e:
            missing.append(f"observation: the GGUF header could not be read ({e})")
    else:
        missing.append(f"observation: no GGUF at {gguf}")
    conv = REPO / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"
    observed["converter_script_sha256_today"] = file_sha256(conv) if conv.is_file() else None
    if not conv.is_file():
        missing.append(f"observation: no converter script at {conv}")
    observed["llama_cpp_pin"] = (REPO / "native" / "llama.cpp.pin").read_text(encoding="utf-8").strip() \
        if (REPO / "native" / "llama.cpp.pin").is_file() else None
    cmp_rows = compare(originals, observed)
    summary = {"format_version": 1, "record_type": "a25_provenance", "missing": missing, "comparison": cmp_rows,
               "counts": {s: sum(r["status"] == s for r in cmp_rows) for s in ("agrees", "differs", "missing")},
               "read_only": "nothing outside this folder was written; no model was loaded and nothing was downloaded"}
    files = {f"originals/{k}": v for k, v in originals.items()}
    files["observed.json"] = encode_json(observed)
    files["summary.json"] = encode_json(summary)
    files["report.md"] = render(summary, observed).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for rel, data in files.items():
            (staging / rel).parent.mkdir(parents=True, exist_ok=True)
            (staging / rel).write_bytes(data)
        man = {"format_version": 1, "record_type": "a25_provenance_manifest",
               "files": {p.relative_to(staging).as_posix(): sha256(p.read_bytes()) for p in sorted(staging.rglob("*")) if p.is_file()},
               "code": {f"grounding/quest/{p.name}": sha256(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
               "runtime": runtime()}
        (staging / "manifest.json").write_bytes(encode_json(man))
        publish(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    return dict(summary, folder=str(out))


def _row(item, status, historical=None, today=None, note=""):
    return {"item": item, "status": status, "historical": historical, "today": today, "note": note}


def compare(originals, observed) -> list:
    rows = []
    desc = json.loads(DESCRIPTION.read_text(encoding="utf-8"))
    rev = desc["hf_revision"]
    commits = sorted({x.split("\n")[0].strip() for k, x in ((k, v.decode("utf-8", "replace")) for k, v in originals.items())
                      if k.endswith(".metadata") and x.strip()})
    rows.append(_row("snapshot download metadata names the description's revision",
                     "missing" if not commits else "agrees" if commits == [rev] else "differs", {"commits": commits}, None,
                     f"model description revision {rev}"))
    ev = []
    for k, v in originals.items():
        if k.startswith("r006/"):
            for line in v.decode("utf-8", "replace").splitlines() if k.endswith(".jsonl") else [v.decode("utf-8", "replace")]:
                try:
                    ev += _evidence_in(json.loads(line))
                except ValueError:
                    continue
    uniq = {json.dumps(e, sort_keys=True): e for e in ev}
    if not uniq:
        rows.append(_row("r006 checkpoint evidence", "missing", None, None, "no checkpoint evidence found in r006's records"))
    else:
        e = list(uniq.values())[0]
        rows.append(_row("r006 checkpoint evidence names the description's revision", "agrees" if e["revision"] == rev else "differs",
                         {"revision": e["revision"], "tie": e["tie"], "distinct_evidence_records": len(uniq)}))
        today = observed.get("snapshot_files_sha256", {})
        for name, f in sorted(e["files"].items()):
            if name.endswith(".safetensors") or name == "config.json":
                t = today.get(name)
                rows.append(_row(f"{name}: r006's recorded hash against the laptop snapshot's today",
                                 "missing" if t is None else "agrees" if t["sha256"] == f.get("sha256") else "differs",
                                 {"sha256": f.get("sha256"), "bytes": f.get("bytes")}, t))
    exp = json.loads(originals["model/export.json"]) if "model/export.json" in originals else None
    g = observed.get("gguf")
    if exp is None:
        rows.append(_row("export.json records the accepted GGUF", "missing"))
    else:
        recs = [(name, f, v) for v in exp.get("variants", {}).values() for name, f in v.get("files", {}).items() if name == GGUF_NAME]
        if not recs:
            rows.append(_row("export.json records the accepted GGUF", "missing", exp, None, f"no {GGUF_NAME} entry"))
        else:
            name, f, v = recs[0]
            rows.append(_row("export.json's GGUF hash is the accepted one", "agrees" if f["sha256"] == ACCEPTED_GGUF_SHA256 else "differs",
                             f["sha256"], ACCEPTED_GGUF_SHA256))
            rows.append(_row("export.json's llama.cpp release is the pinned one",
                             "agrees" if v.get("llama_cpp") == ACCEPTED_LLAMA_CPP else "differs", v.get("llama_cpp"), ACCEPTED_LLAMA_CPP,
                             f"converter {v.get('converter')}"))
    rows.append(_row("today's GGUF is the accepted one", "missing" if not g else "agrees" if g["sha256"] == ACCEPTED_GGUF_SHA256 else "differs",
                     ACCEPTED_GGUF_SHA256, (g or {}).get("sha256")))
    cfg = json.loads(originals["model/hf/config.json"]) if "model/hf/config.json" in originals else None
    meta = (g or {}).get("metadata", {})
    for ck, gk, kind in CONFIG_TO_GGUF:
        hv, gv = (cfg or {}).get(ck), meta.get(gk)
        if hv is None or gv is None:
            rows.append(_row(f"config {ck} against GGUF {gk}", "missing", hv, gv))
        else:
            same = (int(hv) == int(gv)) if kind == "int" else (_f32(hv) == _f32(gv))
            rows.append(_row(f"config {ck} against GGUF {gk}", "agrees" if same else "differs", hv, gv,
                             "compared as float32, the GGUF's storage" if kind == "f32" else ""))
    if cfg is not None and g:
        tied = cfg.get("tie_word_embeddings")
        rows.append(_row("config tie_word_embeddings against the GGUF's output.weight (absent when tied)",
                         "agrees" if tied == (not g["has_output_weight"]) else "differs", tied, {"has_output_weight": g["has_output_weight"]}))
        emb = g["tensor_shapes"].get("token_embd.weight")
        rows.append(_row("config vocab_size and hidden_size against token_embd.weight's shape",
                         "missing" if not emb else "agrees" if sorted(emb) == sorted([cfg.get("vocab_size"), cfg.get("hidden_size")]) else "differs",
                         [cfg.get("vocab_size"), cfg.get("hidden_size")], emb))
        scaling = cfg.get("rope_scaling")
        gscale = {k: v for k, v in meta.items() if k.startswith("qwen2.rope.scaling")}
        rows.append(_row("config rope_scaling against the GGUF's rope scaling keys",
                         "agrees" if (scaling in (None, {}) and not gscale) else "differs" if (scaling in (None, {})) != (not gscale) else "missing",
                         scaling, gscale or None, "none on either side counts as agreeing; anything else needs a look"))
        rows.append(_row("config model_type against the GGUF architecture",
                         "agrees" if cfg.get("model_type") == g["summary"].get("architecture") else "differs",
                         cfg.get("model_type"), g["summary"].get("architecture")))
    return rows


def render(s, observed) -> str:
    c = s["counts"]
    L = ["# Conversion and configuration provenance (A2.5 delivery 1)", "",
         f"{c['agrees']} agree, {c['differs']} differ, {c['missing']} missing. Historical records are copied unchanged under "
         "`originals/`; today's observations are in `observed.json`. " + s["read_only"] + ".", "",
         "| Item | Status | Historical | Today |", "|---|---|---|---|"]
    for r in s["comparison"]:
        h = json.dumps(r["historical"], sort_keys=True)[:90] if r["historical"] is not None else "-"
        t = json.dumps(r["today"], sort_keys=True)[:90] if r["today"] is not None else "-"
        L.append(f"| {r['item']} | {r['status']} | `{h}` | `{t}` |")
    if s["missing"]:
        L += ["", "Missing:", ""] + [f"- {m}" for m in s["missing"]]
    return "\n".join(L) + "\n"
