"""A2.7 export (D115): the deployment candidate's merged model to GGUF Q8_0, with llama.cpp's own converter at the pinned
release.

    python -m grounding.a27 export --merged MERGED --llama LLAMA_DIR --out FOLDER

MERGED is A2.4's merged folder of the candidate. Before converting, the export checks: the identity names the candidate
D115 records (a24.parser.s1, the parser design, the checkpoint selected at epoch 2); every merged file still has the
hash the identity recorded; the llama.cpp checkout is at native/llama.cpp.pin's commit. The converter runs in this
Python, which needs torch, transformers, numpy and sentencepiece (the converter tries a SentencePiece tokenizer before
Qwen's BPE one). Writes <model>.q8_0.gguf, the converter's log, and export.json: the identity's hash, the command, the
software versions, the GGUF's SHA-256 and its metadata read back (architecture, file type, tokens, tensor types, and a
hash of every tokenizer field, which the desktop run compares with the accepted base GGUF's).
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIN = ROOT / "native" / "llama.cpp.pin"
CANDIDATE = {"model_name": "a24.parser.s1", "design": "parser", "epoch": 2}
OUTTYPE = "q8_0"
FILE_TYPE_Q8_0 = 7   # llama_ftype LLAMA_FTYPE_MOSTLY_Q8_0


class ExportError(RuntimeError):
    pass


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_commit() -> tuple:
    release, commit = PIN.read_text(encoding="utf-8").split()[:2]
    return release, commit


def check_candidate(merged: Path, candidate=CANDIDATE) -> dict:
    """The identity of the merged folder, checked against the candidate and against its own recorded file hashes."""
    raw = (merged / "a24-identity.json").read_bytes()
    ident = json.loads(raw)
    problems = []
    if ident.get("record_type") != "a24_finetuned_identity":
        problems.append("not an A2.4 fine-tuned identity")
    if ident.get("model_name") != candidate["model_name"]:
        problems.append(f"model {ident.get('model_name')!r}, not the candidate {candidate['model_name']!r}")
    if (ident.get("configuration") or {}).get("design") != candidate["design"]:
        problems.append(f"design {(ident.get('configuration') or {}).get('design')!r}, not {candidate['design']!r}")
    if (ident.get("selection") or {}).get("epoch") != candidate["epoch"]:
        problems.append(f"selected epoch {(ident.get('selection') or {}).get('epoch')!r}, not {candidate['epoch']}")
    files = ident.get("merged_files") or {}
    if not files:
        problems.append("the identity lists no merged files")
    for name, want in sorted(files.items()):
        p = merged / name
        want = want.get("sha256") if isinstance(want, dict) else want
        if not p.is_file():
            problems.append(f"{name} is missing")
        elif _sha_file(p) != want:
            problems.append(f"{name} changed since the merge")
    if problems:
        raise ExportError(f"{merged}: " + "; ".join(problems))
    return {"identity_sha256": hashlib.sha256(raw).hexdigest(), "model_name": ident["model_name"],
            "seed_index": ident.get("seed_index"), "selection": ident.get("selection"),
            "configuration": ident.get("configuration"), "merged_files": files}


def check_llama(llama_dir: Path) -> dict:
    release, commit = pinned_commit()
    if not (llama_dir / "convert_hf_to_gguf.py").is_file():
        raise ExportError(f"{llama_dir}: no convert_hf_to_gguf.py (run: python tools/build_llama.py fetch)")
    head = subprocess.run(["git", "-C", str(llama_dir), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip() != commit:
        raise ExportError(f"{llama_dir} is at {head.stdout.strip() or '(unknown)'}, not the pinned {release} {commit}")
    return {"release": release, "commit": commit}


def read_gguf(path: Path, llama_dir: Path) -> dict:
    """The metadata that matters, read back with llama.cpp's own gguf-py."""
    sys.path.insert(0, str(llama_dir / "gguf-py"))
    import gguf
    r = gguf.GGUFReader(str(path))

    def field(k):
        f = r.fields.get(k)
        if f is None:
            return None
        if len(f.data) == 1:
            v = f.parts[f.data[0]]
            return bytes(v).decode("utf-8") if f.types and f.types[0] == gguf.GGUFValueType.STRING else v.tolist()[0]
        return [bytes(f.parts[i]).decode("utf-8", "replace") if f.types and f.types[-1] == gguf.GGUFValueType.STRING
                else f.parts[i].tolist()[0] for i in f.data]

    tok_fields = {}
    for k in sorted(r.fields):
        if k.startswith("tokenizer.ggml."):
            tok_fields[k] = hashlib.sha256(json.dumps(field(k), sort_keys=True).encode("utf-8")).hexdigest()
    tokens = field("tokenizer.ggml.tokens")
    return {"architecture": field("general.architecture"), "file_type": field("general.file_type"),
            "n_tokens": len(tokens) if isinstance(tokens, list) else None,
            "eos_token_id": field("tokenizer.ggml.eos_token_id"),
            "tensor_types": dict(Counter(t.tensor_type.name for t in r.tensors)),
            "tokenizer_fields_sha256": tok_fields}


def export(*, merged, llama_dir, out, candidate=CANDIDATE, convert=None) -> dict:
    merged, llama_dir, out = Path(merged), Path(llama_dir), Path(out)
    ident = check_candidate(merged, candidate)
    pin = check_llama(llama_dir)
    out.mkdir(parents=True, exist_ok=True)
    gguf_path = out / f"{ident['model_name']}.{OUTTYPE}.gguf"
    if gguf_path.exists() or (out / "export.json").exists():
        raise ExportError(f"{out} already holds an export")
    tmp = out / (gguf_path.name + ".partial")
    cmd = [sys.executable, str(llama_dir / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(tmp),
           "--outtype", OUTTYPE]
    run = (convert or subprocess.run)(cmd, capture_output=True, text=True)
    (out / "convert.log").write_text((run.stdout or "") + (run.stderr or ""), encoding="utf-8")
    if run.returncode != 0 or not tmp.is_file():
        raise ExportError(f"the converter failed (exit {run.returncode}); see {out / 'convert.log'}")
    meta = read_gguf(tmp, llama_dir)
    problems = []
    if meta["architecture"] != "qwen2":
        problems.append(f"architecture {meta['architecture']!r}")
    if meta["file_type"] != FILE_TYPE_Q8_0:
        problems.append(f"file type {meta['file_type']!r}, not Q8_0 ({FILE_TYPE_Q8_0})")
    if not meta["n_tokens"]:
        problems.append("no tokenizer tokens")
    if problems:
        raise ExportError("the GGUF is not what was asked for: " + "; ".join(problems))
    tmp.replace(gguf_path)
    versions = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "numpy", "sentencepiece"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:   # noqa: BLE001
            versions[mod] = None
    rec = {"format_version": 1, "record_type": "a27_export", "candidate": dict(candidate),
           "identity": ident, "llama_cpp": pin, "command": ["python"] + cmd[1:], "outtype": OUTTYPE,
           "software": versions, "host": f"{platform.system()} {platform.machine()}",
           "gguf": {"file": gguf_path.name, "sha256": _sha_file(gguf_path), "bytes": gguf_path.stat().st_size,
                    "metadata": meta}}
    (out / "export.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return rec
