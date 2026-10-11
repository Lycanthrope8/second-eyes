"""A2.6c desktop Q8_0/U comparison: the frozen 64 requests on the desktop native build, against the new 0.5B float32 rows.

**`export`** (on the RTX PC, which holds the full request file): the 64 request IDs that `desktop.json` froze, after the
request file's frozen hash is checked. Each record carries A2.5's replay input fields:

- the prompt, rebuilt from its document and D104 mapping with the pinned protocol (base64, byte count, SHA-256);
- the codes, targets and code token IDs, with the mapping hash;
- the token IDs, with their count and hash;
- both boundaries.

The constants come from the pinned protocol and the 0.5B model description (the vocabulary size).

**`run`** (on the machine with the host build): A2.5's accepted desktop pieces, unchanged:

- its self-checks;
- the model loaded with stderr captured, at the requested settings (8,192 tokens of context, 2 threads, 16 sequences,
  flags 0);
- the accepted GGUF, checked by its full SHA-256;
- per request, the input check with the native tokenizer;
- **path U only:** the prior state cleared, and the complete prompt evaluated from position 0 with nothing kept;
- the offered logits, log-probabilities, restricted shares and decision.

The host's identity (platform, library hash, build record, runtime and memory) is recorded.

**`compare`:** each desktop row beside the matching new 0.5B float32 row of the RTX PC run. It gives the agreement of
decisions, and the largest differences of offered logits and of restricted shares.

It is a desktop comparison: it establishes no Quest agreement and does not close A2.5's unresolved numerical acceptance.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ...evaluation.iref_vla.protocol import EvaluationInputError, encode_json, encode_jsonl, issue
from ...quest import replay_desktop as RD
from ..iref_vla.choices import build_prompt
from ..iref_vla.protocol import load_protocol
from . import preflight as PF

MODEL_DESCRIPTION = Path(__file__).resolve().parents[3] / "grounding" / "models" / "qwen2.5-0.5b-instruct.json"
LABEL = "desktop Q8_0/U comparison: not Quest agreement; A2.5's numerical acceptance stays unresolved"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fail(where, message):
    raise EvaluationInputError([issue(str(where), "E_A26C_DESKTOP", message)])


def constants(proto=None) -> dict:
    proto = proto or load_protocol()
    vocab = json.loads(MODEL_DESCRIPTION.read_text(encoding="utf-8"))["architecture"]["vocab_size"]
    return {"context_limit_tokens": proto["context_limit_tokens"], "continuation_tokens": proto["continuation_tokens"],
            "vocab_size": vocab, "object_codes": list(proto["object_codes"]), "ask_code": proto["ask_code"],
            "ask_target": proto["ask_target"], "code_token_ids": dict(proto["code_token_ids"])}


def export(*, prep, out, frozen=PF.FROZEN) -> dict:
    prep, out = Path(prep), Path(out)
    if out.exists():
        _fail(out, "the export folder exists")
    req_bytes = (prep / "requests.jsonl").read_bytes()
    if _sha(req_bytes) != frozen["requests_sha256"]:
        _fail(prep, "the request file is not the frozen one")
    ids = json.loads((prep / "desktop.json").read_text(encoding="utf-8"))["request_ids"]
    proto = load_protocol()
    by = {}
    for line in req_bytes.decode("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["request_id"] in set(ids):
                by[r["request_id"]] = r
    if sorted(by) != sorted(ids):
        _fail(prep, "a frozen desktop request is missing from the request file")
    recs = []
    for rid in ids:
        r = by[rid]
        mapping = list(zip(r["codes"], r["choice_object_ids"]))
        pb = build_prompt(proto, mapping, r["document"]).encode("utf-8")
        if _sha(pb) != r["prompt_sha256"]:
            _fail(rid, "the rebuilt prompt does not hash to the recorded prompt")
        recs.append({"request_id": rid, "parent_command_id": r["parent_command_id"], "view": r["view"], "format": r["format"],
                     "prompt_b64": base64.b64encode(pb).decode("ascii"), "prompt_bytes": len(pb), "prompt_sha256": r["prompt_sha256"],
                     "codes": r["codes"], "targets": r["choice_object_ids"], "code_token_ids": r["code_token_ids"],
                     "mapping_sha256": r["mapping_sha256"], "token_ids": r["token_ids"], "input_tokens": r["input_tokens"],
                     "token_ids_sha256": r["token_ids_sha256"], "keep_before_last_token": r["keep_before_last_token"],
                     "keep_before_command_line": r["keep_before_command_line"]})
    body = encode_jsonl(recs)
    meta = {"format_version": 1, "record_type": "a26c_desktop_export", "run_id": PF.RUN_ID, "frozen": dict(frozen),
            "desktop_json_sha256": _sha((prep / "desktop.json").read_bytes()), "constants": constants(proto),
            "requests": len(recs), "records_sha256": _sha(body), "label": LABEL}
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent if out.parent.exists() else Path("."))))
    try:
        (tmp / "desktop-requests.jsonl").write_bytes(body)
        (tmp / "desktop-export.json").write_bytes(encode_json(meta))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return meta


def run(*, export_dir, out, model=None, library=None, native=None, accepted_sha256=RD.ACCEPTED_MODEL_SHA256,
        progress=print) -> dict:
    """Path U on the desktop build, for the exported records only."""
    ex, out = Path(export_dir), Path(out)
    if out.exists():
        _fail(out, "the output folder exists")
    meta = json.loads((ex / "desktop-export.json").read_text(encoding="utf-8"))
    body = (ex / "desktop-requests.jsonl").read_bytes()
    if _sha(body) != meta["records_sha256"]:
        _fail(ex, "the exported records differ from their recorded hash")
    recs = [json.loads(x) for x in body.decode("utf-8").splitlines() if x.strip()]
    cons = meta["constants"]
    model = Path(model) if model is not None else RD.DEFAULT_MODEL
    if not model.is_file() or RD.file_sha256(model) != accepted_sha256:
        _fail(model, "not the accepted model file (full SHA-256 dd753cd6...)")
    library = Path(library) if library is not None else RD.DEFAULT_LIBRARY
    if native is None:
        if not library.is_file():
            _fail(library, "no host build: run python tools/build_llama.py host")
        native = RD.Native(library)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        checks = RD.self_checks()
        if not all(c["passed"] for c in checks):
            raise RuntimeError("an A2.5 self-check failed")
        progress("loading the model (8,192 tokens of context, 2 threads)")
        before = (native.memory_kb(False), native.memory_kb(True))
        ok, load_ms, cap_error = RD.captured_load(native, model, staging / "startup-log.txt")
        if not ok:
            raise RuntimeError("the model did not load: " + native.last_error())
        info = native.info()
        if info[1] != cons["vocab_size"]:
            raise RuntimeError(f"the loaded vocabulary has {info[1]} entries, not {cons['vocab_size']}")
        after = (native.memory_kb(False), native.memory_kb(True))
        rows = []
        for k, rec in enumerate(recs, 1):
            chk, line = RD._input_line(rec, cons, native)
            row = {"request_id": rec["request_id"], "parent_command_id": rec["parent_command_id"], "view": rec["view"],
                   "format": rec["format"], "input_check": line, "path": "U"}
            if chk["outcome"] == "passed":
                n = len(rec["token_ids"])
                okk, ev = RD._eval(native, rec["token_ids"], 0, n, 0, n)
                row["U"] = dict({"evals": [ev]}, **(RD._scored(native, rec, cons["ask_code"], info[1]) if okk
                                                    else {"error": "eval_failed_or_cache_mismatch"}))
            else:
                row["U"] = {"error": "input_check_failed"}
            rows.append(row)
            progress(f"  {k}/{len(recs)} {rec['request_id']}: {row['U'].get('choice_code') or row['U'].get('error')}")
        build = RD.HOST_BUILD_RECORD.read_bytes() if RD.HOST_BUILD_RECORD.is_file() else None
        ident = {"format_version": 1, "record_type": "a26c_desktop_identity", "run_id": PF.RUN_ID, "label": LABEL,
                 "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "export": meta,
                 "model": {"path": str(model), "bytes": model.stat().st_size, "sha256": accepted_sha256},
                 "requested": dict(RD.REQUESTED),
                 "runtime": {"n_ctx": info[0], "n_vocab": info[1], "n_seq": info[2], "flags": info[3], "version": native.version(),
                             "system_info": native.system_info(), "load_ms": load_ms, "capture_error": cap_error},
                 "memory_kb": {"before_load": before, "after_load": after},
                 "host": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                          "cpu_count": os.cpu_count(), "python": sys.version.split()[0], "library": str(library),
                          "library_sha256": RD.file_sha256(library) if library.is_file() else None,
                          "build_record_sha256": _sha(build) if build else None},
                 "self_checks": checks,
                 "code": {p.name: _sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}}
        done = {"format_version": 1, "record_type": "a26c_desktop_done", "requests": len(recs),
                "completed": sum(1 for r in rows if "choice_code" in r["U"] and r["U"].get("error") is None),
                "failed": [r["request_id"] for r in rows if r["U"].get("error")], "label": LABEL}
        (staging / "identity.json").write_bytes(encode_json(ident))
        (staging / "results.jsonl").write_bytes(encode_jsonl(rows))
        (staging / "done.json").write_bytes(encode_json(done))
        staging.rename(out)
    except BaseException:
        failed = out.parent / f"{out.name}.failed-{time.strftime('%Y%m%d-%H%M%S')}"
        if staging.exists():
            staging.rename(failed)   # kept: failed or partial execution evidence is never deleted
        raise
    finally:
        try:
            native.free()
        except Exception:   # noqa: BLE001
            pass
    return done


def compare(*, desktop, run05, out) -> dict:
    """The desktop rows beside the matching new 0.5B float32 rows: decisions and offered-score differences."""
    out = Path(out)
    if out.exists():
        _fail(out, "the output folder exists")
    drows = [json.loads(x) for x in (Path(desktop) / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    frows = {r["request_id"]: r for r in (json.loads(x) for x in (Path(run05) / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip())}
    pairs, missing = [], []
    for d in drows:
        f = frows.get(d["request_id"])
        u = d["U"]
        if f is None or f["technical_status"] != "completed" or u.get("error"):
            missing.append(d["request_id"])
            continue
        fl = {s["code"]: s["logit"] for s in f["scores"]}
        fs = {s["code"]: s["restricted_share"] for s in f["scores"]}
        dl = {o["code"]: o["logit"] for o in u["offered"]}
        ds = {o["code"]: o["restricted_share"] for o in u["offered"]}
        pairs.append({"request_id": d["request_id"], "view": d["view"], "format": d["format"],
                      "float32_choice": f["choice_code"], "desktop_choice": u["choice_code"],
                      "same_choice": f["choice_code"] == u["choice_code"],
                      "max_abs_logit_difference": max(abs(fl[c] - dl[c]) for c in fl),
                      "max_abs_share_difference": max(abs(fs[c] - ds[c]) for c in fs)})
    same = sum(p["same_choice"] for p in pairs)
    summary = {"format_version": 1, "record_type": "a26c_desktop_comparison", "run_id": PF.RUN_ID, "label": LABEL,
               "requests": len(drows), "compared": len(pairs), "not_compared": missing, "same_choice": same,
               "max_abs_logit_difference": max((p["max_abs_logit_difference"] for p in pairs), default=None),
               "max_abs_share_difference": max((p["max_abs_share_difference"] for p in pairs), default=None),
               "inputs": {"desktop_results_sha256": _sha((Path(desktop) / "results.jsonl").read_bytes()),
                          "float32_results_sha256": _sha((Path(run05) / "results.jsonl").read_bytes())}}
    out.mkdir(parents=True)
    (out / "pairs.jsonl").write_bytes(encode_jsonl(pairs))
    (out / "summary.json").write_bytes(encode_json(summary))
    return summary
