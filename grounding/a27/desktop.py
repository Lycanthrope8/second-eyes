"""A2.7 desktop run (D115): the exported candidate through libse_llama's constrained decoding on this PC (the host build).

    python -m grounding.a27 desktop --bundle BUNDLE --export EXPORT --base-gguf BASE --lib LIB --llama LLAMA_DIR
                                    --out FOLDER [--workers 4] [--threads 8] [--resume]

EXPORT is the export folder (export.json and the GGUF); BASE the accepted A2.5 base-model GGUF, whose tokenizer the
export must carry unchanged; LIB the host build of libse_llama with the grammar API. Three parts:
1. checks: the bundle's files, the GGUF's hash against export.json, the candidate's identity against the bundle's, and
   every tokenizer field of the GGUF equal to the base GGUF's;
2. masks (criterion 4, desktop): at every distinct prefix of the 2,715 reference decodes and at the 21 frozen prefixes,
   the tokens llama.cpp's grammar allows equal the Python decoder's. One full-vocabulary application per prefix, so they
   run in --workers processes, each with its own model and grammar;
3. decodes (criterion 7's desktop half): every parent's prompt decoded once, a fresh cache each, written as it goes;
   --resume continues an interrupted run under the same inputs.
Writes masks.json, results.jsonl and manifest.json. Judging is the comparison's job, on the PC with the scenes.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

from ..evaluation.iref_vla.protocol import encode_json, encode_jsonl
from . import export as X
from .native import Native

POLICY = "a27.desktop.v1"
MASK_EXPECTATIONS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "a24" / "mask-expectations.v1.json"
N_CTX = 1024


class DesktopError(RuntimeError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _jl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_bundle(bundle: Path) -> tuple:
    man = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for name, want in man["files"].items():
        if _sha((bundle / name).read_bytes()) != want:
            raise DesktopError(f"{bundle / name} changed since the bundle was built")
    return man, _jl(bundle / "requests.jsonl"), (bundle / "grammar.gbnf").read_text(encoding="utf-8")


def tokenizer_equal(gguf_path: Path, base_path: Path, llama_dir: Path) -> dict:
    a = X.read_gguf(gguf_path, llama_dir)["tokenizer_fields_sha256"]
    b = X.read_gguf(base_path, llama_dir)["tokenizer_fields_sha256"]
    differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {"equal": not differ, "fields": len(a), "differing": differ}


def prefixes(rows, frozen_token_prefixes) -> list:
    """Every distinct state a reference decode passed through (the prefix before each chosen token), and the frozen
    prefixes, in a stable order."""
    seen = {tuple(p) for p in frozen_token_prefixes}
    for r in rows:
        t = r["reference"]["token_ids"]
        for k in range(len(t)):
            seen.add(tuple(t[:k]))
    return sorted(seen, key=lambda p: (len(p), p))


# ------------------------------------------------------------------------------------------- mask workers

_W = {}


def _worker_init(lib, gguf, gbnf, end_token):
    from ..a24 import decoder as DC
    from ..a24 import language as L
    from ..a24 import prep as PR
    from ..preparation.iref_vla import tokens as T
    _W["native"] = Native(lib, gguf, gbnf, end_token, n_ctx=64, n_threads=1)
    _W["con"] = DC.Constraint(L.load_spec(), DC.TokenTable(T.load_pinned_tokenizer(PR.DEV.TOKENIZER)._backend))


def _python_allowed(con, prefix):
    state = con.A.start
    for t in prefix:
        nxt = con.allowed(state)
        if t not in nxt:
            return None
        state = nxt[t]
    return set(con.allowed(state))


def _check_chunk(chunk):
    nat, con = _W["native"], _W["con"]
    out = []
    for prefix in chunk:
        mine, theirs = nat.allowed_after(prefix), _python_allowed(con, prefix)
        if mine != theirs:
            out.append({"prefix": list(prefix), "native": None if mine is None else len(mine),
                        "python": None if theirs is None else len(theirs),
                        "only_native": sorted((mine or set()) - (theirs or set()))[:10],
                        "only_python": sorted((theirs or set()) - (mine or set()))[:10]})
    return len(chunk), out


def check_masks(lib, gguf, gbnf, end_token, todo, workers, progress=print) -> dict:
    t0 = time.perf_counter()
    chunks = [todo[i:i + 50] for i in range(0, len(todo), 50)]
    mism, done = [], 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init, initargs=(str(lib), str(gguf), gbnf, end_token)) as pool:
        for n, bad in pool.imap_unordered(_check_chunk, chunks):
            done += n
            mism.extend(bad)
            if done % 500 < 50:
                progress(f"  masks: {done}/{len(todo)} prefixes, {len(mism)} differing")
    return {"prefixes": len(todo), "differing": len(mism), "mismatches": mism[:50],
            "seconds": round(time.perf_counter() - t0, 1), "workers": workers}


def run(*, bundle, export, base_gguf, lib, llama_dir, out, workers=4, threads=8, resume=False, frozen_prefixes=None,
        progress=print) -> dict:
    bundle, export, out = Path(bundle), Path(export), Path(out)
    man, rows, gbnf = load_bundle(bundle)
    xr = json.loads((export / "export.json").read_text(encoding="utf-8"))
    gguf = export / xr["gguf"]["file"]
    problems = []
    if X._sha_file(gguf) != xr["gguf"]["sha256"]:
        problems.append("the GGUF is not the exported one")
    if xr["identity"]["identity_sha256"] != man["candidate"]["identity_sha256"]:
        problems.append("the export and the bundle name different models")
    if problems:
        raise DesktopError("; ".join(problems))
    lib_sha = X._sha_file(Path(lib))
    bman = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    lock = {"record_type": "a27_desktop_lock", "bundle_manifest_sha256": _sha((bundle / "manifest.json").read_bytes()),
            "gguf_sha256": xr["gguf"]["sha256"], "lib_sha256": lib_sha, "policy": POLICY}
    if out.exists() and (out / "manifest.json").is_file():
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if out.exists():
        if not resume:
            raise DesktopError(f"{out} exists; pass --resume to continue it")
        if json.loads((out / "lock.json").read_text(encoding="utf-8")) != lock:
            raise DesktopError("the lock differs (bundle, GGUF or library): refusing to continue")
    else:
        out.mkdir(parents=True)
        (out / "lock.json").write_bytes(encode_json(lock))
    dec = bman["decoding"]
    progress("checks: the tokenizer against the accepted base GGUF")
    tok_eq = tokenizer_equal(gguf, Path(base_gguf), Path(llama_dir))
    if not tok_eq["equal"]:
        raise DesktopError(f"the export's tokenizer differs from the base GGUF's: {tok_eq['differing']}")
    if not (out / "masks.json").is_file():
        if frozen_prefixes is None:
            from ..a24 import prep as PR
            from ..preparation.iref_vla import tokens as T
            tok = T.load_pinned_tokenizer(PR.DEV.TOKENIZER)
            frozen = json.loads(MASK_EXPECTATIONS.read_text(encoding="utf-8"))["masks"]
            frozen_prefixes = [tok.encode(p) for p in frozen]
        todo = prefixes(rows, frozen_prefixes)
        progress(f"masks: {len(todo)} prefixes in {workers} processes")
        masks = check_masks(lib, gguf, gbnf, dec["end_token"], todo, workers, progress)
        (out / "masks.json").write_bytes(encode_json(masks))
    masks = json.loads((out / "masks.json").read_text(encoding="utf-8"))
    partial = out / "results.partial.jsonl"
    done = {r["parent_command_id"] for r in _jl(partial)} if partial.is_file() else set()
    nat = Native(lib, gguf, gbnf, dec["end_token"], n_ctx=N_CTX, n_threads=threads)
    t0 = time.perf_counter()
    try:
        with open(partial, "a", encoding="utf-8") as f:
            for k, r in enumerate(rows):
                if r["parent_command_id"] in done:
                    continue
                t1 = time.perf_counter()
                res = nat.decode(r["prompt_ids"], dec["max_new_tokens"])
                f.write(json.dumps({"parent_command_id": r["parent_command_id"], **res,
                                    "ms": round((time.perf_counter() - t1) * 1000, 1)}, sort_keys=True) + "\n")
                f.flush()
                if (k + 1) % 250 == 0:
                    progress(f"  decodes: {k + 1}/{len(rows)}")
        version = nat.version()
    finally:
        nat.close()
    got = {r["parent_command_id"]: r for r in _jl(partial)}
    ordered = [got[r["parent_command_id"]] for r in rows]
    body = encode_jsonl(ordered)
    (out / "results.jsonl").write_bytes(body)
    build = Path(lib).with_name("build.json")
    manifest = {"format_version": 1, "record_type": "a27_desktop_run", "policy": POLICY, "lock": lock,
                "export": {"identity_sha256": xr["identity"]["identity_sha256"], "gguf_sha256": xr["gguf"]["sha256"]},
                "library": {"sha256": lib_sha, "version": version,
                            "build": json.loads(build.read_text(encoding="utf-8")) if build.is_file() else None},
                "tokenizer_against_base": tok_eq,
                "masks": {k: masks[k] for k in ("prefixes", "differing", "seconds", "workers")},
                "decodes": {"parents": len(ordered), "statuses": {s: sum(1 for x in ordered if x["status"] == s)
                                                                 for s in sorted({x["status"] for x in ordered})},
                            "seconds_this_session": round(time.perf_counter() - t0, 1), "threads": threads,
                            "median_ms": sorted(x["ms"] for x in ordered)[len(ordered) // 2] if ordered else None},
                "host": {"system": f"{platform.system()} {platform.release()} {platform.machine()}",
                         "processor": platform.processor(), "cpus": os.cpu_count()},
                "outputs": {"results.jsonl": _sha(body), "masks.json": _sha((out / "masks.json").read_bytes())},
                "note": "desktop runs only; judged by the comparison (criteria 4 and 7, desktop halves)"}
    (out / "manifest.json").write_bytes(encode_json(manifest))
    return manifest
