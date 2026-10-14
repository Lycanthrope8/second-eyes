"""Tests of A2.7's export, bundle, desktop run and comparison (D115).

    python grounding/tests/test_a27.py [--lib LIB --gguf TINY.gguf] [--export-llama LLAMA_DIR]

Without options, the parts that need no native library: the frozen headset subset, the candidate and identity checks,
the evaluation's binding checks, the verdicts (agreement bar, technical failures, the reproduction self-check, masks)
and the report's tables. With --lib (a host build of libse_llama with the grammar API) and --gguf (a GGUF with Qwen's
tokenizer, e.g. native/tests/make_tiny_model.py's), the whole desktop flow runs on the real library: a small bundle
whose reference outputs the Python decoder made from the same model's logits, the mask checks in worker processes,
the decodes, and the comparison's verdicts. With --export-llama (the pinned llama.cpp checkout; needs torch,
transformers and sentencepiece), the export converts a labelled random-weight merged folder with the real converter.
Stand-ins are labelled: random weights establish the tooling, never the candidate's accuracy or agreement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PASSES, FAILS = [], []
TOKENIZER = ROOT / "quest-app" / "Assets" / "SecondEyes" / "Models" / "qwen2.5-0.5b-instruct"
SIZES = {"closest": 398, "farthest": 387, "second_closest": 384, "second_farthest": 376, "between": 242,
         "third_farthest": 235, "third_closest": 234, "near": 230, "below": 107, "on": 58}
# 10 each, then 200 in proportion to n - 10: the five largest remainders (on .763, second_farthest .695,
# third_farthest .640, below .605, third_closest .562) take one more; farthest (.557) does not.
EXPECTED = {"closest": 40, "farthest": 39, "second_closest": 39, "second_farthest": 39, "between": 28,
            "third_farthest": 28, "third_closest": 28, "near": 27, "below": 18, "on": 14}

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def raises(fn, text):
    try:
        fn()
    except Exception as e:   # noqa: BLE001
        return text in str(e)
    return False


def parents_like_the_benchmark():
    out = []
    for rel, n in SIZES.items():
        out += [{"partition": "development", "stratum": rel, "parent_command_id": f"iref.dev.{rel}.{i:04d}"}
                for i in range(n)]
    out += [{"partition": "legacy-development", "stratum": rel, "parent_command_id": f"iref.legacy.{rel}.{i}"}
            for rel in ("below", "closest", "farthest", "near") for i in range(16)]
    return out


def table_widths_constant(md: str) -> bool:
    lines, i = md.split("\n"), 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
            cols = _cells(lines[i])
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                if _cells(lines[j]) != cols:
                    return False
                j += 1
            i = j
        else:
            i += 1
    return True


def _cells(line: str) -> int:
    n, k = 0, 0
    while k < len(line):
        if line[k] == "\\":
            k += 2
            continue
        n += line[k] == "|"
        k += 1
    return n


def test_subset():
    from grounding.a27 import bundle as B
    print("-- the frozen headset subset")
    ps = parents_like_the_benchmark()
    sel = B.select(ps)
    check("300 development parents, at least 10 per relation, the rest in proportion (largest remainder): "
          + ", ".join(f"{k} {v}" for k, v in EXPECTED.items()), sel["allocation"] == EXPECTED and sel["size"] == 300,
          str(sel["allocation"]))
    shuffled = ps[:]
    random.Random(3).shuffle(shuffled)
    again = B.select([dict(p, outcome="correct" if i % 2 else "wrong_object") for i, p in enumerate(shuffled)])
    check("deterministic and blind to outcomes: the same subset from shuffled parents carrying outcome fields",
          again["ids"] == sel["ids"] and again["mask_sample"] == sel["mask_sample"])
    check("only development parents: no legacy parent is chosen", all(".dev." in i for i in sel["ids"]))
    by = {}
    for i in sel["mask_sample"]:
        by.setdefault(i.split(".")[2], []).append(i)
    check("the mask sample is 2 chosen parents per relation (20), the first of each relation's ranking",
          len(sel["mask_sample"]) == 20 and all(len(v) == 2 for v in by.values()) and set(sel["mask_sample"]) <= set(sel["ids"])
          and all(sorted([p for p in sel["ids"] if p.split(".")[2] == r], key=B.rank_key)[:2] == sorted(v, key=B.rank_key)
                  for r, v in by.items()))
    small = B.allocate({"a": 4, "b": 100, "c": 200}, total=60, minimum=10)
    check("a relation smaller than the minimum is taken whole; the allocation always sums to the total",
          small["a"] == 4 and sum(small.values()) == 60, str(small))
    check("a subset larger than the parents is refused", raises(lambda: B.allocate({"a": 5}, total=6), "subset of 6"))


def merged_fixture(tmp: Path, **override) -> Path:
    m = tmp / "merged"
    if m.exists():
        shutil.rmtree(m)
    m.mkdir(parents=True)
    files = {}
    for name, body in (("config.json", b'{"model_type": "qwen2"}'), ("model.safetensors", b"weights")):
        (m / name).write_bytes(body)
        files[name] = hashlib.sha256(body).hexdigest()
    ident = {"record_type": "a24_finetuned_identity", "model_name": "a24.parser.s1", "seed_index": 1,
             "configuration": {"design": "parser", "format": None, "name": "parser"},
             "selection": {"epoch": 2, "value": 1.0}, "merged_files": files}
    ident.update(override)
    (m / "a24-identity.json").write_text(json.dumps(ident), encoding="utf-8")
    return m


def test_candidate(tmp: Path):
    from grounding.a27 import export as X
    print("-- the candidate's identity")
    ok = X.check_candidate(merged_fixture(tmp))
    check("the candidate's merged folder passes: a24.parser.s1, the parser design, epoch 2, files as recorded",
          ok["model_name"] == "a24.parser.s1" and len(ok["identity_sha256"]) == 64)
    check("another model is refused", raises(lambda: X.check_candidate(merged_fixture(tmp, model_name="a24.parser.s2")),
                                             "not the candidate"))
    check("another selected checkpoint is refused",
          raises(lambda: X.check_candidate(merged_fixture(tmp, selection={"epoch": 3})), "not 2"))
    check("the direct design is refused (parser-only scope)",
          raises(lambda: X.check_candidate(merged_fixture(tmp, configuration={"design": "direct"})), "not 'parser'"))
    m = merged_fixture(tmp)
    (m / "model.safetensors").write_bytes(b"changed")
    check("a merged file changed since the merge is refused", raises(lambda: X.check_candidate(m), "changed since"))
    (m / "model.safetensors").unlink()
    check("a missing merged file is refused", raises(lambda: X.check_candidate(m), "missing"))


def test_evaluation_binding(tmp: Path):
    from grounding.a24 import language as L
    from grounding.a24 import prep as PR
    from grounding.a27 import bundle as B
    print("-- the reference evaluation's binding")
    prep = tmp / "prep"
    prep.mkdir(exist_ok=True)
    (prep / "manifest.json").write_text('{"x": 1}', encoding="utf-8")
    m = merged_fixture(tmp)
    sha = lambda b: hashlib.sha256(b).hexdigest()   # noqa: E731

    def evaluation(**over):
        ev = tmp / "eval"
        if ev.exists():
            shutil.rmtree(ev)
        ev.mkdir()
        body = json.dumps({"parent_command_id": "p1"}).encode() + b"\n"
        (ev / "results.jsonl").write_bytes(body)
        lock = {"language_sha256": L.spec_sha256(), "prep_manifest_sha256": sha((prep / "manifest.json").read_bytes()),
                "parser_prompt_sha256": sha(json.dumps(PR.load_settings()["parser_prompt"], sort_keys=True).encode())}
        man = {"record_type": "a24_parser_evaluation", "model_name": "a24.parser.s1",
               "identity_sha256": sha((m / "a24-identity.json").read_bytes()), "lock": lock,
               "outputs": {"results.jsonl": sha(body)}}
        for k, v in over.items():
            if k in lock:
                lock[k] = v
            else:
                man[k] = v
        (ev / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        return ev
    man, rows, _ = B.check_evaluation(evaluation(), m, prep)
    check("the candidate's float32 evaluation on this preparation, language and prompt is accepted", "p1" in rows)
    check("an evaluation of another model is refused",
          raises(lambda: B.check_evaluation(evaluation(identity_sha256="0" * 64), m, prep), "another model"))
    check("another language is refused",
          raises(lambda: B.check_evaluation(evaluation(language_sha256="0" * 64), m, prep), "language changed"))
    check("another parser prompt is refused",
          raises(lambda: B.check_evaluation(evaluation(parser_prompt_sha256="0" * 64), m, prep), "prompt changed"))
    ev = evaluation()
    (ev / "results.jsonl").write_bytes(b'{"parent_command_id": "p2"}\n')
    check("results changed since the evaluation are refused", raises(lambda: B.check_evaluation(ev, m, prep),
                                                                       "changed since the evaluation"))


def rows_and_results(n, differ=(), failures=()):
    rows, results = [], []
    for i in range(n):
        ref = {"status": "completed", "valid": True, "token_ids": [10, 20, 151645], "text": "chair | near | table",
               "views": {"full_inventory": {"outcome": "correct", "target_id": "a", "technical": False,
                                            "candidate_ids": None},
                         "source_known_nyu": {"outcome": "correct", "target_id": "a", "technical": False,
                                              "candidate_ids": None}}}
        rows.append({"parent_command_id": f"p{i}", "stratum": "near", "headset": i < 3, "reference": ref})
        if i in failures:
            results.append({"parent_command_id": f"p{i}", "status": "technical_failure", "token_ids": [], "error": "x"})
        elif i in differ:
            results.append({"parent_command_id": f"p{i}", "status": "completed", "token_ids": [10, 21, 151645],
                            "text": "chair | on | table"})
        else:
            results.append({"parent_command_id": f"p{i}", "status": "completed", "token_ids": [10, 20, 151645],
                            "text": "chair | near | table"})
    return rows, results


def resolve_stub(row, d):
    o = "correct" if d["token_ids"] == [10, 20, 151645] else "insufficient_information"
    t = "a" if o == "correct" else None
    return {v: {"outcome": o, "target_id": t, "technical": False, "candidate_ids": None}
            for v in ("full_inventory", "source_known_nyu")}


def test_verdicts():
    from grounding.a27 import compare as CP
    print("-- the verdicts")
    ok_masks = {"prefixes": 10, "differing": 0}
    rows, res = rows_and_results(100, differ={5})
    v = CP.judge(rows, res, ok_masks, resolve_stub)
    check("99 identical of 100 meets the 99% bar (at least, not more than); the difference is listed with both "
          "outputs and both views' outcomes", v["criterion_7_desktop"] == "met" and v["identical"] == 99
          and v["differences"][0]["outcomes"]["full_inventory"] == {"reference": "correct", "q8_0": "insufficient_information"}
          and v["outcome_changes"] == 2)
    rows, res = rows_and_results(100, differ={5}, failures={7})
    v = CP.judge(rows, res, ok_masks, resolve_stub)
    check("a technical failure is listed apart and never counts as agreement: 98 of 100 does not meet the bar",
          v["criterion_7_desktop"] == "not met" and len(v["technical_failures"]) == 1 and v["identical"] == 98)
    v = CP.judge(*rows_and_results(4), {"prefixes": 10, "differing": 1}, resolve_stub)
    check("any differing mask fails criterion 4 (desktop)", v["criterion_4_desktop"] == "not met")
    rows, res = rows_and_results(3)
    bad = lambda row, d: {k: dict(o, target_id="b") for k, o in resolve_stub(row, d).items()}   # noqa: E731
    check("an unchanged output that does not reproduce A2.4's recorded outcomes stops the comparison",
          raises(lambda: CP.judge(rows, res, ok_masks, bad), "resolution path is wrong"))
    rows, res = rows_and_results(100, differ={1, 2}, failures={3})
    v = CP.judge(rows, res, ok_masks, resolve_stub)
    rec = {"candidate": {"model_name": "a24.parser.s1", "identity_sha256": "f" * 64}, "gguf_sha256": "e" * 64,
           "library": {"sha256": "d" * 64, "version": "b11277"}, "tokenizer_against_base": {"equal": True, "fields": 9},
           "decodes": {"statuses": {"completed": 99}, "median_ms": 400.0},
           "subset": {"size": 300, "allocation": EXPECTED, "ids_sha256": "c" * 64, "mask_sample_sha256": "b" * 64}, **v}
    md = CP.render(rec)
    check("every Markdown table in the report keeps a constant column count, with outputs' separators escaped",
          table_widths_constant(md) and "chair \\| on \\| table" in md, md[:200])


def test_desktop_native(lib, gguf, tmp: Path):
    """The whole desktop flow on the real library with a labelled random-weight model."""
    import ctypes as C
    import numpy as np
    from grounding.a24 import decoder as DC
    from grounding.a24 import language as L
    from grounding.a24 import prep as PR
    from grounding.a27 import compare as CP
    from grounding.a27 import desktop as D
    from grounding.a27 import export as X
    from grounding.a27.native import Native
    from grounding.evaluation.iref_vla.protocol import encode_json, encode_jsonl
    from grounding.preparation.iref_vla import tokens as T
    print("-- the desktop flow on the real library (labelled random-weight model)")
    spec = L.load_spec()
    tok = T.load_pinned_tokenizer(TOKENIZER)
    con = DC.Constraint(spec, DC.TokenTable(tok._backend))
    gbnf = L.gbnf(spec)
    nat = Native(lib, gguf, gbnf, con.T.eos_id, n_ctx=512, n_threads=2)
    nat.lib.se_eval.restype, nat.lib.se_eval.argtypes = C.c_int32, [C.c_void_p, C.POINTER(C.c_int32), C.c_int32, C.c_int32]
    nat.lib.se_logits.restype, nat.lib.se_logits.argtypes = C.c_int32, [C.c_void_p, C.POINTER(C.c_float), C.c_int32]
    nat.lib.se_n_cached.restype, nat.lib.se_n_cached.argtypes = C.c_int32, [C.c_void_p]
    texts = ["the chair near the table", "the lamp on the desk", "the second closest door to the sofa",
             "the box between the bed and the shelf"]
    settings = PR.load_settings()
    rows = []
    for i, t in enumerate(texts):
        ids = tok.encode(PR.parser_prompt(t, settings))

        def step(prev, ids=ids):
            if prev is None:
                arr = (C.c_int32 * len(ids))(*ids)
                assert nat.lib.se_eval(nat.m, arr, len(ids), 0) == 0
            else:
                one = (C.c_int32 * 1)(prev)
                assert nat.lib.se_eval(nat.m, one, 1, nat.lib.se_n_cached(nat.m)) == 0
            buf = (C.c_float * nat.n_vocab)()
            nat.lib.se_logits(nat.m, buf, nat.n_vocab)
            return np.frombuffer(buf, dtype=np.float32).copy()
        ref = DC.validate(DC.decode(step, con, spec["max_new_tokens"]), spec)
        rows.append({"partition": "development", "group": "g", "scene": "s", "stratum": "near",
                     "parent_command_id": f"p{i}", "prompt_ids": ids, "headset": i < 2, "mask_sample": i == 0,
                     "reference": {"status": ref["status"], "valid": ref["valid"], "token_ids": ref["token_ids"],
                                   "text": ref["text"], "views": {}}})
    nat.close()
    bundle = tmp / "bundle"
    bundle.mkdir()
    req, g = encode_jsonl(rows), gbnf.encode("utf-8")
    (bundle / "requests.jsonl").write_bytes(req)
    (bundle / "grammar.gbnf").write_bytes(g)
    sha = lambda b: hashlib.sha256(b).hexdigest()   # noqa: E731
    (bundle / "manifest.json").write_bytes(encode_json({
        "candidate": {"model_name": "a24.parser.s1", "identity_sha256": "a" * 64},
        "decoding": {"end_token": con.T.eos_id, "max_new_tokens": spec["max_new_tokens"]},
        "files": {"requests.jsonl": sha(req), "grammar.gbnf": sha(g)}}))
    export = tmp / "export"
    export.mkdir()
    shutil.copyfile(gguf, export / "tiny.gguf")
    (export / "export.json").write_text(json.dumps({"identity": {"identity_sha256": "a" * 64},
                                                    "gguf": {"file": "tiny.gguf", "sha256": X._sha_file(export / "tiny.gguf")}}),
                                        encoding="utf-8")
    llama_dir = next((p for p in (ROOT / "third_party" / "llama.cpp", Path("/home/claude/llcpp")) if (p / "gguf-py").is_dir()), None)
    if llama_dir is None:
        check("the desktop flow needs llama.cpp's gguf-py (third_party/llama.cpp)", False)
        return
    frozen = json.loads(D.MASK_EXPECTATIONS.read_text(encoding="utf-8"))["masks"]
    out = tmp / "desktop"
    man = D.run(bundle=bundle, export=export, base_gguf=gguf, lib=lib, llama_dir=llama_dir, out=out, workers=2,
                threads=2, frozen_prefixes=[tok.encode(p) for p in frozen], progress=lambda *_: None)
    check("the export's tokenizer equals the base GGUF's, field by field", man["tokenizer_against_base"]["equal"])
    check(f"masks: at all {man['masks']['prefixes']} prefixes (the references' and the 21 frozen), in 2 worker "
          "processes, the native allowed tokens equal the Python decoder's", man["masks"]["differing"] == 0
          and man["masks"]["prefixes"] >= 21, str(man["masks"]))
    res = [json.loads(x) for x in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    same = all(r["token_ids"] == row["reference"]["token_ids"] and r["status"] == row["reference"]["status"]
               for r, row in zip(res, rows))
    check("every desktop decode equals the Python decoder's on the same model, tokens and status", same and len(res) == 4,
          str([(r["status"], len(r["token_ids"])) for r in res]))
    v = CP.judge(rows, [dict(r, text="") for r in res], json.loads((out / "masks.json").read_text(encoding="utf-8")),
                 lambda row, d: {})
    check("and the comparison's verdicts follow: criteria 4 and 7 (desktop) met, no technical failure",
          v["criterion_4_desktop"] == "met" and v["criterion_7_desktop"] == "met" and not v["technical_failures"])
    again = D.run(bundle=bundle, export=export, base_gguf=gguf, lib=lib, llama_dir=llama_dir, out=out,
                  progress=lambda *_: None)
    check("a finished run is returned as it stands, never redone", again["outputs"] == man["outputs"])
    (export / "tiny.gguf").write_bytes(b"other")
    check("a GGUF other than the exported one is refused",
          raises(lambda: D.run(bundle=bundle, export=export, base_gguf=gguf, lib=lib, llama_dir=llama_dir,
                               out=tmp / "d2", progress=lambda *_: None), "not the exported one"))


def test_export_real(llama_dir, tmp: Path):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from grounding.a27 import export as X
    print("-- the export with the pinned converter (labelled random-weight merged folder)")
    m = tmp / "merged-real"
    cfg = Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(str(m), safe_serialization=True)
    for n in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        shutil.copyfile(TOKENIZER / n, m / n)
    files = {p.name: X._sha_file(p) for p in m.iterdir()}
    (m / "a24-identity.json").write_text(json.dumps({
        "record_type": "a24_finetuned_identity", "model_name": "a24.parser.s1", "configuration": {"design": "parser"},
        "selection": {"epoch": 2}, "merged_files": files}), encoding="utf-8")
    rec = X.export(merged=m, llama_dir=llama_dir, out=tmp / "export-real")
    meta = rec["gguf"]["metadata"]
    check("the pinned converter writes a Q8_0 qwen2 GGUF with the full vocabulary, read back by llama.cpp's gguf-py",
          meta["architecture"] == "qwen2" and meta["file_type"] == 7 and meta["n_tokens"] == 151936
          and "Q8_0" in meta["tensor_types"], str(meta)[:300])
    check("export.json records the identity, the pinned commit, the command and the GGUF's hash",
          rec["llama_cpp"]["commit"] == X.pinned_commit()[1] and rec["identity"]["model_name"] == "a24.parser.s1"
          and len(rec["gguf"]["sha256"]) == 64 and rec["command"][1].endswith("convert_hf_to_gguf.py"))
    check("a second export into the same folder is refused",
          raises(lambda: X.export(merged=m, llama_dir=llama_dir, out=tmp / "export-real"), "already holds"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib")
    ap.add_argument("--gguf")
    ap.add_argument("--export-llama")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_subset()
        test_candidate(tmp)
        test_evaluation_binding(tmp)
        test_verdicts()
        if a.lib and a.gguf:
            test_desktop_native(a.lib, a.gguf, tmp)
        if a.export_llama:
            test_export_real(Path(a.export_llama), tmp)
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
