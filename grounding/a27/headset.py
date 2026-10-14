"""A2.7 delivery 2 (D115): the headset's parser folder, built on the RTX PC, and the judge of its headset results.

    python -m grounding.a27 headset --bundle BUNDLE --desktop DESKTOP --export EXPORT --prep PREP --work WORK
                                    --maps-from DIR [--maps-from DIR ...] --out FOLDER
    python -m grounding.a27 headset-judge --folder FOLDER --results RESULTS --out JUDGED

The folder, pushed to the headset like A2.5's goldens (the manifest last), holds:
- assets/: the eight schemas, relations.v1.json, directions.v1.json, resolver-protocol.json, language.v1.json,
  settings.v1.json, grammar.gbnf and parser.json (end token, max new tokens, context, the GGUF's name and SHA-256);
- maps/<map_id>.json: every category map found under the --maps-from folders (A2.2d's bundle for the interactive
  snapshots, A2.6's work for the acceptance scenes); one ID with two different maps is refused;
- startup-check.json: parser prompts for fixed command texts with their pinned token IDs; a parser session refuses to
  start if the native tokenization differs (the parser's counterpart of A2.5's startup guard, D104);
- acceptance/: the headset halves of criteria 1, 2, 4 and 7 (D115) for Delivery 3: decodes.jsonl (the frozen
  300-command subset: prompt text and IDs, the desktop Q8_0 tokens and status, equal to float32's), masks.jsonl (the 21
  frozen prefixes and every step of the 20-command mask sample's reference decodes: count and SHA-256 of the allowed
  IDs, from the Python decoder), resolutions.json (the subset in both views: interpretation, records by index, A2.4's
  recorded outcome) and readers.jsonl (every distinct reference output of the whole bundle and the 97-string corpus);
  every case carries its expected grounding query (D117), the readers' in the context reader-context.json records;
- manifest.json: every file's SHA-256 and size. Its own SHA-256 identifies the folder in every logged outcome.
The judge (D117) first verifies the evidence: the folder file by file against its manifest, with every count; the results
naming this folder and the exported candidate's model, after a passed startup check; one result per expected case, none
missing, none duplicated, none extra. Only then does it compare, criterion by criterion. A prompt whose native tokens
differ fails criterion 7. `headset-amend` applies Delivery 2's corrections to an existing folder without A2.6's slow load.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from ..a24 import evaluate as E
from ..a24 import gbnf_check as GC
from ..a24 import language as L
from ..a24 import prep as PR
from ..evaluation.iref_vla.protocol import encode_json, encode_jsonl
from . import goldens as GD

REPO = Path(__file__).resolve().parents[2]
SCHEMAS = GD.SCHEMA_FILES
STARTUP_TEXTS = ("the chair near the table", "the second closest lamp to the bed", "the box between the sofa and the door",
                 "a white mug on the desk")
CONTEXT_TOKENS = 1024


class HeadsetError(RuntimeError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _jl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def collect_maps(folders) -> dict:
    maps = {}
    for folder in folders:
        for p in sorted(Path(folder).rglob("category-map*.json")):
            body = p.read_bytes()
            try:
                d = json.loads(body)
            except ValueError:
                continue
            if not isinstance(d, dict) or d.get("record_type") != "category_map":
                continue
            mid = d["map_id"]
            if mid in maps and maps[mid] != body and json.loads(maps[mid]) != d:
                raise HeadsetError(f"category map {mid} has two different contents ({p})")
            maps.setdefault(mid, body)
    return maps


def build(*, bundle, desktop, export, prep, work, maps_from, out, tokenizer=None, port_out=None) -> dict:
    """port_out: also write the full acceptance export there (every parent in both views, for the PC's C# check),
    sharing the one slow load of A2.4's parent list."""
    from ..a24 import decoder as DC
    from ..preparation.iref_vla import tokens as T
    from . import acceptance as AC
    bundle, desktop, export, work, out = Path(bundle), Path(desktop), Path(export), Path(work), Path(out)
    if out.exists():
        raise HeadsetError(f"{out} already exists")
    bman = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    dman = json.loads((desktop / "manifest.json").read_text(encoding="utf-8"))
    xr = json.loads((export / "export.json").read_text(encoding="utf-8"))
    if dman["lock"]["bundle_manifest_sha256"] != _sha((bundle / "manifest.json").read_bytes()):
        raise HeadsetError("the desktop run used another bundle")
    if dman["outputs"]["results.jsonl"] != _sha((desktop / "results.jsonl").read_bytes()):
        raise HeadsetError("the desktop results changed since the run")
    if xr["gguf"]["sha256"] != dman["export"]["gguf_sha256"]:
        raise HeadsetError("the export and the desktop run name different GGUFs")
    spec, settings = L.load_spec(), PR.load_settings()
    tok = tokenizer or T.load_pinned_tokenizer(PR.DEV.TOKENIZER)
    con = DC.Constraint(spec, DC.TokenTable(tok._backend))
    rows = _jl(bundle / "requests.jsonl")
    desk = {r["parent_command_id"]: r for r in _jl(desktop / "results.jsonl")}
    parents = {p["parent_command_id"]: p for p in E._parents(prep, work)}   # verifies every A2.6 bundle: slow, once
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    files = {}

    def put(rel, data: bytes):
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files[rel] = {"sha256": _sha(data), "bytes": len(data)}
    for name in SCHEMAS:
        put(f"assets/schemas/{name}", (REPO / "schemas" / name).read_bytes())
    rel_dir = REPO / "grounding" / "relations"
    put("assets/relations.v1.json", (rel_dir / "relations.v1.json").read_bytes())
    put("assets/directions.v1.json", (rel_dir / "directions.v1.json").read_bytes())
    put("assets/resolver-protocol.json", encode_json(GD.resolver_protocol()))
    put("assets/language.v1.json", L.SPEC_PATH.read_bytes())
    put("assets/settings.v1.json", PR.SETTINGS_PATH.read_bytes())
    put("assets/grammar.gbnf", L.gbnf(spec).encode("utf-8"))
    put("assets/parser.json", encode_json({"format_version": 1, "record_type": "a27_parser_config", "end_token": con.T.eos_id,
                                           "max_new_tokens": spec["max_new_tokens"], "context_tokens": CONTEXT_TOKENS,
                                           "gguf_file": xr["gguf"]["file"], "gguf_sha256": xr["gguf"]["sha256"],
                                           "candidate": xr["identity"]["model_name"]}))
    maps = collect_maps(maps_from)
    for mid, body in sorted(maps.items()):
        put(f"maps/{mid}.json", body)
    checks = []
    for text in STARTUP_TEXTS:
        prompt = PR.parser_prompt(text, settings)
        checks.append({"text": text, "prompt_sha256": _sha(prompt.encode("utf-8")), "ids": tok.encode(prompt)})
    put("startup-check.json", encode_json({"format_version": 1, "record_type": "a27_parser_startup_check", "prompts": checks}))
    decodes = []
    for r in rows:
        if not r["headset"]:
            continue
        d = desk[r["parent_command_id"]]
        prompt = PR.parser_prompt(parents[r["parent_command_id"]]["text"], settings)
        if tok.encode(prompt) != r["prompt_ids"]:
            raise HeadsetError(f"{r['parent_command_id']}: the rebuilt prompt differs from the bundle's tokens")
        decodes.append({"parent_command_id": r["parent_command_id"], "stratum": r["stratum"], "prompt": prompt,
                        "prompt_ids": r["prompt_ids"], "tokens": d["token_ids"], "status": d["status"],
                        "reference_tokens": r["reference"]["token_ids"]})
    put("acceptance/decodes.jsonl", encode_jsonl(decodes))
    sample = [r for r in rows if r["mask_sample"]]
    frozen = json.loads((REPO / "grounding" / "tests" / "fixtures" / "a24" / "mask-expectations.v1.json").read_text())["masks"]
    prefixes = {tuple(tok.encode(p)) for p in frozen}
    for r in sample:
        t = r["reference"]["token_ids"]
        for k in range(len(t)):
            prefixes.add(tuple(t[:k]))
    masks = []
    for prefix in sorted(prefixes, key=lambda p: (len(p), p)):
        state = con.A.start
        for tk in prefix:
            state = con.allowed(state)[tk]
        allowed = sorted(con.allowed(state))
        masks.append({"prefix": list(prefix), "count": len(allowed),
                      "ids_sha256": _sha(",".join(map(str, allowed)).encode("ascii"))})
    put("acceptance/masks.jsonl", encode_jsonl(masks))
    port = AC.build_body(bundle=bundle, prep=prep, work=work, only_headset=True, parents=parents)
    _put_acceptance(put, port)
    manifest = {"format_version": 1, "record_type": "a27_parser_folder", "candidate": xr["identity"]["model_name"],
                "gguf_sha256": xr["gguf"]["sha256"], "bundle_manifest_sha256": _sha((bundle / "manifest.json").read_bytes()),
                "desktop_manifest_sha256": _sha((desktop / "manifest.json").read_bytes()), "files": files,
                "counts": {"maps": len(maps), "decodes": len(decodes), "masks": len(masks),
                           "resolutions": len(port["resolutions"]), "readers": len(port["readers"])}}
    mbody = encode_json(manifest)
    (tmp / "manifest.json").write_bytes(mbody)
    tmp.rename(out)
    summary = {"manifest_sha256": _sha(mbody), **manifest["counts"], "bytes": sum(f["bytes"] for f in files.values())}
    if port_out is not None:
        port_out = Path(port_out)
        full = AC.build_body(bundle=bundle, prep=prep, work=work, parents=parents)
        port_out.mkdir(parents=True, exist_ok=False)
        data = encode_json(full)
        (port_out / "port-acceptance.json").write_bytes(data)
        summary["port"] = {"resolutions": len(full["resolutions"]), "readers": len(full["readers"]), "records": len(full["records"]),
                           "sha256": _sha(data), "bytes": len(data)}
        (port_out / "summary.json").write_bytes(encode_json(summary["port"]))
    return summary


def _put_acceptance(put, body):
    """The resolutions (with their queries), the full reader cases and the readers' context, as the headset reads them."""
    put("acceptance/resolutions.json", encode_json({k: body[k] for k in ("format_version", "record_type", "bundle_manifest_sha256",
                                                                          "protocol", "records", "resolutions")}))
    put("acceptance/readers.jsonl", encode_jsonl(body["readers"]))
    ctx = body["reader_context"]
    put("acceptance/reader-context.json", encode_json({"format_version": 1, "record_type": "a27_reader_context",
                                                       "scene": body["records"][ctx["scene"]], "command": body["records"][ctx["command"]]}))


def verify_folder(folder: Path) -> dict:
    """Every file the manifest lists, by SHA-256 and size, and the counts against the files; the manifest, or HeadsetError."""
    folder = Path(folder)
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if man.get("record_type") != "a27_parser_folder":
        raise HeadsetError(f"{folder} is not an A2.7 parser folder")
    for name, f in man["files"].items():
        q = folder / name
        if not q.is_file() or q.stat().st_size != f["bytes"] or _sha(q.read_bytes()) != f["sha256"]:
            raise HeadsetError(f"{name} differs from its manifest")
    acc = folder / "acceptance"
    got = {"decodes": len(_jl(acc / "decodes.jsonl")), "masks": len(_jl(acc / "masks.jsonl")),
           "resolutions": len(json.loads((acc / "resolutions.json").read_text(encoding="utf-8"))["resolutions"]),
           "readers": len(_jl(acc / "readers.jsonl"))}
    for k, n in got.items():
        if man["counts"].get(k) != n or n == 0:
            raise HeadsetError(f"the folder's {k}: {n} cases, the manifest says {man['counts'].get(k)}")
    return man


def amend(*, parser, port, out_parser, out_port) -> dict:
    """Delivery 2's corrections (D117) on the existing folders, without A2.6's slow load. The parser folder gains the
    full reader cases (every distinct reference output and the corpus, from the full port export), every case's
    expected query and the readers' context; its 600 resolutions, decodes, masks, maps, assets and startup check are
    carried over unchanged. The port export gains the queries."""
    from . import acceptance as AC
    parser, port, out_parser, out_port = Path(parser), Path(port), Path(out_parser), Path(out_port)
    for o in (out_parser, out_port):
        if o.exists():
            raise HeadsetError(f"{o} already exists")
    man = verify_folder(parser)
    old_man = _sha((parser / "manifest.json").read_bytes())
    pbytes = (port / "port-acceptance.json").read_bytes()
    full = json.loads(pbytes)
    if full["bundle_manifest_sha256"] != man["bundle_manifest_sha256"]:
        raise HeadsetError("the port export and the parser folder come from different bundles")
    for e in full["resolutions"] + full["readers"]:
        e.pop("query", None)
    AC.add_queries(full)
    out_port.mkdir(parents=True)
    data = encode_json(full)
    (out_port / "port-acceptance.json").write_bytes(data)
    port_summary = {"resolutions": len(full["resolutions"]), "readers": len(full["readers"]), "records": len(full["records"]),
                    "sha256": _sha(data), "bytes": len(data), "amended_from": _sha(pbytes)}
    (out_port / "summary.json").write_bytes(encode_json(port_summary))
    res = json.loads((parser / "acceptance" / "resolutions.json").read_text(encoding="utf-8"))
    keys = {(e["parent_command_id"], e["view"]) for e in res["resolutions"]}
    in_full = {(e["parent_command_id"], e["view"]): e["expected"] for e in full["resolutions"]}
    for e in res["resolutions"]:
        k = (e["parent_command_id"], e["view"])
        if k not in in_full or in_full[k] != e["expected"]:
            raise HeadsetError(f"the folder's resolution {k} is not the full export's")
    body = dict(res, readers=[{k: r[k] for k in r if k != "query"} for r in full["readers"]])
    for e in body["resolutions"]:
        e.pop("query", None)
    AC.add_queries(body)
    tmp = out_parser.with_name(out_parser.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    files = {}

    def put(rel, data: bytes):
        q = tmp / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(data)
        files[rel] = {"sha256": _sha(data), "bytes": len(data)}
    replaced = {"acceptance/resolutions.json", "acceptance/readers.jsonl", "acceptance/reader-context.json"}
    for name in sorted(man["files"]):
        if name not in replaced:
            put(name, (parser / name).read_bytes())
    _put_acceptance(put, body)
    counts = dict(man["counts"], readers=len(body["readers"]), resolutions=len(keys))
    new_man = dict(man, files=files, counts=counts, amended={"decision": "D117", "from_manifest_sha256": old_man,
                                                              "port_sha256": _sha(pbytes)})
    mbody = encode_json(new_man)
    (tmp / "manifest.json").write_bytes(mbody)
    tmp.rename(out_parser)
    verify_folder(out_parser)
    return {"manifest_sha256": _sha(mbody), **counts, "port": port_summary}


def _issue(kind, where, expected, got):
    return {"kind": kind, "where": where, "expected": expected, "got": got}


def _results(path: Path, key, expected_keys, name) -> dict:
    """One result per expected case: a duplicate or an unexpected case refuses the evidence; a missing one is kept as missing."""
    got = {}
    for r in _jl(path):
        k = key(r)
        if k in got:
            raise HeadsetError(f"{name}: a duplicate result for {k}")
        if k not in expected_keys:
            raise HeadsetError(f"{name}: a result for {k}, which the folder does not expect")
        got[k] = r
    return got


def _qsha(q):
    return None if q is None else _sha(q.encode("utf-8"))


def judge(*, folder, results, out) -> dict:
    """The headset's results (pulled from the app's parser-check folder) against the folder's expectations (D117: the
    evidence is verified first; any failed prerequisite leaves every criterion not met)."""
    folder, results, out = Path(folder), Path(results), Path(out)
    man = verify_folder(folder)
    man_sha = _sha((folder / "manifest.json").read_bytes())
    run = json.loads((results / "run.json").read_text(encoding="utf-8"))
    if run.get("manifest_sha256") != man_sha:
        raise HeadsetError("the headset ran another parser folder")
    parser = json.loads((folder / "assets" / "parser.json").read_text(encoding="utf-8"))
    prereq = []
    if run.get("gguf_sha256") != parser["gguf_sha256"]:
        prereq.append(f"the model on the headset ({run.get('gguf_sha256')}) is not the exported candidate")
    if run.get("startup_check") != "passed":
        prereq.append(f"the startup check did not pass: {run.get('startup_check')}")
    acc = folder / "acceptance"
    exp = {d["parent_command_id"]: d for d in _jl(acc / "decodes.jsonl")}
    mexp = {tuple(m["prefix"]): m for m in _jl(acc / "masks.jsonl")}
    rexp = {(e["parent_command_id"], e["view"]): e for e in json.loads((acc / "resolutions.json").read_text(encoding="utf-8"))["resolutions"]}
    dexp = {r["text"]: r for r in _jl(acc / "readers.jsonl")}
    got = _results(results / "decodes.jsonl", lambda r: r["parent_command_id"], exp, "decodes") if (results / "decodes.jsonl").is_file() else {}
    mgot = _results(results / "masks.jsonl", lambda r: tuple(r["prefix"]), mexp, "masks") if (results / "masks.jsonl").is_file() else {}
    rgot = _results(results / "resolutions.jsonl", lambda r: (r["parent_command_id"], r["view"]), rexp, "resolutions") \
        if (results / "resolutions.jsonl").is_file() else {}
    dgot = _results(results / "readers.jsonl", lambda r: r["text"], dexp, "readers") if (results / "readers.jsonl").is_file() else {}
    for k, n in (("decodes", len(got)), ("masks", len(mgot)), ("resolutions", len(rgot)), ("readers", len(dgot))):
        if run.get(k) is not None and run.get(k) != n:
            prereq.append(f"run.json counts {run.get(k)} {k}, the results hold {n}")
    diffs, technical, missing = [], [], {"decodes": 0, "masks": 0, "resolutions": 0, "readers": 0}
    same, token_mismatch = 0, 0
    for pid, e in exp.items():
        g = got.get(pid)
        if g is None:
            missing["decodes"] += 1
            continue
        if g["status"] == "technical_failure":
            technical.append({"criterion": 7, "parent_command_id": pid, "error": g.get("error")})
            continue
        if g["prompt_ids"] != e["prompt_ids"]:
            token_mismatch += 1
            diffs.append(_issue("prompt_tokens", pid, len(e["prompt_ids"]), len(g["prompt_ids"])))
            continue
        if g["tokens"] == e["tokens"] and g["status"] == e["status"]:
            same += 1
        else:
            diffs.append(_issue("decode", pid, e["tokens"], g["tokens"]))
    mask_diff = 0
    for p, m in mexp.items():
        g = mgot.get(p)
        if g is None:
            missing["masks"] += 1
        elif (g["count"], g["ids_sha256"]) != (m["count"], m["ids_sha256"]):
            mask_diff += 1
            if mask_diff <= 20:
                diffs.append(_issue("mask", list(p), m["count"], g["count"]))
    res_diff = 0
    for k, e in rexp.items():
        g = rgot.get(k)
        want = e["expected"]
        if g is None:
            missing["resolutions"] += 1
            continue
        if g.get("technical_failure"):
            technical.append({"criterion": 1, "parent_command_id": k[0], "view": k[1], "error": g.get("error")})
            continue
        if want["technical"]:   # the reference output itself was technical: nothing to resolve, only the flag compares
            ok = g["technical"] is True
        else:
            ok = ({"outcome": g["outcome"], "target_id": g["target_id"], "candidate_ids": g["candidate_ids"], "technical": g["technical"]}
                  == {"outcome": want["outcome"], "target_id": want["target_id"], "candidate_ids": want["candidate_ids"] or [],
                      "technical": False}) and g.get("query_sha256") == _qsha(e.get("query"))
        if not ok:
            res_diff += 1
            if res_diff <= 20:
                diffs.append(_issue("resolution", f"{k[0]} {k[1]}", dict(want, query_sha256=_qsha(e.get("query"))), g))
    reader_diff = 0
    for t, e in dexp.items():
        g = dgot.get(t)
        if g is None:
            missing["readers"] += 1
            continue
        ok = g["valid"] == e["valid"] and (not e["valid"] or ((g["render"], g["interpretation"]) == (e["render"], e["interpretation"])
                                                                and g.get("query_sha256") == _qsha(e.get("query"))))
        if not ok:
            reader_diff += 1
            if reader_diff <= 20:
                diffs.append(_issue("reader", t, e, g))
    ok_all = not prereq
    tech_of = lambda c: [t for t in technical if t["criterion"] == c]  # noqa: E731
    verdict = {"criterion_1_headset": "met" if ok_all and res_diff == 0 and not missing["resolutions"] and not tech_of(1) else "not met",
               "criterion_2_headset": "met" if ok_all and reader_diff == 0 and not missing["readers"] else "not met",
               "criterion_4_headset": "met" if ok_all and mask_diff == 0 and not missing["masks"] else "not met",
               "criterion_7_headset": "met" if ok_all and same == len(exp) and not token_mismatch else "not met",
               "prerequisites": prereq or "met: this folder, the exported candidate's model, the startup check passed, counts consistent",
               "decodes": {"identical": same, "of": len(exp), "prompt_token_mismatches": token_mismatch, "missing": missing["decodes"]},
               "masks": {"differing": mask_diff, "of": len(mexp), "missing": missing["masks"]},
               "resolutions": {"differing": res_diff, "of": len(rexp), "missing": missing["resolutions"]},
               "readers": {"differing": reader_diff, "of": len(dexp), "missing": missing["readers"]},
               "technical_failures": technical, "differences": diffs, "run": run, "folder_manifest_sha256": man_sha}
    out.mkdir(parents=True, exist_ok=False)
    (out / "headset-judgement.json").write_bytes(encode_json(verdict))
    return verdict
