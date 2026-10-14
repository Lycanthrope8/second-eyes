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
  recorded outcome) and readers.jsonl (every distinct reference output and the 97-string corpus);
- manifest.json: every file's SHA-256 and size. Its own SHA-256 identifies the folder in every logged outcome.
The judge compares the headset's results with the folder's expectations, criterion by criterion.
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
    put("acceptance/resolutions.json", encode_json({k: port[k] for k in ("format_version", "record_type", "bundle_manifest_sha256",
                                                                          "protocol", "records", "resolutions")}))
    put("acceptance/readers.jsonl", encode_jsonl(port["readers"]))
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


def _issue(kind, where, expected, got):
    return {"kind": kind, "where": where, "expected": expected, "got": got}


def judge(*, folder, results, out) -> dict:
    """The headset's results (pulled from the app's parser-check folder) against the folder's expectations."""
    folder, results, out = Path(folder), Path(results), Path(out)
    man_bytes = (folder / "manifest.json").read_bytes()
    run = json.loads((results / "run.json").read_text(encoding="utf-8"))
    if run.get("manifest_sha256") != _sha(man_bytes):
        raise HeadsetError("the headset ran another parser folder")
    diffs, technical = [], []
    exp = {d["parent_command_id"]: d for d in _jl(folder / "acceptance" / "decodes.jsonl")}
    got = {d["parent_command_id"]: d for d in _jl(results / "decodes.jsonl")}
    same = 0
    for pid, e in exp.items():
        g = got.get(pid)
        if g is None or g["status"] == "technical_failure":
            technical.append({"criterion": 7, "parent_command_id": pid, "error": None if g is None else g.get("error")})
            continue
        if g["prompt_ids"] != e["prompt_ids"]:
            diffs.append(_issue("tokenization", pid, len(e["prompt_ids"]), len(g["prompt_ids"])))
        if g["tokens"] == e["tokens"] and g["status"] == e["status"]:
            same += 1
        else:
            diffs.append(_issue("decode", pid, e["tokens"], g["tokens"]))
    mexp = {tuple(m["prefix"]): m for m in _jl(folder / "acceptance" / "masks.jsonl")}
    mgot = {tuple(m["prefix"]): m for m in _jl(results / "masks.jsonl")}
    mask_diff = [p for p, m in mexp.items() if p not in mgot or (mgot[p]["count"], mgot[p]["ids_sha256"]) != (m["count"], m["ids_sha256"])]
    for p in mask_diff[:20]:
        diffs.append(_issue("mask", list(p), mexp[p]["count"], (mgot.get(p) or {}).get("count")))
    rexp = json.loads((folder / "acceptance" / "resolutions.json").read_text(encoding="utf-8"))["resolutions"]
    rgot = {(r["parent_command_id"], r["view"]): r for r in _jl(results / "resolutions.jsonl")}
    res_diff = 0
    for e in rexp:
        g = rgot.get((e["parent_command_id"], e["view"]))
        want = e["expected"]
        if g is None or g.get("technical_failure"):
            technical.append({"criterion": 1, "parent_command_id": e["parent_command_id"], "view": e["view"],
                              "error": None if g is None else g.get("error")})
            continue
        mine = {"outcome": g["outcome"], "target_id": g["target_id"], "candidate_ids": g["candidate_ids"], "technical": g["technical"]}
        if want["technical"]:   # the reference output itself was technical: nothing to resolve, only the flag compares
            same_res = g["technical"] is True
        else:
            same_res = mine == {"outcome": want["outcome"], "target_id": want["target_id"],
                                "candidate_ids": want["candidate_ids"] or [], "technical": False}
        if not same_res:
            res_diff += 1
            if res_diff <= 20:
                diffs.append(_issue("resolution", f"{e['parent_command_id']} {e['view']}", want, mine))
    rd_exp = _jl(folder / "acceptance" / "readers.jsonl")
    rd_got = {r["text"]: r for r in _jl(results / "readers.jsonl")}
    reader_diff = 0
    for e in rd_exp:
        g = rd_got.get(e["text"])
        if g is None or g["valid"] != e["valid"] or (e["valid"] and (g["render"], g["interpretation"]) != (e["render"], e["interpretation"])):
            reader_diff += 1
            if reader_diff <= 20:
                diffs.append(_issue("reader", e["text"], e, g))
    verdict = {"criterion_1_headset": "met" if res_diff == 0 and not [t for t in technical if t["criterion"] == 1] else "not met",
               "criterion_2_headset": "met" if reader_diff == 0 else "not met",
               "criterion_4_headset": "met" if not mask_diff else "not met",
               "criterion_7_headset": "met" if same == len(exp) else "not met",
               "decodes": {"identical": same, "of": len(exp)}, "masks": {"differing": len(mask_diff), "of": len(mexp)},
               "resolutions": {"differing": res_diff, "of": len(rexp)}, "readers": {"differing": reader_diff, "of": len(rd_exp)},
               "technical_failures": technical, "differences": diffs, "run": run}
    out.mkdir(parents=True, exist_ok=False)
    (out / "headset-judgement.json").write_bytes(encode_json(verdict))
    return verdict
