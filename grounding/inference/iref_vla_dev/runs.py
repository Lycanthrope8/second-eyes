"""A2.6c system runs: the models through A2.3d's accepted runner, and the rules through A2.3b's accepted sequence.

**Models.** `smoke` and `run` pass the read-only preflight first, then call A2.3d's `smoke` and `run_compare`
unchanged, with the context the preflight built:

- float32, eager attention, evaluation mode, batch one;
- one final-position forward with no cache, and no generation;
- the existing TF32 and determinism settings;
- the canaries on the independent last-hidden-state path;
- the smoke record that freezes the settings;
- the run lock, which resumes only when every frozen input, checkpoint, setting and code hash agrees;
- every request kept, with its technical status.

The preflight receipt is written beside each output. The run's results are read back against the frozen requests before
publication (`verify_results`).

**Rules.** The unchanged A2.2b parser and A2.1e resolver run per scene bundle, exactly as A2.3d's rules do (A2.3b's
`join` re-verifies each entry against the bundle: the request's document must be the bundle's rendered document):

- one parse per text;
- one resolver call per selected parent and inventory view, on the derived scene and command the models are offered;
- one result per parent and view, serving both formats: it is never counted as two observations.

No annotation is read by any system here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ...evaluation.iref_vla.protocol import encode_json, encode_jsonl, issue, runtime, sha256, EvaluationInputError
from ...evaluation.iref_vla_pilot import baseline as BL
from ..iref_vla_compare import design as CD
from ..iref_vla_compare import run as CR
from . import preflight as PF

RULES_FILES = ("manifest.json", "parses.jsonl", "rules.jsonl", "summary.json")


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    tmp.replace(path)


# --------------------------------------------------------------------------------------------- the models
def verify_results(folder, prep) -> list:
    """The run's results against the frozen requests: one row per request in order, each request's prompt hash, the
    technical statuses, complete offered scores for completed rows, and the manifest's output hashes."""
    f, bad = Path(folder), []
    try:
        man = json.loads((f / "manifest.json").read_text(encoding="utf-8"))
        rows = [json.loads(x) for x in (f / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        index = [json.loads(x) for x in (Path(prep) / "requests-index.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, ValueError) as e:
        return [f"unreadable results: {e}"]
    for name, h in man.get("outputs", {}).items():
        if sha256((f / name).read_bytes()) != h:
            bad.append(f"{name}: changed")
    if [r.get("request_id") for r in rows] != [x["request_id"] for x in index]:
        return bad + ["the results do not hold one row per frozen request, in order"]
    for r, x in zip(rows, index):
        if r.get("prompt_sha256") != x["prompt_sha256"]:
            bad.append(f"{x['request_id']}: another prompt")
        st = r.get("technical_status")
        if st not in ("completed", "context_budget_exceeded", "execution_failed"):
            bad.append(f"{x['request_id']}: unknown status {st!r}")
        elif st == "completed":
            codes = [m[0] for m in r["mapping"]]
            if [s.get("code") for s in r["scores"]] != codes or r["choice_code"] not in codes or not set(r["tied_codes"]) <= set(codes):
                bad.append(f"{x['request_id']}: incomplete offered scores or a choice outside them")
    return bad[:20]


def _preflight(prep, work, tokenizer_dirs, tokenizers, frozen, progress):
    toks = tokenizers or PF.load_tokenizers(tokenizer_dirs, CD.load_policy())
    return PF.preflight(prep=prep, work=work, tokenizers=toks, frozen=frozen, progress=progress)


def smoke(*, prep, work, model_key, model_dir, tokenizer_dirs=None, device, out, model_loader=None, tokenizers=None,
          evidence_fn=None, frozen=PF.FROZEN, progress=print) -> dict:
    pre = _preflight(prep, work, tokenizer_dirs, tokenizers, frozen, progress)
    _write(Path(f"{out}.preflight.json"), encode_json(pre["receipt"]))
    ctx = PF.context_for(pre, model_key, model_dir, evidence_fn=evidence_fn)
    return CR.smoke(requests=prep, model_key=model_key, model_dir=model_dir, tokenizer_dir=None, device=device, out=out,
                    model_loader=model_loader, ctx=ctx)


def run(*, prep, work, model_key, model_dir, tokenizer_dirs=None, device, smoke_record, out, resume=False, model_loader=None,
        tokenizers=None, evidence_fn=None, frozen=PF.FROZEN, progress=print) -> dict:
    pre = _preflight(prep, work, tokenizer_dirs, tokenizers, frozen, progress)
    _write(Path(f"{out}.preflight.json"), encode_json(pre["receipt"]))
    ctx = PF.context_for(pre, model_key, model_dir, evidence_fn=evidence_fn)
    return CR.run_compare(requests=prep, model_key=model_key, model_dir=model_dir, tokenizer_dir=None, device=device,
                          smoke_record=smoke_record, out=out, resume=resume, model_loader=model_loader, progress=progress,
                          ctx=ctx, verify_results=verify_results)


# --------------------------------------------------------------------------------------------- the rules
def run_rules(*, prep, work, out, frozen=PF.FROZEN, relation_config=None, direction_config=None) -> dict:
    """One resolver result per selected parent and view, scene by scene; returns the summary."""
    from ...preparation.iref_vla_dev import dev as DEV
    out = Path(out)
    if out.exists():
        raise EvaluationInputError([issue(str(out), "E_A26C_OUTPUT_EXISTS", "the output folder exists")])
    prep, work = Path(prep), Path(work)
    req_bytes = (prep / "requests.jsonl").read_bytes()
    if hashlib.sha256(req_bytes).hexdigest() != frozen["requests_sha256"]:
        raise EvaluationInputError([issue(str(prep), "E_A26C_FROZEN", "the request file is not the frozen one")])
    reqs = [json.loads(x) for x in req_bytes.decode("utf-8").splitlines() if x.strip()]
    rel, dirs = Path(relation_config or DEV.REL), Path(direction_config or DEV.DIRS)
    rules_protocol = BL.IN.relabel(BL.INPUT, BL.A2PROTOCOL.load_protocol)
    by_scene = {}
    for r in reqs:
        by_scene.setdefault(r["scene"], []).append(r)
    parses, rules, scenes = [], [], []
    for scene_name in sorted(by_scene):
        rs = by_scene[scene_name]
        src = BL.IN.load_bundle(work / scene_name / "bundle")
        parents = list(dict.fromkeys(r["parent_command_id"] for r in rs))
        by_key = {}
        for r in rs:
            m = src["meas"][(r["parent_command_id"], r["view"], r["format"])]
            by_key[(r["parent_command_id"], r["view"], r["format"])] = {
                "request_id": r["request_id"], "derived_scene_id": r["derived_scene_id"],
                "derived_command_id": r["derived_command_id"], "object_ids": list(r["choice_object_ids"][:-1]),
                "mapping": [[c, t, i] for c, t, i in zip(r["codes"], r["choice_object_ids"], r["code_token_ids"])],
                "source_document_path": m["paths"]["document"], "source_document_sha256": r["document_sha256"]}
        entries = BL.IN.join({"selected": parents, "by_key": by_key}, src, sample=False)
        cmap = src["category_map"]
        config = BL.PREDICT.resolver_config(rules_protocol, cmap, None)
        found = BL.RV.check_config(config)
        if found:
            raise EvaluationInputError([issue(f"{scene_name} resolver config {p['path']}", "E_A26C_RULES", p["message"]) for p in found])
        vocabulary, colours = cmap["model_vocabulary"], rules_protocol["colour_labels"]
        by_text, parse_of = {}, {}
        for e in entries:
            if e["parent"] in parse_of:
                continue
            if e["text"] not in by_text:
                by_text[e["text"]] = BL.PARSE.parse_record(e["parent"], e["text"], categories=vocabulary, colours=colours)
            p = dict(copy.deepcopy(by_text[e["text"]]), parent_command_id=e["parent"])
            issues = BL.A2PROTOCOL.parse_record_issues(p, f"parse {e['parent']}")
            if issues or p["parse_status"] != "parsed":
                raise EvaluationInputError([issue(f"parse {e['parent']}", "E_A26C_RULES",
                                                  "a selected parent no longer parses under the accepted parser")])
            parse_of[e["parent"]] = p
            parses.append(p)
        meta = {r["parent_command_id"]: r for r in rs}
        for e in entries:
            scene, command = e["scene"], e["command"]
            query = {"schema_version": 1, "record_type": "grounding_query",
                     "query_id": command["command_id"] + rules_protocol["query_suffix"], "scene_id": scene["scene_id"],
                     "scene_revision": scene["scene_revision"], "evidence_profile": scene["evidence_profile"],
                     "command_id": command["command_id"], "interpretations": [copy.deepcopy(parse_of[e["parent"]]["interpretation"])]}
            found, _ = BL.RV.check_query(query)
            if found:
                raise EvaluationInputError([issue(f"query {e['parent']} {e['view']}", "E_A26C_RULES", p["message"]) for p in found])
            resolution = BL.RESOLVE.resolve(scene, command, query, resolver_config=config, relation_config_path=rel,
                                            direction_config_path=dirs, category_maps=[cmap], trace=False)
            rec = BL.rules_record(e, query, resolution)
            m = meta[e["parent"]]
            rules.append(dict(rec, record_type="a26c_rules", run_id=PF.RUN_ID, scene=scene_name, group=m["group"],
                              partition=m["partition"], stratum=m["stratum"],
                              request_ids={f: PF.request_id(e["parent"], e["view"], f) for f in PF.FORMATS},
                              observations=1))
        scenes.append({"scene": scene_name, "bundle_manifest_sha256": src["manifest_sha256"],
                       "preparation_index_sha256": src["index_sha256"], "parents": len(parents), "entries": len(entries)})
    summary = dict(BL.summarize_rules(parses, rules, BL.MODES[1]), record_type="a26c_rules_summary", run_id=PF.RUN_ID,
                   note="one resolver result per parent and view; both formats share it")
    files = {"parses.jsonl": encode_jsonl(parses), "rules.jsonl": encode_jsonl(rules), "summary.json": encode_json(summary)}
    manifest = {"format_version": 1, "record_type": "a26c_rules_manifest", "run_id": PF.RUN_ID,
                "frozen": dict(frozen), "requests_sha256": hashlib.sha256(req_bytes).hexdigest(), "scenes": scenes,
                "rules_protocol": {"protocol_id": rules_protocol["protocol_id"],
                                   "file_sha256": sha256(BL.A2PROTOCOL.PROTOCOL_PATH.read_bytes())},
                "library_files": {"relation_config_sha256": sha256(rel.read_bytes()), "direction_config_sha256": sha256(dirs.read_bytes())},
                "outputs": {k: sha256(v) for k, v in files.items()},
                "code": {p.name: sha256(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "runtime": runtime(), "notes": ["No annotation was read.", "One result per parent and view."]}
    files["manifest.json"] = encode_json(manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.partial"
    if staging.exists():
        raise EvaluationInputError([issue(str(staging), "E_A26C_OUTPUT_EXISTS", "a partial rules folder exists")])
    staging.mkdir()
    for k, v in files.items():
        (staging / k).write_bytes(v)
    staging.rename(out)
    return summary
