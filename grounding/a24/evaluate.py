"""A2.4's development evaluation (D109). Runs in the accepted evaluation environment (`.venv`, unchanged: no PEFT),
after all nine fits and their selections, on A2.6's frozen benchmark, complete and unfiltered.

**Direct** (`direct`): each merged checkpoint goes through A2.6c's preflight-bound steps (`runs.smoke`, `runs.run`) with
the fine-tuned identity as the runner's evidence function, over all 10,860 frozen requests (the runner's readback needs
every request). The runner's own records carry the base model key, which names the tokenizer and protocol family; the
evidence it records (tie and digest) names the fine-tuned model, and `model.json` beside the run binds the run to that
model by name and identity hash. Each configuration's primary comparison is its trained format; the other format's
rows are a predeclared cross-format transfer analysis.

**Parser** (`parser`): one constrained decode per parent (development and legacy), on the merged model loaded by the
accepted `TorchModel` (float32, eager, deterministic settings); the text is validated by the independent reader, then
resolved by the accepted resolver in each inventory view, exactly as the rules baseline resolves. No target is read.

**Scoring** (`score`) reads the targets: A2.6c's `score` for the direct runs (with the base 0.5B and 7B runs re-scored
from their saved results, never re-run), and the same outcome rule as the rules for the parser. **Analysis**
(`analyze`): A2.6c's estimators and environment bootstrap (unchanged code, imported), per seed, the three seeds' mean,
their spread reported apart, paired differences, legacy apart, and the predeclared supplementary breakdown by the
rules' saved per-row outcome.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..evaluation.iref_vla.protocol import EvaluationInputError, encode_json, encode_jsonl, issue, runtime
from . import decoder as DC
from . import identity as ID
from . import language as L
from . import prep as PR

BASE_KEY = "qwen2.5-0.5b-instruct"
LARGE_KEY = "qwen2.5-7b-instruct"
VIEWS = ("full_inventory", "source_known_nyu")
FORMATS = ("coordinates_v2", "coordinates_relations_v2")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _jl(p) -> list:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def _fail(where, message):
    raise EvaluationInputError([issue(str(where), "E_A24_EVAL", message)])


def model_protocol():
    from ..inference.iref_vla.protocol import load_protocol
    from ..inference.iref_vla_compare import design as CD
    from ..inference.iref_vla_compare import run as CR
    return CR.model_protocol(load_protocol(), CD.load_policy(), BASE_KEY)


def _binding(out: Path, doc: dict) -> None:
    p = out / "model.json"
    if p.is_file():
        old = json.loads(p.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k not in ("written_utc", "run")} != {k: v for k, v in doc.items() if k != "run"}:
            _fail(p, "this folder belongs to another model or binding")
        doc = dict(old, **{k: v for k, v in doc.items() if v is not None})
    p.write_text(json.dumps(dict(doc, written_utc=_now()), indent=1, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------- direct
def direct(*, prep, work, preflight, model_dir, out, device="cuda", resume=False, smoke_only=False, frozen=None,
           tokenizers=None, model_loader=None, progress=print) -> dict:
    """model_loader: tests only (the accepted runner's CPU path for labelled stand-ins); never set by the commands."""
    from ..inference.iref_vla_dev import preflight as PF
    from ..inference.iref_vla_dev import runs as RN
    mp, _ = model_protocol()
    ident = ID.load_identity(model_dir, mp)
    if ident["configuration"]["design"] != "direct" and not smoke_only:
        _fail(model_dir, "not a direct-selection model (a parser model takes the runner's smoke only, as a mechanical "
                         "check of its identity path)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fmt = ident["configuration"]["format"]
    _binding(out, {"record_type": "a24_direct_evaluation", "model_name": ident["model_name"],
                   "identity_sha256": ident["identity_sha256"], "model_dir": str(model_dir), "primary_format": fmt,
                   "cross_format": [f for f in FORMATS if f != fmt] if fmt else [],
                   "runner_model_key": f"{BASE_KEY}: the tokenizer and protocol family the accepted runner uses; the "
                                       "weights are the fine-tuned model named here (see the evidence tie)",
                   "run": None})
    common = dict(prep=prep, work=work, preflight_dir=preflight, model_key=BASE_KEY, model_dir=model_dir, device=device,
                  evidence_fn=ID.finetuned_evidence, frozen=frozen or PF.FROZEN, progress=progress)
    if model_loader is not None:
        common["model_loader"] = model_loader
    if tokenizers is not None:
        common["tokenizers"] = tokenizers
    else:
        common["tokenizer_dirs"] = {BASE_KEY: str(PR.DEV.TOKENIZER)}
    smoke = out / "smoke.json"
    if not smoke.is_file():
        RN.smoke(**common, out=smoke)
    rec = json.loads(smoke.read_text(encoding="utf-8"))
    if not rec.get("accepted") or ident["model_name"] not in (rec.get("checkpoint_tie") or ""):
        _fail(smoke, "the runner's smoke did not accept this model through its fine-tuned identity")
    if smoke_only:
        return {"model_name": ident["model_name"], "smoke": rec}
    res = RN.run(**common, smoke_record=smoke, out=out / "run", resume=resume)
    _binding(out, {"record_type": "a24_direct_evaluation", "model_name": ident["model_name"],
                   "identity_sha256": ident["identity_sha256"], "model_dir": str(model_dir), "primary_format": fmt,
                   "cross_format": [f for f in FORMATS if f != fmt],
                   "runner_model_key": f"{BASE_KEY}: the tokenizer and protocol family the accepted runner uses; the "
                                       "weights are the fine-tuned model named here (see the evidence tie)",
                   "run": {"manifest_sha256": _sha((out / "run" / "manifest.json").read_bytes()), "folder": "run"}})
    return {"model_name": ident["model_name"], "run": res}


# --------------------------------------------------------------------------------------------- parser
def _parents(prep, work):
    """Each sampled parent once (index order), with its text and each view's derived scene and command paths."""
    from ..inference.iref_vla import prepare as I
    out, seen, bundles = [], set(), {}
    for x in _jl(Path(prep) / "requests-index.jsonl"):
        if x["parent_command_id"] in seen:
            continue
        seen.add(x["parent_command_id"])
        if x["scene"] not in bundles:
            bundles[x["scene"]] = I.read_bundle(Path(work) / x["scene"] / "bundle")
        src = bundles[x["scene"]]
        views = {v: {"scene_path": src["index"][(x["parent_command_id"], v)]["scene_path"],
                     "command_path": src["index"][(x["parent_command_id"], v)]["command_path"]} for v in VIEWS}
        cmd = json.loads((Path(work) / x["scene"] / "bundle" / views[VIEWS[0]]["command_path"]).read_text(encoding="utf-8"))
        out.append({k: x[k] for k in ("partition", "group", "scene", "stratum", "parent_command_id")} | {
            "text": cmd["text"], "view_paths": views})
    return out


def parser(*, prep, work, model_dir, out, device="cuda", resume=False, frozen=None, model_loader=None,
           progress=print) -> dict:
    from ..inference.iref_vla.model import TorchModel
    from ..inference.iref_vla_dev import preflight as PF
    from ..preparation.iref_vla import tokens as T
    from . import labels as LB
    mp, desc = model_protocol()
    ident = ID.load_identity(model_dir, mp)
    if ident["configuration"]["design"] != "parser":
        _fail(model_dir, "not a parser model")
    ev = ID.finetuned_evidence(model_dir, mp, desc)
    pman_bytes = (Path(prep) / "manifest.json").read_bytes()
    pman = json.loads(pman_bytes)
    if pman["hashes"]["requests"] != (frozen or PF.FROZEN)["requests_sha256"]:
        _fail(prep, "not the frozen A2.6 benchmark")
    if _sha((Path(prep) / "requests-index.jsonl").read_bytes()) != pman["files"]["requests-index.jsonl"]:
        _fail(prep, "the request index changed since the preparation")
    settings = PR.load_settings()
    lock = {"record_type": "a24_parser_evaluation_lock", "model_name": ident["model_name"],
            "identity_sha256": ident["identity_sha256"], "evidence_tie": ev["tie"],
            "prep_manifest_sha256": _sha(pman_bytes), "language_sha256": L.spec_sha256(),
            "parser_prompt_sha256": _sha(json.dumps(settings["parser_prompt"], sort_keys=True).encode()),
            "code": PR.code_hashes()}
    out = Path(out)
    if out.exists() and (out / "manifest.json").is_file():
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if out.exists():
        if not resume:
            _fail(out, "exists; pass resume to continue it")
        if json.loads((out / "lock.json").read_text(encoding="utf-8")) != lock:
            _fail(out, "the lock differs (model, inputs or code): refusing to continue; partial rows kept")
    else:
        out.mkdir(parents=True)
        (out / "lock.json").write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    started = _now()
    parents = _parents(prep, work)
    done = {json.loads(x)["parent_command_id"] for x in (out / "rows.partial.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()} if (out / "rows.partial.jsonl").is_file() else set()
    tok = T.load_pinned_tokenizer(PR.DEV.TOKENIZER)
    constraint = DC.Constraint(L.load_spec(), DC.TokenTable(tok._backend))
    spec = L.load_spec()
    tm = (model_loader or TorchModel.load)(model_dir, device, mp)
    checker, maps = LB.Checker(spec), {}
    t0 = time.perf_counter()
    with open(out / "rows.partial.jsonl", "a", encoding="utf-8") as f:
        for k, p in enumerate(parents):
            if p["parent_command_id"] in done:
                continue
            ids = tok.encode(PR.parser_prompt(p["text"], settings))
            res = DC.validate(DC.decode(DC.torch_step(tm.model, ids, device), constraint, spec["max_new_tokens"]), spec)
            b = Path(work) / p["scene"] / "bundle"
            if p["scene"] not in maps:
                maps[p["scene"]] = json.loads((b / "model_records" / "category-map.json").read_text(encoding="utf-8"))
            views = {}
            for v in VIEWS:
                vp = p["view_paths"][v]
                if not res["valid"]:
                    views[v] = {"outcome": f"decode:{res['status']}" if res["status"] != "completed" else "invalid_output",
                                "target_id": None, "technical": True}
                    continue
                scene = json.loads((b / vp["scene_path"]).read_text(encoding="utf-8"))
                command = json.loads((b / vp["command_path"]).read_text(encoding="utf-8"))
                t1 = time.perf_counter()
                rr = checker.resolve(L.interpretation(res["parsed"]), scene, command, maps[p["scene"]])
                views[v] = {"outcome": rr["outcome"], "target_id": rr.get("target_id"), "technical": False,
                            "candidate_ids": rr.get("candidate_ids"), "resolve_ms": round((time.perf_counter() - t1) * 1000, 3)}
            f.write(json.dumps({**{x: p[x] for x in ("partition", "group", "scene", "stratum", "parent_command_id")},
                                "input_tokens": len(ids), "status": res["status"], "valid": res["valid"],
                                "text": res["text"], "token_ids": res["token_ids"], "steps": res["steps"],
                                "mask_ms": res["mask_ms"], "model_ms": res["model_ms"], "views": views},
                               sort_keys=True) + "\n")
            f.flush()
            if (k + 1) % 100 == 0:
                progress(f"  {k + 1}/{len(parents)} parents")
    rows = {r["parent_command_id"]: r for r in _jl(out / "rows.partial.jsonl")}
    ordered = [rows[p["parent_command_id"]] for p in parents]
    body = encode_jsonl(ordered)
    (out / "results.jsonl").write_bytes(body)
    man = {"format_version": 1, "record_type": "a24_parser_evaluation", "model_name": ident["model_name"],
           "identity_sha256": ident["identity_sha256"], "lock": lock, "started_utc": started, "finished_utc": _now(),
           "seconds_this_session": round(time.perf_counter() - t0, 1), "parents": len(ordered),
           "statuses": dict(Counter(r["status"] for r in ordered)),
           "outputs": {"results.jsonl": _sha(body)}, "model_info": tm.info() if hasattr(tm, "info") else None,
           "runtime": runtime(), "note": "one decode per parent; both views resolved from it (not independent)"}
    (out / "manifest.json").write_bytes(encode_json(man))
    return man


# --------------------------------------------------------------------------------------------- scoring
def score(*, prep, rules, base_runs, direct_evals, parser_evals, out, frozen=None) -> dict:
    """base_runs: model key -> A2.6c run folder (re-scored, not re-run); direct_evals and parser_evals: folders written
    by `direct` and `parser`. Writes out/direct (A2.6c's score) and out/parser-scores.jsonl."""
    from ..inference.iref_vla_dev import preflight as PF
    from ..inference.iref_vla_dev import score as SC
    out = Path(out)
    if out.exists():
        _fail(out, "the output folder exists")
    runs = dict(base_runs)
    names = {}
    for d in direct_evals:
        m = json.loads((Path(d) / "model.json").read_text(encoding="utf-8"))
        if not m.get("run"):
            _fail(d, "the direct evaluation has no completed run")
        if _sha((Path(d) / "run" / "manifest.json").read_bytes()) != m["run"]["manifest_sha256"]:
            _fail(d, "the run changed since it was bound to its model")
        runs[m["model_name"]] = Path(d) / "run"
        names[m["model_name"]] = m
    out.mkdir(parents=True)
    SC.score(prep=prep, rules=rules, runs=runs, out=out / "direct", frozen=frozen or PF.FROZEN)
    pman = json.loads((Path(prep) / "manifest.json").read_text(encoding="utf-8"))
    tpath = Path(prep) / "reference_only" / "targets.jsonl"
    if _sha(tpath.read_bytes()) != pman["files"]["reference_only/targets.jsonl"]:
        _fail(tpath, "changed since the preparation")
    targets = {t["parent_command_id"]: t for t in _jl(tpath)}
    prows, parsers = [], {}
    for d in parser_evals:
        man = json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8"))
        if _sha((Path(d) / "results.jsonl").read_bytes()) != man["outputs"]["results.jsonl"]:
            _fail(d, "the parser results changed since they were written")
        parsers[man["model_name"]] = {"manifest_sha256": _sha((Path(d) / "manifest.json").read_bytes())}
        for r in _jl(Path(d) / "results.jsonl"):
            t = targets[r["parent_command_id"]]
            for v in VIEWS:
                x = r["views"][v]
                if x["technical"]:
                    o = f"technical:{x['outcome']}"
                elif x["outcome"] == "resolved":
                    o = "correct" if x["target_id"] == t["target_object_id"] else "resolved_other_object"
                else:
                    o = x["outcome"]
                prows.append({**{k: r[k] for k in ("partition", "group", "scene", "stratum", "parent_command_id")},
                              "view": v, "system": man["model_name"], "outcome": o, "parsing_label": t["parsing_label"],
                              "target_offered": t["target_offered"][v], "observations": 1})
    body = encode_jsonl(prows)
    (out / "parser-scores.jsonl").write_bytes(body)
    man = {"format_version": 1, "record_type": "a24_scores_manifest", "direct_models": sorted(names),
           "base_runs": {k: _sha((Path(v) / "manifest.json").read_bytes()) for k, v in base_runs.items()},
           "parsers": parsers, "outputs": {"parser-scores.jsonl": _sha(body),
                                           "direct/manifest.json": _sha((out / "direct" / "manifest.json").read_bytes())},
           "counts": {"parser": {s: dict(Counter(r["outcome"] for r in prows if r["system"] == s)) for s in parsers}},
           "scored_utc": _now(), "runtime": runtime()}
    (out / "manifest.json").write_bytes(encode_json(man))
    return man


# --------------------------------------------------------------------------------------------- analysis
def _system(name):
    """(kind, configuration, primary format) of a system name."""
    if name.startswith("a24.direct."):
        cfg = name.rsplit(".s", 1)[0][len("a24."):]
        return "direct", cfg, cfg.split(".", 1)[1]
    if name.startswith("a24.parser."):
        return "parser", "parser", None
    return "baseline", name, None


def observations(srows, rrows, prows, partition):
    """combo -> {(env, stratum): [outcome per sampled parent]}; combos are (system, view, format or None, role)."""
    obs = defaultdict(lambda: defaultdict(list))
    for r in rrows:
        if r["partition"] == partition:
            obs[("rules", r["view"], None, "baseline")][(r["group"], r["stratum"])].append(r["outcome"])
    for r in prows:
        if r["partition"] == partition:
            obs[(r["system"], r["view"], None, "primary")][(r["group"], r["stratum"])].append(r["outcome"])
    for r in srows:
        if r["partition"] != partition:
            continue
        obs[("always_b", r["view"], r["format"], "diagnostic")][(r["group"], r["stratum"])].append(r["always_b"]["outcome"])
        for k, m in r["models"].items():
            kind, _, fmt = _system(k)
            role = "baseline" if kind == "baseline" else ("primary" if r["format"] == fmt else "cross_format")
            obs[(k, r["view"], r["format"], role)][(r["group"], r["stratum"])].append(m["outcome"])
    return obs


def _name(c) -> str:
    return " | ".join(x for x in (c[0], c[1], c[2] or "(both formats)") if x) + ("" if c[3] in ("primary", "baseline")
                                                                                 else f" [{c[3]}]")


def analyze(*, prep, scores, out, b=None, seed=None, supersedes=None) -> dict:
    """supersedes: an earlier analysis folder this one replaces; it is kept unchanged and its hash is recorded."""
    from ..inference.iref_vla_dev import analysis as AN
    b = AN.B if b is None else b
    seed = AN.SEED if seed is None else seed
    prep, scores, out = Path(prep), Path(scores), Path(out)
    if out.exists():
        _fail(out, "the output folder exists")
    sman = json.loads((scores / "manifest.json").read_text(encoding="utf-8"))
    if _sha((scores / "parser-scores.jsonl").read_bytes()) != sman["outputs"]["parser-scores.jsonl"]:
        _fail(scores, "parser scores changed since scoring")
    dman = json.loads((scores / "direct" / "manifest.json").read_text(encoding="utf-8"))
    for name, h in dman["outputs"].items():
        if _sha((scores / "direct" / name).read_bytes()) != h:
            _fail(scores / "direct" / name, "changed since scoring")
    srows, rrows = _jl(scores / "direct" / "scores.jsonl"), _jl(scores / "direct" / "rules-scores.jsonl")
    prows = _jl(scores / "parser-scores.jsonl")
    srecs = _jl(prep / "strata.jsonl")
    strata = {(s["group"], s["stratum"]): (s["N"], s["n"]) for s in srecs if s["stratum"]}
    result = {"format_version": 1, "record_type": "a24_analysis", "analysed_utc": _now(),
              "configuration": {"replicates": b, "seed": seed, "estimators": "A2.6c's, imported unchanged",
                                "interval": "95% by nearest rank (A2.3e, D97)", "seeds": PR.load_settings()["analysis"]["seeds"],
                                "supplementary": PR.load_settings()["analysis"]["supplementary"]},
              "partitions": {}}
    for part in ("development", "legacy-development"):
        obs = observations(srows, rrows, prows, part)
        aggs = {c: AN.aggregates(o, strata) for c, o in obs.items()}
        block = {"systems": {}, "environments": sorted({e for a in aggs.values() for e in a})}
        for c, agg in aggs.items():
            block["systems"][_name(c)] = {"combo": list(c), "estimate": AN.estimate(agg),
                                          "outcome_rates_command_weighted": AN.categories(agg),
                                          "parents": int(sum(a["n"] for a in agg.values()))}
        groups = defaultdict(list)   # (configuration, view, format, role) -> seed systems
        for c in aggs:
            kind, cfg, _ = _system(c[0])
            if kind != "baseline" and c[0] != "rules" and c[0] != "always_b":
                groups[(cfg, c[1], c[2], c[3])].append(c)
        if part == "development":
            envs = block["environments"]
            reps = {c: {k: [] for k in ("command_weighted", "environment_weighted", "unweighted")} for c in aggs}
            for cnt in AN.draws(envs, b, seed):
                for c, agg in aggs.items():
                    for k, v in AN.estimate(agg, cnt).items():
                        reps[c][k].append(v)
            for c in aggs:
                block["systems"][_name(c)]["interval"] = {k: AN.interval(v) for k, v in reps[c].items()}
            block["seed_summaries"] = {}
            mean_reps = {}
            for g, cs in sorted(groups.items(), key=lambda x: str(x[0])):
                cs = sorted(cs)
                pts = {k: [block["systems"][_name(c)]["estimate"][k] for c in cs] for k in reps[cs[0]]}
                mean_reps[g] = {k: [statistics.fmean(vals) for vals in zip(*(reps[c][k] for c in cs))] for k in reps[cs[0]]}
                block["seed_summaries"][" | ".join(x for x in g if x)] = {
                    "systems": [c[0] for c in cs],
                    "mean": {k: statistics.fmean(v) for k, v in pts.items()},
                    "mean_interval": {k: AN.interval(v) for k, v in mean_reps[g].items()},
                    "seed_spread": {k: {"min": min(v), "max": max(v), "sd": statistics.stdev(v) if len(v) > 1 else None}
                                    for k, v in pts.items()},
                    "note": "seed spread is training-seed variation; the interval is environment uncertainty of the mean"}
            block["differences"] = differences(aggs, reps, mean_reps, groups, block)
            block["supplementary"] = supplementary(srows, rrows, prows, strata, part)
        result["partitions"][part] = block
    if supersedes is not None:
        old_json = Path(supersedes) / "analysis.json"
        result["supersedes"] = {"folder": str(supersedes), "analysis_sha256": _sha(old_json.read_bytes()),
                                "changes": analysis_diff(old_json, None, new=result)}
    out.mkdir(parents=True)
    (out / "analysis.json").write_bytes(encode_json(result))
    (out / "report.md").write_text(render(result), encoding="utf-8")
    return result


def _walk(x, path=()):
    """Every interval ({low, high}) and every other number of an analysis, by its JSON path."""
    if isinstance(x, dict):
        if "low" in x and "high" in x and not isinstance(x.get("low"), dict):
            yield ("interval", path, (x["low"], x["high"]))
            return
        for k, v in x.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _walk(v, path + (str(i),))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield ("number", path, x)


def analysis_diff(old, new_path=None, new=None) -> dict:
    """What changed between two analyses (A2.6c's or A2.4's analysis.json): every interval endpoint's change in
    percentage points, any interval whose crossing of zero changed, and any other number that changed. Numbers that
    describe the run itself (the bootstrap's replicate count, a superseded record) are compared too."""
    a = json.loads(Path(old).read_text(encoding="utf-8"))
    b = new if new is not None else json.loads(Path(new_path).read_text(encoding="utf-8"))
    b = {k: v for k, v in b.items() if k != "supersedes"}
    a = {k: v for k, v in a.items() if k != "supersedes"}
    ia = {p: v for kind, p, v in _walk(a) if kind == "interval"}
    ib = {p: v for kind, p, v in _walk(b) if kind == "interval"}
    na = {p: v for kind, p, v in _walk(a) if kind == "number"}
    nb = {p: v for kind, p, v in _walk(b) if kind == "number"}
    crosses = lambda lo, hi: lo is not None and hi is not None and lo <= 0 <= hi   # noqa: E731
    changes, crossing = [], []
    for p in sorted(set(ia) & set(ib)):
        (l1, h1), (l2, h2) = ia[p], ib[p]
        for e1, e2 in ((l1, l2), (h1, h2)):
            if e1 is not None and e2 is not None and e1 != e2:
                changes.append(abs(e2 - e1) * 100)
        if crosses(l1, h1) != crosses(l2, h2):
            crossing.append(" / ".join(p))
    other = [" / ".join(p) for p in sorted(set(na) & set(nb)) if na[p] != nb[p]]
    return {"intervals_compared": len(set(ia) & set(ib)), "endpoints_changed": len(changes),
            "max_endpoint_change_pp": max(changes) if changes else 0.0,
            "zero_crossing_changes": crossing, "other_numbers_changed": other,
            "only_in_old": len(set(ia) - set(ib)) + len(set(na) - set(nb)),
            "only_in_new": len(set(ib) - set(ia)) + len(set(nb) - set(na))}


def differences(aggs, reps, mean_reps, groups, block) -> dict:
    """Paired, within each replicate: every fine-tuned system against the base, the rules and always-B in its own view and
    format; the parser against each direct configuration at the same seed; and the same for the seeds' means."""
    from ..inference.iref_vla_dev import analysis as AN
    out = {}

    def pair(a_vals, b_vals, a_pt, b_pt):
        d = {}
        for k in a_vals:
            reps_d = [x - y for x, y in zip(a_vals[k], b_vals[k]) if x is not None and y is not None]
            d[k] = {"estimate": None if a_pt[k] is None or b_pt[k] is None else a_pt[k] - b_pt[k],
                    "interval": AN.interval(reps_d)}
        return d
    est = {c: block["systems"][_name(c)]["estimate"] for c in aggs}
    base = {(c[1], c[2]): c for c in aggs if c[0] == BASE_KEY}
    rules = {c[1]: c for c in aggs if c[0] == "rules"}
    alwb = {(c[1], c[2]): c for c in aggs if c[0] == "always_b"}
    for c in aggs:
        kind, cfg, fmt = _system(c[0])
        if kind == "baseline" or c[3] == "cross_format":
            continue
        comps = [rules.get(c[1])]
        if kind == "direct":
            comps += [base.get((c[1], c[2])), alwb.get((c[1], c[2]))]
        else:
            comps += [base.get((c[1], f)) for f in FORMATS]
            seed = c[0].rsplit(".s", 1)[1]
            comps += [x for x in aggs if x[0].endswith(f".s{seed}") and x[0].startswith("a24.direct.")
                      and x[1] == c[1] and x[3] == "primary"]
        for o in comps:
            if o is not None:
                out[f"{_name(c)} minus {_name(o)}"] = pair(reps[c], reps[o], est[c], est[o])
    for g, vals in mean_reps.items():
        cs = sorted(groups[g])
        pt = {k: statistics.fmean(est[c][k] for c in cs) for k in vals}
        if g[3] != "primary":
            continue
        for o in [rules.get(g[1]), base.get((g[1], g[2])) if g[2] else None]:
            if o is not None:
                out[f"mean of seeds: {' | '.join(x for x in g[:3] if x)} minus {_name(o)}"] = pair(vals, reps[o], pt, est[o])
    return out


def supplementary(srows, rrows, prows, strata, part) -> dict:
    """Supplementary coverage diagnostic (predeclared; headline denominators unchanged): each system's command-weighted
    accuracy within each of the rules' saved per-row outcomes, per view and format."""
    rule_of = {(r["parent_command_id"], r["view"]): r["outcome"] for r in rrows if r["partition"] == part}
    acc = defaultdict(lambda: [0.0, 0.0, 0])

    def add(key, r, outcome):
        N, n = strata[(r["group"], r["stratum"])]
        a = acc[key]
        a[0] += N / n * (outcome == "correct")
        a[1] += N / n
        a[2] += 1
    for r in srows:
        if r["partition"] != part:
            continue
        bin_ = rule_of[(r["parent_command_id"], r["view"])]
        for k, m in r["models"].items():
            add((k, r["view"], r["format"], bin_), r, m["outcome"])
    for r in prows:
        if r["partition"] == part:
            add((r["system"], r["view"], None, rule_of[(r["parent_command_id"], r["view"])]), r, r["outcome"])
    return {" | ".join(str(x) for x in k if x): {"rows": v[2], "command_weighted_accuracy": v[0] / v[1] if v[1] else None}
            for k, v in sorted(acc.items(), key=lambda x: str(x[0]))}


def _p(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render(r) -> str:
    L_ = ["# A2.4 development evaluation (the frozen A2.6 benchmark)", "",
          "Command-weighted and environment-weighted estimates with 95% environment-bootstrap intervals (nearest rank). "
          "Each fine-tuned system is shown on its own trained format; cross-format rows are labelled. Seed spread is "
          "training-seed variation and is never folded into the intervals. Development and legacy are apart.", ""]
    dev = r["partitions"]["development"]
    L_ += ["## Systems (development)", "",
           "| System | View | Format | Command-weighted | Environment-weighted | Parents |", "|---|---|---|---|---|---|"]
    for name, s in sorted(dev["systems"].items()):
        cells = (name.split(" | ") + ["", ""])[:3]   # system, view and format each in its own cell
        i = s.get("interval", {})
        cw = f"{_p(s['estimate']['command_weighted'])} ({_p(i.get('command_weighted', {}).get('low'))} to " \
             f"{_p(i.get('command_weighted', {}).get('high'))})"
        ew = f"{_p(s['estimate']['environment_weighted'])} ({_p(i.get('environment_weighted', {}).get('low'))} to " \
             f"{_p(i.get('environment_weighted', {}).get('high'))})"
        L_.append(f"| {cells[0]} | {cells[1]} | {cells[2]} | {cw} | {ew} | {s['parents']} |")
    L_ += ["", "## Seeds (development)", ""]
    for g, s in dev.get("seed_summaries", {}).items():
        L_.append(f"- {g}: mean command-weighted {_p(s['mean']['command_weighted'])} "
                  f"({_p(s['mean_interval']['command_weighted']['low'])} to {_p(s['mean_interval']['command_weighted']['high'])}); "
                  f"seed spread {_p(s['seed_spread']['command_weighted']['min'])} to {_p(s['seed_spread']['command_weighted']['max'])}")
    L_ += ["", "## Paired differences (development, command-weighted)", ""]
    for k, d in dev.get("differences", {}).items():
        x = d["command_weighted"]
        L_.append(f"- {k}: {_p(x['estimate'])} ({_p(x['interval']['low'])} to {_p(x['interval']['high'])})")
    L_ += ["", "## Supplementary: accuracy within the rules' saved per-row outcome (labelled; not a causal estimate)", ""]
    for k, v in dev.get("supplementary", {}).items():
        L_.append(f"- {k}: {v['rows']} rows, {_p(v['command_weighted_accuracy'])}")
    leg = r["partitions"].get("legacy-development", {})
    L_ += ["", "## Legacy (point estimates; one environment)", ""]
    for name, s in sorted(leg.get("systems", {}).items()):
        L_.append(f"- {name}: {_p(s['estimate']['command_weighted'])}")
    return "\n".join(L_) + "\n"
