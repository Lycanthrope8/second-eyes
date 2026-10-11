"""A2.6b: the development preparation (ScanNet; D104 requests in both formats and both views).

**Two freezes** (ChatGPT's review of `eccd18c`, correction 2):

- **`freeze`** fixes, before any preparation:
  - the official-first partition and every group's assignment;
  - the cap, the eligibility rules and the sampling procedure, with their salts;
  - the desktop subset's rule;
  - the analysis plan, with the bootstrap's literal seed.
- **`prepare`** reads that frozen specification and builds the requests. Its request manifest, sampling tables and
  desktop subset are hashed before any inference.

**The chain per scene** is the accepted one:

1. A2.2a's import, under the scene's release identity (`release.import_release_scene`);
2. A2.2c's selection audit;
3. A2.2d's materialization and measurement.

Each runs with `sample_only=False`. Only development scenes and the legacy group are read: no other scene's files are
read from the zip.

**Eligibility** is A2.3a's accepted rule (both views materialized; all four view-format renderings succeeded), plus:

- 3 to 10 selected objects in both views;
- a unique target (correction 1).

Every exclusion keeps its reason.

**Correction 1:** a parent's annotation entries are resolved, never taken in entry order.

- **The target:** one target over all entries gives `unique_target`. Several targets give `conflicting_targets`, which
  is quarantined from unique-target scoring.
- **The parsing label:** relation, relation type and anchor sets are compared separately, and give `consistent` or
  `disagreeing`. A unique target is not a unique parsing label.
- **The stratum:** the relation when the entries agree on it; otherwise the explicit stratum `relation_disagreement`.
- Nothing is called semantic ambiguity.

**Sampling (correction 3).** Per environment, strata are filled round-robin, in ascending stratum name, one parent per
nonempty stratum per round, up to the cap. Within a stratum, parents are taken by ascending
SHA-256(salt LF environment LF stratum LF parent ID). N (eligible) and n (sampled) are recorded per environment and
stratum. Environments with nothing eligible are reported, never replaced.

**The requests** mirror A2.5's D104 golden builder (`quest/prompt_goldens._golden`) with the format as a parameter:

- the accepted serializer;
- stable choice letters over the object IDs in sorted order, with K last;
- the protocol's prompt layout;
- the pinned tokenizer;
- the context ceiling;
- the two token boundaries.

A2.5's code is not changed.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import importlib
import json
import multiprocessing
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

from ...adapters.iref_vla import convert as V1C
from ...adapters.iref_vla import release as R

REPO = Path(__file__).resolve().parents[3]
REL = REPO / "grounding" / "relations" / "relations.v1.json"
DIRS = REPO / "grounding" / "relations" / "directions.v1.json"
TOKENIZER = REPO / "quest-app" / "Assets" / "SecondEyes" / "Models" / "qwen2.5-0.5b-instruct"
MODEL_DESCRIPTION = REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct.json"
VIEWS = ("full_inventory", "source_known_nyu")
FORMATS = ("coordinates_v2", "coordinates_relations_v2")
SPEC_ID = "a26.devprep.v1"
SAMPLING_SALT = "second-eyes/a26/sampling/v1"
DESKTOP_SALT = "second-eyes/a26/desktop/v1"
CAP = 64
DESKTOP_PARENTS = 32
MIN_OBJECTS, MAX_OBJECTS = 3, 10
BOOTSTRAP = {"replicates": 10000, "seed": 20261010, "interval": "95% percentile",
             "cluster": "development environment, paired across systems, formats and views",
             "weights": "recomputed within each replicate"}
WEIGHTS = {"command_weighted": "each sampled parent weighs N/n of its environment and stratum: an estimate over the "
                               "eligible development commands",
           "environment_weighted": "within each environment, strata reweighted to that environment's eligible relation "
                                   "proportions; environments then averaged with equal weight",
           "unweighted": "the stratified sample as drawn, labelled as such"}


class DevPrepError(ValueError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _dumps(doc) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl(rows) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _publish(out: Path, files: dict) -> Path:
    out = Path(out)
    if out.exists():
        raise DevPrepError(f"{out} exists; each run writes a new folder")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for name, data in files.items():
            f = tmp / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


# --------------------------------------------------------------------------------------------- the frozen specification
ELIGIBILITY = ["A2.3a's accepted rule: both views materialized, and all four view x format measurements rendered",
               f"{MIN_OBJECTS} to {MAX_OBJECTS} selected objects in both views (too_few_objects otherwise)",
               "a unique target over all of the parent's annotation entries (conflicting_targets are quarantined)"]


def freeze(*, inventory_dir, official_dir, out) -> dict:
    """The partition, cap, eligibility, sampling, desktop subset and analysis plan, frozen before preparation."""
    inv_bytes = (Path(inventory_dir) / "inventory.json").read_bytes()
    inv = json.loads(inv_bytes)
    if set(inv["sources"]) != {"Scannet"}:
        raise DevPrepError("A2.6 is ScanNet only; the inventory holds " + ", ".join(sorted(inv["sources"])))
    lists = R.load_official(official_dir, ["Scannet"])
    prop = R.propose_partitions(inv, official=lists)
    if prop["conflicts"]:
        raise DevPrepError(f"official lists conflict for {prop['conflicts']}")
    part = {g: a["partition"] for g, a in prop["assignment"].items()}
    dev = sorted(g for g, p in part.items() if p == "development")
    legacy = sorted(g for g, p in part.items() if p == "legacy-development")
    scenes = {g: sorted(s.split("/", 1)[1] for s in inv["groups"][g]["scenes"]) for g in dev + legacy}
    spec = {
        "format_version": 1, "record_type": "a26_frozen_specification", "spec_id": SPEC_ID, "frozen_utc": _now(),
        "inventory": {"folder": Path(inventory_dir).name, "inventory_sha256": _sha(inv_bytes),
                      "release_version": inv["release_version"], "commit": inv["commit"]},
        "official_lists": R.OFFICIAL_SPLITS["Scannet"],
        "partition": {"policy": prop["policy"], "policy_sha256": prop["policy_sha256"], "counts": prop["counts"],
                      "coverage": prop["official_coverage"], "assignment": part},
        "development_groups": dev, "legacy_groups": legacy, "scenes": scenes,
        "eligibility": ELIGIBILITY,
        "sampling": {"cap_per_environment": CAP, "salt": SAMPLING_SALT,
                     "strata": "the parent's relation when its annotation entries agree on it; otherwise "
                               "relation_disagreement",
                     "allocation": "round-robin over the environment's nonempty strata in ascending stratum name, one "
                                   "parent per stratum per round, until the cap or every stratum is exhausted",
                     "order_within_stratum": "ascending SHA-256(salt LF environment LF stratum LF parent command ID)",
                     "groups_fixed": "assigned environments are never replaced; zero-eligible ones are reported"},
        "desktop_subset": {"salt": DESKTOP_SALT, "parents": DESKTOP_PARENTS, "format": "coordinates_v2",
                           "views": list(VIEWS),
                           "rule": "the selected development parents with the lowest SHA-256(salt LF parent command ID); "
                                   "both views, coordinates_v2; a desktop Q8_0/U comparison, never Quest agreement"},
        "requests": {"formats": list(FORMATS), "views": list(VIEWS), "mapping": "D104: stable letters over the "
                     "object IDs in sorted order, K last; the inference protocol's prompt layout"},
        "analysis": {"bootstrap": BOOTSTRAP, "weights": WEIGHTS,
                     "population": "the eligible development commands, not every command of the release"},
        "closed": ["calibration", "test", "the final lab commands"], "gate_b": "no Gate B inference",
    }
    body = _dumps(spec)
    lines = ["# A2.6 frozen specification (partition, cap, sampling, analysis)", "",
             f"Specification SHA-256 `{_sha(body)}`. Inventory `{spec['inventory']['inventory_sha256'][:12]}...`.", "",
             f"Development environments: {len(dev)}; legacy: {legacy}; cap {CAP} per environment; sampling salt "
             f"`{SAMPLING_SALT}`; bootstrap seed {BOOTSTRAP['seed']}.", ""]
    lines += [f"| {k} | {v['groups']} groups | {v['scenes']} scenes |" for k, v in sorted(prop["counts"].items())]
    folder = _publish(out, {"spec.json": body, "report.md": ("\n".join(lines) + "\n").encode("utf-8")})
    return dict(spec, folder=str(folder), spec_sha256=_sha(body))


# --------------------------------------------------------------------------------------------- correction 1
def resolve(entries) -> dict:
    """A parent's annotation entries, resolved: target, parsing label and stratum, independent of entry order."""
    pay = [e["source_payload"] for e in entries]
    targets = sorted({p["target_index"] for p in pay})
    relations = sorted({p["relation"] for p in pay})
    types = sorted({p["relation_type"] for p in pay})
    anchors = sorted({tuple(sorted(a["index"] for a in p["anchors"].values())) for p in pay})
    if not entries:
        status = "no_annotation"
    else:
        status = "unique_target" if len(targets) == 1 else "conflicting_targets"
    return {"entries": len(entries), "annotation_ids": sorted(e["annotation_id"] for e in entries),
            "target_status": status, "target_indices": targets,
            "target_object_id": V1C.object_id(targets[0]) if status == "unique_target" else None,
            "parsing_label": "consistent" if len(relations) == 1 and len(types) == 1 and len(anchors) == 1 else "disagreeing",
            "relations": relations, "relation_types": types, "anchor_sets": [list(a) for a in anchors],
            "stratum": relations[0] if len(relations) == 1 else "relation_disagreement"}


# --------------------------------------------------------------------------------------------- eligibility
def _modules():
    return (importlib.import_module("grounding.subscenes.iref_vla.audit"),
            importlib.import_module("grounding.preparation.iref_vla.prepare"),
            importlib.import_module("grounding.inference.iref_vla.prepare"))


def eligibility_rows(scene_work, scene, group, partition) -> list:
    """One row per parent command of a prepared scene: eligible or not, with every reason, and correction 1's fields."""
    _, _, I = _modules()
    src = I.read_bundle(Path(scene_work) / "bundle")
    accepted = set(I.eligible_parents(src))
    ann = json.loads((Path(scene_work) / "import" / "reference_only" / "iref-annotations.json").read_text(encoding="utf-8"))
    by_cmd = {}
    for e in ann["entries"]:
        by_cmd.setdefault(e["command_id"], []).append(e)
    rows = []
    for p in src["parents"]:
        res = resolve(by_cmd.get(p, []))
        views, reasons = {}, []
        for v in VIEWS:
            r = src["index"][(p, v)]
            ids = r.get("planned_object_ids") or []
            views[v] = {"status": r["preparation_status"], "objects": len(ids), "derived_scene_id": r.get("derived_scene_id"),
                        "derived_command_id": r.get("derived_command_id"), "scene_path": r.get("scene_path"),
                        "command_path": r.get("command_path"),
                        "target_offered": res["target_object_id"] in ids if res["target_object_id"] else None}
            if r["preparation_status"] != "materialized":
                reasons.append(f"{v}: {r['preparation_status']}")
            elif len(ids) < MIN_OBJECTS:
                reasons.append(f"{v}: too_few_objects")
            elif len(ids) > MAX_OBJECTS:
                reasons.append(f"{v}: over_object_budget")
            for f in FORMATS:
                s = src["meas"][(p, v, f)]["status"]
                if s != "rendered" and r["preparation_status"] == "materialized":
                    reasons.append(f"{v}/{f}: {s}")
        if p not in accepted and not reasons:
            reasons.append("not eligible under A2.3a's rule")
        if res["target_status"] != "unique_target":
            reasons.append(res["target_status"])
        rows.append({"scene": scene, "group": group, "partition": partition, "parent_command_id": p,
                     "eligible": not reasons, "reasons": reasons, "views": views, **res})
    return rows


# --------------------------------------------------------------------------------------------- sampling (correction 3)
def _key(salt, *parts) -> str:
    return _sha("\n".join((salt,) + parts).encode("utf-8"))


def sample(rows, groups, cap=CAP, salt=SAMPLING_SALT):
    """(strata, selected): N and n per environment and stratum, and the selected parents, by the frozen procedure.
    groups lists every assigned environment, so a zero-eligible one is reported, never replaced."""
    by_group = {g: {} for g in groups}
    for r in rows:
        if r["eligible"]:
            by_group.setdefault(r["group"], {}).setdefault(r["stratum"], []).append(r)
    strata, selected = [], []
    for g in sorted(by_group):
        st = by_group[g]
        order = {s: sorted(rs, key=lambda r: _key(salt, g, s, r["parent_command_id"])) for s, rs in st.items()}
        names = sorted(order)
        n = {s: 0 for s in names}
        left = cap
        while left > 0 and any(n[s] < len(order[s]) for s in names):
            for s in names:
                if left == 0:
                    break
                if n[s] < len(order[s]):
                    n[s] += 1
                    left -= 1
        if not names:
            strata.append({"group": g, "stratum": None, "N": 0, "n": 0})
        for s in names:
            strata.append({"group": g, "stratum": s, "N": len(order[s]), "n": n[s]})
            selected += [dict(r, sampling_rank=k) for k, r in enumerate(order[s][:n[s]])]
    return strata, selected


# --------------------------------------------------------------------------------------------- the chain per scene
ATTEMPTS = 3
PAUSES_S = (2.0, 6.0)


def _transient(e) -> bool:
    """A Windows file lock at publication (antivirus or indexer holding a new file): a PermissionError, possibly
    wrapped by a stage's output error. Data errors are never transient."""
    return isinstance(e, PermissionError) or "PermissionError" in str(e) or "WinError 5" in str(e)


def _stage(fn, out: Path, attempts: list, name: str, sleep=time.sleep):
    """Runs one stage; retries it only when it failed transiently and published nothing."""
    for k in range(1, ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:   # noqa: BLE001
            if k == ATTEMPTS or not _transient(e) or out.exists():
                raise
            attempts.append({"stage": name, "attempt": k, "error": f"{type(e).__name__}: {e}"[:300]})
            sleep(PAUSES_S[min(k, len(PAUSES_S)) - 1])


def _scene_job(job, tokenizer=None, sleep=time.sleep):
    """Import, audit and prepare one scene into its own work folder. A failure is returned, never raised, so every
    scene is accounted for; transient file locks are retried (each retry recorded)."""
    scene, group, partition, files, expected, vocab, work, tokenizer_dir, model_description = job
    A, P, _ = _modules()
    w = Path(work)
    retries = []
    try:
        _stage(lambda: R.import_release_scene("Scannet", scene, files, vocab, w / "import", partition=partition,
                                              expected_sha256=expected), w / "import", retries, "import", sleep)
        mi, ro = w / "import" / "model_inputs", w / "import" / "reference_only"
        args = dict(scene=mi / "scene.annotated.json", commands=mi / "commands", category_map=mi / "category-map.json",
                    inventory_views=ro / "inventory-views.json")
        _stage(lambda: A.run_audit(**args, out=w / "audit", sample_only=False), w / "audit", retries, "audit", sleep)
        _stage(lambda: P.run_preparation(**args, selection_audit=w / "audit", relation_config=REL, direction_config=DIRS,
                                         tokenizer_dir=tokenizer_dir, model_description=model_description,
                                         out=w / "bundle", tokenizer=tokenizer, sample_only=False),
               w / "bundle", retries, "preparation", sleep)
        return {"scene": scene, "group": group, "partition": partition, "status": "prepared", "retries": retries}
    except Exception as e:   # noqa: BLE001 - an accepted stage's refusal is recorded against its scene
        return {"scene": scene, "group": group, "partition": partition, "status": "failed", "retries": retries,
                "error": f"{type(e).__name__}: {e}"[:2000]}


# --------------------------------------------------------------------------------------------- D104 requests
def request_record(scene, command, cmap, proto, tok, fmt, work_cap=None) -> dict:
    """One D104 request: quest/prompt_goldens._golden's construction with the format as a parameter. The relations
    format takes A2.2d's relation work cap (work_cap: the bundle's recorded cap), as A2.2d and A2.3d pass it."""
    from ...inference.iref_vla.choices import build_prompt, choice_mapping
    from ...quest import replay_inputs as RI
    from ...quest.replay_bundle import boundary_before_command_line, boundary_before_last_token
    from ...serialization.serializer import serialize
    res = serialize(scene, command, format=fmt, relation_config_path=REL, direction_config_path=DIRS, category_maps=[cmap],
                    max_relation_work_units=None if fmt == FORMATS[0] else work_cap)
    if res.status != "ok":
        raise DevPrepError(f"the serializer did not render the request ({res.status})")
    doc = res.document
    ids = [o["object_id"] for o in sorted(scene["objects"], key=lambda o: o["object_id"])]
    mapping = choice_mapping(ids, proto)
    codes, targets = [m[0] for m in mapping], [m[1] for m in mapping]
    code_ids = [proto["code_token_ids"][c] for c in codes]
    prompt = build_prompt(proto, mapping, doc)
    token_ids = [int(x) for x in tok.encode(prompt)]
    if len(token_ids) + proto["continuation_tokens"] > proto["context_limit_tokens"]:
        raise DevPrepError(f"{len(token_ids)} tokens leave no room within the context limit")
    last = boundary_before_last_token(tok, prompt, token_ids)
    cmd = boundary_before_command_line(tok, prompt, token_ids, proto, mapping)
    pb = prompt.encode("utf-8")
    return {"document": doc, "document_sha256": _sha(doc.encode("utf-8")), "prompt_sha256": _sha(pb),
            "prompt_bytes": len(pb), "token_ids": token_ids, "input_tokens": len(token_ids),
            "token_ids_sha256": RI.token_ids_sha256(token_ids), "codes": codes, "choice_object_ids": targets,
            "code_token_ids": code_ids, "mapping_sha256": RI.mapping_sha256(codes, targets, code_ids),
            "keep_before_last_token": last["keep_tokens"], "keep_before_command_line": cmd["keep_tokens"]}


def request_id(parent, view, fmt) -> str:
    return "a26-" + _sha(f"{parent}\n{view}\n{fmt}".encode("utf-8"))[:24]


def build_requests(selected, work, proto, tok):
    """(requests, failures): four requests per selected parent. A parent with any failed request is left out whole,
    so the manifest stays matched across formats and views; its failures are listed."""
    requests, failures = [], []
    cmaps = {}
    for r in sorted(selected, key=lambda r: (r["partition"], r["group"], r["stratum"], r["sampling_rank"])):
        b = Path(work) / r["scene"] / "bundle"
        if r["scene"] not in cmaps:
            cmaps[r["scene"]] = (json.loads((b / "model_records" / "category-map.json").read_text(encoding="utf-8")),
                                 json.loads((b / "summary.json").read_text(encoding="utf-8"))["work_cap"])
        mine, bad = [], []
        for v in VIEWS:
            vv = r["views"][v]
            scene = json.loads((b / vv["scene_path"]).read_text(encoding="utf-8"))
            command = json.loads((b / vv["command_path"]).read_text(encoding="utf-8"))
            for f in FORMATS:
                rid = request_id(r["parent_command_id"], v, f)
                try:
                    rec = request_record(scene, command, cmaps[r["scene"]][0], proto, tok, f, cmaps[r["scene"]][1])
                except Exception as e:   # noqa: BLE001 - recorded with its parent
                    bad.append({"request_id": rid, "parent_command_id": r["parent_command_id"], "view": v, "format": f,
                                "error": f"{type(e).__name__}: {e}"[:500]})
                    continue
                mine.append({"request_id": rid, "partition": r["partition"], "group": r["group"], "scene": r["scene"],
                             "stratum": r["stratum"], "parent_command_id": r["parent_command_id"], "view": v, "format": f,
                             "derived_scene_id": vv["derived_scene_id"], "derived_command_id": vv["derived_command_id"],
                             "purpose": "operational", "cache_mode": "Off", **rec})
        if bad:
            failures += bad
        else:
            requests += mine
    return requests, failures


def desktop_subset(requests, n_parents=DESKTOP_PARENTS, salt=DESKTOP_SALT) -> list:
    parents = sorted({r["parent_command_id"] for r in requests if r["partition"] == "development"}, key=lambda p: _key(salt, p))
    chosen = set(parents[:n_parents])
    return sorted(r["request_id"] for r in requests if r["parent_command_id"] in chosen and r["format"] == "coordinates_v2")


# --------------------------------------------------------------------------------------------- the preparation
INDEX_FIELDS = ("request_id", "partition", "group", "scene", "stratum", "parent_command_id", "view", "format",
                "derived_scene_id", "derived_command_id", "document_sha256", "prompt_sha256", "prompt_bytes", "input_tokens",
                "token_ids_sha256", "mapping_sha256", "codes", "keep_before_last_token", "keep_before_command_line")


def prepare(*, spec, zip_path, inventory_dir, vocabulary, out, work, workers=1, tokenizer_dir=TOKENIZER,
            model_description=MODEL_DESCRIPTION, review=None, tokenizer=None, progress=print) -> dict:
    """The frozen specification applied: chain per scene, eligibility, sampling, D104 requests, desktop subset. review
    names files copied beside the manifest for checking (receipt, inventory summary and proposal)."""
    from ...inference.iref_vla.protocol import load_protocol
    from . import _tokens
    spec_bytes = Path(spec).read_bytes()
    S = json.loads(spec_bytes)
    if S.get("record_type") != "a26_frozen_specification" or S.get("spec_id") != SPEC_ID:
        raise DevPrepError(f"{spec} is not an {SPEC_ID} frozen specification")
    inv_bytes = (Path(inventory_dir) / "inventory.json").read_bytes()
    if _sha(inv_bytes) != S["inventory"]["inventory_sha256"]:
        raise DevPrepError("the inventory is not the one the specification froze")
    inv = json.loads(inv_bytes)
    expected = {r["scene"]: r["input_sha256"] for r in inv["scenes"] if r["retained"]}
    texts = {r["scene"]: r["texts"] for r in inv["scenes"] if r["retained"]}
    vocab = Path(vocabulary).read_bytes()
    if _sha(vocab) != R.VOCABULARY["sha256"]:
        raise DevPrepError("the vocabulary is not the pinned NYU_Object_Classes.csv")
    work = Path(work)
    if work.exists():
        raise DevPrepError(f"{work} exists; each preparation uses a new work folder")
    groups = S["development_groups"] + S["legacy_groups"]
    wanted = {s: (g, S["partition"]["assignment"][g]) for g in groups for s in S["scenes"][g]}
    found, _, _ = R.read_source(zip_path, "Scannet", only=set(wanted))
    jobs = [(s, g, p, found[s], expected[s], vocab, work / s, str(tokenizer_dir), str(model_description))
            for s, (g, p) in sorted(wanted.items()) if s in found and texts.get(s, 0) > 0]
    results = [{"scene": s, "group": g, "partition": p, "status": "failed", "error": "not in the zip"}
               for s, (g, p) in sorted(wanted.items()) if s not in found]
    results += [{"scene": s, "group": g, "partition": p, "status": "no_statements",
                 "error": "the release has no statement for this scene (the inventory's parent texts: 0)"}
                for s, (g, p) in sorted(wanted.items()) if s in found and texts.get(s, 0) == 0]
    work.mkdir(parents=True)
    if workers > 1 and tokenizer is None:
        ctx = multiprocessing.get_context("spawn")   # as on Windows: workers re-import, nothing is inherited
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for k, res in enumerate(ex.map(_scene_job, jobs), 1):
                results.append(res)
                progress(f"  {k}/{len(jobs)} {res['scene']}: {res['status']}")
    else:
        for k, j in enumerate(jobs, 1):
            res = _scene_job(j, tokenizer=tokenizer)
            results.append(res)
            progress(f"  {k}/{len(jobs)} {res['scene']}: {res['status']}")
    results.sort(key=lambda r: r["scene"])
    rows = []
    for res in results:
        if res["status"] == "prepared":
            rows += eligibility_rows(work / res["scene"], res["scene"], res["group"], res["partition"])
    strata, selected = sample(rows, groups, S["sampling"]["cap_per_environment"], S["sampling"]["salt"])
    proto = load_protocol()
    tok = tokenizer if tokenizer is not None else _tokens.load(tokenizer_dir)
    requests, failures = build_requests(selected, work, proto, tok)
    built = {r["parent_command_id"] for r in requests}
    desktop = desktop_subset(requests, S["desktop_subset"]["parents"], S["desktop_subset"]["salt"])
    for r in strata:
        r["partition"] = S["partition"]["assignment"][r["group"]]
    sampling = {"format_version": 1, "record_type": "a26_sampling", "spec_sha256": _sha(spec_bytes),
                "procedure": S["sampling"], "strata": strata,
                "selected": [{"group": r["group"], "partition": r["partition"], "stratum": r["stratum"],
                              "sampling_rank": r["sampling_rank"], "parent_command_id": r["parent_command_id"],
                              "scene": r["scene"], "requests_built": r["parent_command_id"] in built} for r in selected]}
    targets = [{"parent_command_id": r["parent_command_id"], "group": r["group"], "partition": r["partition"],
                "target_object_id": r["target_object_id"], "target_offered": {v: r["views"][v]["target_offered"] for v in VIEWS},
                "parsing_label": r["parsing_label"], "relations": r["relations"], "relation_types": r["relation_types"],
                "anchor_sets": r["anchor_sets"], "entries": r["entries"], "annotation_ids": r["annotation_ids"]} for r in selected]
    reasons = Counter(x.split(": ")[-1] for r in rows for x in r["reasons"])
    by_part = lambda rs, key: dict(Counter(r[key] for r in rs))   # noqa: E731
    counts = {
        "scenes": dict(Counter(r["status"] for r in results)),
        "scenes_retried": sorted(r["scene"] for r in results if r.get("retries")),
        "parents": by_part(rows, "partition"), "eligible_parents": by_part([r for r in rows if r["eligible"]], "partition"),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "target_status": dict(Counter(r["target_status"] for r in rows)),
        "parsing_label": dict(Counter(r["parsing_label"] for r in rows if r["target_status"] == "unique_target")),
        "selected_parents": by_part(selected, "partition"),
        "parents_with_all_four_requests": len(built), "requests": len(requests), "request_failures": len(failures),
        "zero_eligible_environments": sorted({s["group"] for s in strata if s["N"] == 0}),
        "contributing_environments": len({s["group"] for s in strata if s["n"] > 0}),
        "desktop_requests": len(desktop)}
    files = {
        "spec.json": spec_bytes, "scenes.jsonl": _jsonl(results), "strata.jsonl": _jsonl(strata),
        "sampling.json": _dumps(sampling),
        "requests.jsonl": _jsonl(requests), "requests-index.jsonl": _jsonl([{k: r[k] for k in INDEX_FIELDS} for r in requests]),
        "request-failures.jsonl": _jsonl(failures),
        "desktop.json": _dumps({"format_version": 1, "record_type": "a26_desktop_subset", "rule": S["desktop_subset"],
                                "request_ids": desktop}),
        "reference_only/eligibility.jsonl": _jsonl(rows), "reference_only/targets.jsonl": _jsonl(targets)}
    for name, path in (review or {}).items():
        files[f"review/{name}"] = Path(path).read_bytes()
    hashes = {"spec": _sha(spec_bytes), "requests": _sha(files["requests.jsonl"]), "sampling": _sha(files["sampling.json"]),
              "desktop": _sha(files["desktop.json"])}
    manifest = {"format_version": 1, "record_type": "a26_preparation_manifest", "spec_id": SPEC_ID, "prepared_utc": _now(),
                "hashes": hashes, "counts": counts, "files": {n: _sha(d) for n, d in sorted(files.items())},
                "code": {p.relative_to(REPO).as_posix(): _sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "status": "prepared; frozen before inference; awaiting review"}
    files["manifest.json"] = _dumps(manifest)
    files["report.md"] = report(manifest).encode("utf-8")
    folder = _publish(out, files)
    return dict(manifest, folder=str(folder))


def report(m) -> str:
    c = m["counts"]
    L = ["# A2.6b preparation (frozen before inference; awaiting review)", "",
         f"Specification `{m['hashes']['spec'][:12]}...`; requests `{m['hashes']['requests'][:12]}...`; sampling "
         f"`{m['hashes']['sampling'][:12]}...`; desktop subset `{m['hashes']['desktop'][:12]}...`.", "",
         f"- Scenes: {c['scenes']}; retried after a transient file lock: {c['scenes_retried']}",
         f"- Parents: {c['parents']}; eligible: {c['eligible_parents']}",
         f"- Exclusion reasons: {c['exclusion_reasons']}",
         f"- Target status: {c['target_status']}; parsing label (unique targets): {c['parsing_label']}",
         f"- Selected parents: {c['selected_parents']}; with all four requests: {c['parents_with_all_four_requests']}",
         f"- Requests: {c['requests']}; request failures: {c['request_failures']}; desktop subset: {c['desktop_requests']}",
         f"- Contributing environments: {c['contributing_environments']}; zero-eligible: {c['zero_eligible_environments']}",
         "", "Final counts, not planning estimates. Calibration, test and the lab commands stay closed; no Gate B."]
    return "\n".join(L) + "\n"
