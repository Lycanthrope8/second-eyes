"""A2.4's training preparation (D109): two commands, each writing a new folder.

**`freeze`** fixes, before anything is prepared or fitted:

- the inputs, by hash: A2.6's frozen specification (`9fa98001...`, whose partition assignment is the only source of
  training environments), the inventory bytes (`87bacea9...f831`), the language and the training settings;
- the environments: every training environment (never legacy, development, calibration or test), ordered by
  SHA-256(salt LF environment ID); the first 120 fit, the next 25 are the inner validation, the rest is the ordered
  reserve. Physical environments keep their rescans together (A2.6a's groups);
- the cohort caps (9 and 8 per environment), minimums (900 and 180 parents) and the reserve rule: when a cohort falls
  short after preparation, the next 10 reserve environments in order join the short cohort (fitting first), at most 4
  times; a remaining shortfall is reported for the Stage A review, never made up by weakening a check.

**`prepare`** applies it:

1. per round, the round's scans are checked against the training authorization (release.TrainingAuthorization:
   A2.6's frozen assignment, this specification's cohorts and scans, the inventory bytes) before any member is read;
   then A2.4's own wrapper runs the shared A2.2a import, A2.2c audit and A2.2d materialization per scene, in spawned
   worker processes, with A2.6b's transient-lock retry;
2. A2.6b's eligibility, then `labels.label_row` in the same worker: the label, its contract, the rule-parser
   diagnostic, and the resolver's support per view;
3. per environment, A2.6b's sampling with one change, declared and frozen: the round-robin's stratum order is salted
   per environment. When a cap is smaller than an environment's number of nonempty strata, name order systematically
   favours the earlier names; salting reduces that preference without guaranteeing every relation is chosen;
4. the examples: for each sampled parent, D104 requests in both formats for each supported view (A2.6b's
   `request_record`, the benchmark's construction), and one parser example; every token boundary checked, nothing
   truncated;
5. counts by parent and by view, before and after filtering; the bounded audit sample; the pilot specification draft;
   a hashed manifest. Answers and labels sit in `reference_only/` and in the training targets only, never in a prompt.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import json
import multiprocessing
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..adapters.iref_vla import release as R
from ..preparation.iref_vla_dev import dev as DEV
from . import language as L

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SETTINGS_PATH = HERE / "settings.v1.json"
SPEC_ID = "a24.trainprep.v1"
FROZEN = {"a26_spec_sha256": "9fa980010eb46f25bb471ede64a37b366d812adb654e23e8370edae450c538b0",
          "inventory_sha256": "87bacea9455064dc20fc615a0b70a0c72a81caf95ad2c5ba1ec263a8d708f831"}
ENV_SALT = "second-eyes/a24/environments/v1"
SAMPLING_SALT = "second-eyes/a24/sampling/v1"
STRATUM_SALT = "second-eyes/a24/strata/v1"
AUDIT_SALT = "second-eyes/a24/audit/v1"
COHORTS = ("fitting", "validation")
VIEWS, FORMATS = DEV.VIEWS, DEV.FORMATS
AUDIT_PER_CELL = 3


class PrepError(ValueError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _key(salt, *parts) -> str:
    return _sha("\n".join((salt,) + parts).encode("utf-8"))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _publish(out, files) -> Path:
    """A2.6b's publication, retried when a transient Windows lock (antivirus or indexer) refuses it; each attempt writes a
    new temporary folder, so a retry loses nothing. Data errors and an existing output are never retried."""
    for k in range(1, DEV.ATTEMPTS + 1):
        try:
            return DEV._publish(out, files)
        except Exception as e:   # noqa: BLE001
            if k == DEV.ATTEMPTS or not DEV._transient(e) or Path(out).exists():
                raise
            time.sleep(DEV.PAUSES_S[min(k, len(DEV.PAUSES_S)) - 1])


def load_settings(path=SETTINGS_PATH) -> dict:
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    if s.get("record_type") != "a24_training_settings":
        raise PrepError(f"{path} is not an A2.4 settings file")
    return s


def code_hashes() -> dict:
    return {p.relative_to(REPO).as_posix(): _sha(p.read_bytes()) for p in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.json"))}


def parser_prompt(text: str, settings) -> str:
    p = settings["parser_prompt"]
    w = p["wrapper"]
    return w["before_system"] + p["system_message"] + w["between"] + text + w["after_user"]


# --------------------------------------------------------------------------------------------- freeze
def freeze(*, a26_spec, inventory_dir, out, frozen=FROZEN, settings_path=SETTINGS_PATH) -> dict:
    a26_bytes = Path(a26_spec).read_bytes()
    if _sha(a26_bytes) != frozen["a26_spec_sha256"]:
        raise PrepError("not A2.6's frozen specification")
    A26 = json.loads(a26_bytes)
    inv_bytes = (Path(inventory_dir) / "inventory.json").read_bytes()
    if _sha(inv_bytes) != frozen["inventory_sha256"] or A26["inventory"]["inventory_sha256"] != frozen["inventory_sha256"]:
        raise PrepError("the inventory is not the frozen one (bound by its bytes' hash, not its folder name)")
    inv = json.loads(inv_bytes)
    settings = load_settings(settings_path)
    c = settings["cohort"]
    asg = A26["partition"]["assignment"]
    training = sorted((g for g, p in asg.items() if p == "training"), key=lambda g: _key(ENV_SALT, g))
    if any(g in R.LEGACY_GROUPS for g in training):
        raise PrepError("a legacy group is assigned to training")
    retained = {r["scene"] for r in inv["scenes"] if r["retained"]}
    scenes = {g: sorted(s.split("/", 1)[1] for s in inv["groups"][g]["scenes"] if s.split("/", 1)[1] in retained)
              for g in training}
    nf, nv = c["fitting_environments"], c["validation_environments"]
    spec = {"format_version": 1, "record_type": "a24_training_specification", "spec_id": SPEC_ID, "decision": "D109",
            "frozen_utc": _now(),
            "inputs": {"a26_spec_sha256": _sha(a26_bytes), "inventory_sha256": _sha(inv_bytes),
                       "inventory_folder_at_freeze": Path(inventory_dir).name,
                       "language": {"id": L.LANGUAGE_ID, "sha256": L.spec_sha256()},
                       "settings": {"id": settings["settings_id"], "sha256": _sha(Path(settings_path).read_bytes())},
                       "code": code_hashes()},
            "environments": {"order": f"ascending SHA-256(salt LF environment ID), salt {ENV_SALT!r}",
                             "source": "A2.6's frozen assignment: partition 'training' only; legacy, development, "
                                       "calibration and test are never read",
                             "fitting": training[:nf], "validation": training[nf:nf + nv],
                             "reserve": training[nf + nv:], "scenes": scenes},
            "cohorts": {"fitting_cap": c["fitting_cap_per_environment"], "validation_cap": c["validation_cap_per_environment"],
                        "fitting_minimum": c["fitting_minimum"], "validation_minimum": c["validation_minimum"],
                        "reserve_batch": c["reserve_batch_environments"], "reserve_batches_at_most": c["reserve_batches_at_most"],
                        "reserve_rule": "after preparation, a cohort below its minimum takes the next reserve batch in "
                                        "order (fitting first when both are short); a shortfall left after the last batch "
                                        "is reported for review"},
            "sampling": {"strata": "the parent's relation (consistent labels only)",
                         "allocation": "round-robin over the environment's nonempty strata, in ascending "
                                       f"SHA-256({STRATUM_SALT!r} LF environment LF stratum), one parent per stratum per "
                                       "round, up to the cap",
                         "order_within_stratum": f"ascending SHA-256({SAMPLING_SALT!r} LF environment LF stratum LF "
                                                 "parent command ID)",
                         "difference_from_a26": "the stratum order is salted per environment instead of by name: when "
                                                "a cap is smaller than the number of an environment's nonempty strata, "
                                                "name order systematically favours the earlier names. Salting reduces "
                                                "that preference; it does not guarantee every relation is selected, and "
                                                "the A2.6 benchmark's own sampling is unchanged",
                         "reported": "eligible N and selected n for every environment and stratum, zero-selected "
                                     "strata included (strata.jsonl)"},
            "labels": {"policy": "D109: positive references only; incompatible views excluded and reported; no K labels; "
                                 "annotation, mapping and rule-parser disagreements quarantined with their reasons",
                       "cohort": "eligible under A2.6b; a consistent parsing label; a v1 label; a valid query; the rule "
                                 "parser agreeing; the source-known view supported",
                       "direct_rows": "each view of a cohort parent whose own check passes (target offered, resolved to "
                                      "the source target)", "parser": "one example per cohort parent"},
            "closed": ["development (except as the frozen A2.6 benchmark, at evaluation)", "calibration", "test",
                       "the final lab commands"], "gate_b": "no Gate B inference"}
    body = (json.dumps(spec, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    report = (f"# A2.4 training specification\n\nSpecification SHA-256 `{_sha(body)}`. Training environments "
              f"{len(training)}: fitting {len(spec['environments']['fitting'])}, inner validation "
              f"{len(spec['environments']['validation'])}, reserve {len(spec['environments']['reserve'])}.\n")
    folder = _publish(Path(out), {"spec.json": body, "report.md": report.encode("utf-8")})
    return dict(spec, folder=str(folder), spec_sha256=_sha(body))


# --------------------------------------------------------------------------------------------- one scene, in a worker
def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _chain(job, sleep=time.sleep) -> dict:
    """A2.4's orchestration of the shared components for one training scene: A2.2a's import under the training
    authorization (release.import_release_scene re-checks the scene), A2.2c's audit and A2.2d's materialization, with
    A2.6b's transient-lock retry (DEV._stage). dev.py's own scene job is not used and not changed."""
    scene, group, cohort, files, expected, vocab, work, tokenizer_dir, model_description, auth = job[:10]
    A, PR, _ = DEV._modules()
    w, retries = Path(work), []
    try:
        DEV._stage(lambda: R.import_release_scene("Scannet", scene, files, vocab, w / "import", partition="training",
                                                  expected_sha256=expected, training=auth), w / "import", retries,
                   "import", sleep)
        mi, ro = w / "import" / "model_inputs", w / "import" / "reference_only"
        args = dict(scene=mi / "scene.annotated.json", commands=mi / "commands", category_map=mi / "category-map.json",
                    inventory_views=ro / "inventory-views.json")
        DEV._stage(lambda: A.run_audit(**args, out=w / "audit", sample_only=False), w / "audit", retries, "audit", sleep)
        DEV._stage(lambda: PR.run_preparation(**args, selection_audit=w / "audit", relation_config=DEV.REL,
                                              direction_config=DEV.DIRS, tokenizer_dir=tokenizer_dir,
                                              model_description=model_description, out=w / "bundle", tokenizer=None,
                                              sample_only=False), w / "bundle", retries, "preparation", sleep)
        return {"scene": scene, "group": group, "partition": "training", "cohort": cohort, "status": "prepared",
                "retries": retries}
    except Exception as e:   # noqa: BLE001 - an accepted stage's refusal is recorded against its scene
        return {"scene": scene, "group": group, "partition": "training", "cohort": cohort, "status": "failed",
                "retries": retries, "error": f"{type(e).__name__}: {e}"[:2000]}


def _sha_file(p) -> str:
    return _sha(Path(p).read_bytes())


def _tree(folder) -> dict:
    f = Path(folder)
    return {p.relative_to(f).as_posix(): _sha_file(p) for p in sorted(f.rglob("*")) if p.is_file()}


def _commands_sha(folder) -> str:
    """A2.2c's commands hash: SHA-256 of the sorted lines '<command_id> <file SHA-256>\\n'."""
    lines = sorted(f"{_read(p)['command_id']} {_sha_file(p)}\n" for p in Path(folder).glob("*.json"))
    return _sha("".join(lines).encode("utf-8"))


def _verify_reuse(job) -> dict:
    """Whether an earlier attempt's chain output for one scan is exactly what this run would write, from the scan's own
    records: the three stages complete; today's importer reproduces the import byte for byte from the frozen source
    files (the inputs and the importer's code); the audit's and the materialization's recorded input hashes are this
    import's, their recorded code, library, model-description, tokenizer and runtime identities are today's, and their
    outputs are unchanged (the accepted verify_bundle for the bundle). Nothing is written to the scan's folder."""
    from ..preparation.iref_vla import prepare as A22D
    from ..preparation.iref_vla import tokens as T22
    from ..preparation.iref_vla_dev import _tokens
    scene, group, cohort, files, expected, vocab, work, tokenizer_dir, model_description, auth = job[:10]
    w = Path(work)
    base = {"scene": scene, "group": group, "partition": "training", "cohort": cohort, "reused": True}
    missing = [st for st in ("import", "audit", "bundle") if not (w / st).is_dir()]
    if missing:
        return dict(base, status="reuse_refused", reuse_checks={"complete": False},
                    problems=["complete", f"{', '.join(missing)} missing: the chain did not finish this scan"])
    checks, problems = {"complete": True}, []
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            again = Path(td) / "import"
            DEV._stage(lambda: R.import_release_scene("Scannet", scene, files, vocab, again, partition="training",
                                                      expected_sha256=expected, training=auth), again, [], "re-import")
            checks["import_reproduced_byte_for_byte"] = _tree(again) == _tree(w / "import")
        mi, ro = w / "import" / "model_inputs", w / "import" / "reference_only"
        now = {"scene_sha256": _sha_file(mi / "scene.annotated.json"), "category_map_sha256": _sha_file(mi / "category-map.json"),
               "inventory_views_sha256": _sha_file(ro / "inventory-views.json"), "commands_sha256": _commands_sha(mi / "commands")}
        am, bm = _read(w / "audit" / "manifest.json"), _read(w / "bundle" / "manifest.json")
        same_code = lambda code: all((REPO / p).is_file() and _sha_file(REPO / p) == h for p, h in code.items())   # noqa: E731
        checks["audit_outputs_unchanged"] = all(_sha_file(w / "audit" / n) == h for n, h in am["outputs"].items())
        checks["audit_inputs_are_this_import"] = all(am["inputs"].get(k) == v for k, v in now.items())
        checks["audit_code_unchanged"] = same_code(am["code"])
        issues = A22D.verify_bundle(w / "bundle")
        problems += [f"bundle: {x}" for x in issues[:5]]
        checks["bundle_verified_by_the_accepted_reader"] = not issues
        checks["bundle_inputs_are_this_import"] = all(bm["inputs"].get(k) == v for k, v in now.items())
        checks["bundle_code_unchanged"] = same_code(bm["code"])
        checks["bundle_libraries_unchanged"] = (bm["libraries"]["relation_config"]["sha256"] == _sha_file(DEV.REL)
                                                and bm["libraries"]["direction_config"]["sha256"] == _sha_file(DEV.DIRS))
        checks["model_description_unchanged"] = bm["model_description"]["sha256"] == _sha_file(model_description)
        tok = _tokens.load(tokenizer_dir)
        checks["tokenizer_unchanged"] = (bm["tokenizer"]["files"] == T22.PINNED_FILES
                                         and bm["tokenizer"]["identity"] == tok.identity)
        rt = dict(A22D.runtime(), **(getattr(tok, "versions", None) or {}))
        checks["runtime_unchanged"] = all(bm["runtime"].get(k) == v for k, v in rt.items())
    except Exception as e:   # noqa: BLE001 - an unreadable record refuses the reuse; it is never repaired
        problems.append(f"{type(e).__name__}: {e}"[:500])
        checks["records_readable"] = False
    problems = [k for k, v in checks.items() if not v] + problems
    return dict(base, status="prepared" if not problems else "reuse_refused", reuse_checks=checks, problems=problems)


def _scene_rows(job):
    """The chain (or, for a verified earlier scan, its reuse), then eligibility and label rows; a failure is returned
    with its scene, never raised."""
    from . import labels as LB
    res = _verify_reuse(job) if len(job) > 10 and job[10] else _chain(job)
    scene, group, partition, work = res["scene"], res["group"], res["partition"], Path(job[6])
    if res["status"] != "prepared":
        return res, [], []
    try:
        elig = DEV.eligibility_rows(work, scene, group, partition)
        spec, checker = L.load_spec(), LB.Checker(L.load_spec())
        b = work / "bundle"
        cmap = _read(b / "model_records" / "category-map.json")
        ann = _read(work / "import" / "reference_only" / "iref-annotations.json")
        by_cmd = defaultdict(list)
        for e in ann["entries"]:
            by_cmd[e["command_id"]].append(e)
        rows = []
        for r in elig:
            p = r["parent_command_id"]
            text = _read(work / "import" / "model_inputs" / "commands" / f"{p}.json")["text"]
            views = {}
            for v in VIEWS:
                vv = r["views"][v]
                views[v] = (_read(b / vv["scene_path"]), _read(b / vv["command_path"])) \
                    if vv["status"] == "materialized" and vv["scene_path"] else None
            row = LB.label_row(r, by_cmd.get(p, []), text, views, cmap, checker)
            row["annotation"] = [{"relation": e["source_payload"]["relation"],
                                  "relation_type": e["source_payload"]["relation_type"],
                                  "target_class": e["source_payload"]["target_class"],
                                  "target_color_used": e["source_payload"].get("target_color_used"),
                                  "target_size_used": e["source_payload"].get("target_size_used"),
                                  "anchors": {k: [a["class"], a.get("color_used"), a.get("size_used")]
                                              for k, a in sorted(e["source_payload"]["anchors"].items())}}
                                 for e in by_cmd.get(p, [])]
            row["view_paths"] = {v: {"scene_path": r["views"][v]["scene_path"], "command_path": r["views"][v]["command_path"],
                                     "derived_scene_id": r["views"][v]["derived_scene_id"],
                                     "derived_command_id": r["views"][v]["derived_command_id"]} for v in VIEWS}
            rows.append(row)
        return res, elig, rows
    except Exception as e:   # noqa: BLE001 - recorded against its scene
        return dict(res, status="failed", error=f"labels: {type(e).__name__}: {e}"[:2000]), [], []


def sample(rows, groups, cap, salt=SAMPLING_SALT, stratum_salt=STRATUM_SALT):
    """(strata, selected) over retained rows, per environment, A2.6b's round-robin with a salted stratum order. Each
    stratum row gives N (retained), n (selected) and N_eligible (eligible under A2.6b before the label checks)."""
    by_group = {g: {} for g in groups}
    pre = Counter((r["group"], r["stratum"]) for r in rows if r.get("eligible"))
    for r in rows:
        if r["retained"] and r["group"] in by_group:
            by_group[r["group"]].setdefault(r["stratum"], []).append(r)
    strata, selected = [], []
    for g in sorted(by_group):
        order = {s: sorted(rs, key=lambda r: _key(salt, g, s, r["parent_command_id"])) for s, rs in by_group[g].items()}
        names = sorted(order, key=lambda s: _key(stratum_salt, g, s))
        n, left = {s: 0 for s in names}, cap
        while left > 0 and any(n[s] < len(order[s]) for s in names):
            for s in names:
                if left and n[s] < len(order[s]):
                    n[s] += 1
                    left -= 1
        if not names:
            strata.append({"group": g, "stratum": None, "N": 0, "n": 0, "N_eligible": 0})
        for s in names:
            strata.append({"group": g, "stratum": s, "N": len(order[s]), "n": n[s], "N_eligible": pre[(g, s)]})
            selected += [dict(r, sampling_rank=k) for k, r in enumerate(order[s][:n[s]])]
        for s in sorted({st for (gg, st) in pre if gg == g} - set(names)):
            strata.append({"group": g, "stratum": s, "N": 0, "n": 0, "N_eligible": pre[(g, s)]})
    return strata, selected


# --------------------------------------------------------------------------------------------- examples
def direct_examples(selected, work, proto, tok, cohort):
    """Two requests (both formats) per supported view of each sampled parent, with the answer letter. A parent with a
    failed request leaves out all of its direct rows, so the formats stay matched; failures are listed."""
    from ..inference.iref_vla.choices import build_prompt, check_boundary
    out, failures, cm = [], [], {}
    for r in sorted(selected, key=lambda r: (r["group"], r["stratum"], r["sampling_rank"])):
        b = Path(work) / r["scene"] / "bundle"
        if r["scene"] not in cm:
            cm[r["scene"]] = (_read(b / "model_records" / "category-map.json"), _read(b / "summary.json")["work_cap"])
        mine, bad = [], []
        for v in r["direct_views"]:
            vp = r["view_paths"][v]
            scene, command = _read(b / vp["scene_path"]), _read(b / vp["command_path"])
            for f in FORMATS:
                try:
                    rec = DEV.request_record(scene, command, cm[r["scene"]][0], proto, tok, f, cm[r["scene"]][1])
                    mapping = list(zip(rec["codes"], rec["choice_object_ids"]))
                    check_boundary(tok, build_prompt(proto, mapping, rec["document"]), mapping, proto)
                    letter = dict((o, c) for c, o in mapping)[r["target_object_id"]]
                except Exception as e:   # noqa: BLE001 - recorded with its parent
                    bad.append({"parent_command_id": r["parent_command_id"], "view": v, "format": f,
                                "error": f"{type(e).__name__}: {e}"[:500]})
                    continue
                mine.append({"key": f"{r['parent_command_id']}|{v}", "cohort": cohort, "parent_command_id":
                             r["parent_command_id"], "group": r["group"], "scene": r["scene"], "stratum": r["stratum"],
                             "view": v, "format": f, "derived_scene_id": vp["derived_scene_id"],
                             "derived_command_id": vp["derived_command_id"], "token_ids": rec["token_ids"],
                             "input_tokens": rec["input_tokens"], "token_ids_sha256": rec["token_ids_sha256"],
                             "prompt_sha256": rec["prompt_sha256"], "codes": rec["codes"],
                             "choice_object_ids": rec["choice_object_ids"], "code_token_ids": rec["code_token_ids"],
                             "mapping_sha256": rec["mapping_sha256"], "target_object_id": r["target_object_id"],
                             "answer_code": letter, "answer_token_id": proto["code_token_ids"][letter]})
        if bad:
            failures += bad
        else:
            out += mine
    return out, failures


def parser_examples(selected, settings, tok, constraint, cohort):
    out, failures = [], []
    eos = constraint.T.eos_id
    for r in sorted(selected, key=lambda r: (r["group"], r["stratum"], r["sampling_rank"])):
        prompt = parser_prompt(r["text"], settings)
        pids, oids = tok.encode(prompt), tok.encode(r["label"])
        joined = tok.encode(prompt + r["label"])
        ok, at = constraint.accepts_ids(oids + [eos])
        if joined != pids + oids or not ok:
            failures.append({"parent_command_id": r["parent_command_id"], "error": "the prompt and output do not tokenize "
                             "apart" if joined != pids + oids else f"the output's tokens leave the grammar at step {at}"})
            continue
        out.append({"key": r["parent_command_id"], "cohort": cohort, "parent_command_id": r["parent_command_id"],
                    "group": r["group"], "scene": r["scene"], "stratum": r["stratum"], "text": r["text"],
                    "label": r["label"], "prompt_ids": pids, "output_ids": oids + [eos], "input_tokens": len(pids),
                    "output_tokens": len(oids) + 1, "target_object_id": r["target_object_id"],
                    "direct_views": r["direct_views"], "view_paths": r["view_paths"]})
    return out, failures


# --------------------------------------------------------------------------------------------- the audit
def _visible(scene, codes=None) -> list:
    """The model-visible objects of a derived scene, compactly: ID, letter, category, colours, centre and size."""
    letter = {o: c for c, o in (codes or [])}
    rows = []
    for o in sorted(scene["objects"], key=lambda o: o["object_id"]):
        cat = o.get("category", {})
        g = o.get("geometry", {})
        col = (o.get("attributes", {}) or {}).get("colours", {})

        def val(f):
            return [round(x, 2) for x in f["value"]] if isinstance(f, dict) and f.get("state") == "known" else None
        rows.append({"object_id": o["object_id"], "letter": letter.get(o["object_id"]),
                     "category": (cat.get("value") or {}).get("model") if cat.get("state") == "known" else "unknown",
                     "colours": col.get("value") if col.get("state") == "known" else None,
                     "centre": val(g.get("center") or g.get("centre") or {}), "size": val(g.get("size") or {})})
    return rows


COLOUR_WORDS = tuple(L.COLOURS)
_SPEC_CACHE = []


def _spec():
    if not _SPEC_CACHE:
        _SPEC_CACHE.append(L.load_spec())
    return _SPEC_CACHE[0]


def text_order(text, label) -> str:
    """For a 'between' label: whether its two anchor categories occur in the command in the label's order, found by
    whole-word search in the text (independent of the label builder and the rule parser): 'same', 'reversed',
    'same_category' or 'not_found'."""
    p = L.parse_output(label, _spec())
    if p["relation"] != "between":
        return "n/a"
    a, b = (x["category"] for x in p["anchors"])
    if a == b:
        return "same_category"
    import re as _re

    def pos(c):
        m = _re.search(r"(?<![a-z])" + _re.escape(c) + r"(?![a-z])", text)
        return m.start() if m else None
    pa, pb = pos(a), pos(b)
    if pa is None or pb is None:
        return "not_found"
    return "same" if pa < pb else "reversed"


def colour_word_category(label) -> bool:
    p = L.parse_output(label, _spec())
    return any(o["category"] in COLOUR_WORDS or o["category"].split(" ")[0] in COLOUR_WORDS
               for o in [p["target"]] + p["anchors"])


def audit_sample(rows, work) -> list:
    """Every retained construction and every exclusion reason, up to three parents each, by salted hash."""
    from ..inference.iref_vla.choices import choice_mapping
    from ..inference.iref_vla.protocol import load_protocol
    proto = load_protocol()
    cells = defaultdict(list)
    for r in rows:
        if r["retained"]:
            p = L.parse_output(r["label"], L.load_spec())
            cell = ("retained", f"{p['relation']}{p['k'] or ''}", "target colour" if p["target"]["colour"] else "-",
                    "anchor colour" if any(a["colour"] for a in p["anchors"]) else "-",
                    "both views" if len(r["direct_views"]) == 2 else "source-known only")
            cells[cell].append(r)
            if colour_word_category(r["label"]):
                cells[("retained", "category containing a colour word")].append(r)
            if p["relation"] == "between":
                cells[("retained", "between: anchor order in text " + text_order(r["text"], r["label"]))].append(r)
        elif r["eligible"]:
            for reason in r["parent_reasons"]:
                cells[("quarantined", reason)].append(r)
        for v in VIEWS:
            vr = r["views"].get(v, {}).get("reason")
            if r["retained"] and vr:
                cells[("view excluded", v, vr)].append(r)
    picked = []
    for cell in sorted(cells):
        for r in sorted(cells[cell], key=lambda r: _key(AUDIT_SALT, "|".join(cell), r["parent_command_id"]))[:AUDIT_PER_CELL]:
            b = Path(work) / r["scene"] / "bundle"
            views = {}
            for v in VIEWS:
                vp = r["view_paths"][v]
                if not vp["scene_path"]:
                    views[v] = {"materialized": False}
                    continue
                scene = _read(b / vp["scene_path"])
                ids = sorted(o["object_id"] for o in scene["objects"])
                mp = choice_mapping(ids, proto) if len(ids) <= len(proto["object_codes"]) else []
                views[v] = {"materialized": True, "objects": _visible(scene, mp), "check": r["views"].get(v)}
            picked.append({"cell": list(cell), "parent_command_id": r["parent_command_id"], "scene": r["scene"],
                           "text": r["text"], "label": r["label"], "rule_parse": r["rule_parse"],
                           "between_text_order": text_order(r["text"], r["label"]) if r["label"] else None,
                           "rule_disagreement": r["rule_disagreement"], "target_object_id": r["target_object_id"],
                           "annotation": r.get("annotation"), "parent_reasons": r["parent_reasons"],
                           "retained": r["retained"], "views": views})
    return picked


def audit_markdown(items) -> str:
    L_ = ["# A2.4 label audit (for independent review)", "",
          "Each item: the command, the annotation's fields, the derived label, the rule parser's reading, and per view the "
          "model-visible objects (letter, category, colours, centre and size in metres) with the resolver's check. "
          "The expected target is the source's. Passing the label builder's own checks is not the claim; this sample is "
          "for a reader to check that the label means what the command says.", ""]
    for k, it in enumerate(items, 1):
        L_.append(f"## {k}. {' / '.join(it['cell'])}")
        L_.append(f"- Command: \"{it['text']}\" (`{it['parent_command_id']}`, {it['scene']})")
        if it.get("annotation"):
            L_.append(f"- Annotation: {json.dumps(it['annotation'], ensure_ascii=False)}")
        L_.append((f"- Label: `{it['label']}`" if it["label"] else "- Label: none (quarantined before labelling)")
                  + f"; rule parser: `{(it['rule_parse'] or {}).get('string')}`"
                  + (f" (differs: {', '.join(it['rule_disagreement'])})" if it["rule_disagreement"] else ""))
        L_.append(f"- Expected target: `{it['target_object_id']}`; retained: {it['retained']}"
                  + (f"; reasons: {', '.join(it['parent_reasons'])}" if it["parent_reasons"] else ""))
        for v, vv in it["views"].items():
            if not vv.get("materialized"):
                L_.append(f"- {v}: not materialized")
                continue
            c = vv["check"]
            if c is None:
                L_.append(f"- {v}: not checked (the parent was quarantined before its view checks: "
                          f"{', '.join(it['parent_reasons'])})")
            else:
                L_.append(f"- {v}: resolver {c.get('outcome')} -> {c.get('target_id')}; supported {c.get('supported')}"
                          + (f" ({c['reason']})" if c.get("reason") else ""))
            L_.append("  " + "; ".join(f"{o['letter'] or '-'} {o['object_id']} {o['category']}"
                                       + (f" {o['colours']}" if o["colours"] else "") + f" c={o['centre']} s={o['size']}"
                                       for o in vv["objects"]))
        L_.append("")
    return "\n".join(L_) + "\n"


# --------------------------------------------------------------------------------------------- prepare
def _run_scenes(envs, cohort_of, S, zip_path, auth, expected, texts, vocab, work, tokenizer_dir, model_description,
                workers, progress, reuse=False):
    """One round: the round's scans are authorized, then read from the zip, then prepared."""
    wanted = [s for g in envs for s in S["environments"]["scenes"][g]]
    found, _, _ = R.read_training_source(zip_path, auth, wanted)
    jobs, results = [], []
    for g in envs:
        for s in S["environments"]["scenes"][g]:
            if s not in found:
                results.append(({"scene": s, "group": g, "partition": "training", "status": "failed",
                                 "error": "not in the zip"}, [], []))
            elif texts.get(s, 0) == 0:
                results.append(({"scene": s, "group": g, "partition": "training", "status": "no_statements"}, [], []))
            else:
                jobs.append((s, g, cohort_of[g], found[s], expected[s], vocab, work / s, str(tokenizer_dir),
                             str(model_description), auth, bool(reuse and (work / s).exists())))
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for k, res in enumerate(ex.map(_scene_rows, jobs), 1):
                results.append(res)
                progress(f"  {k}/{len(jobs)} {res[0]['scene']}: {res[0]['status']}" + (" (reused, verified)"
                         if res[0].get("reused") and res[0]["status"] == "prepared" else ""))
    else:
        for k, j in enumerate(jobs, 1):
            res = _scene_rows(j)
            results.append(res)
            progress(f"  {k}/{len(jobs)} {res[0]['scene']}: {res[0]['status']}")
    for res in results:
        res[0]["cohort"] = cohort_of[res[0]["group"]]
        res[0]["round_authorization"] = dict(auth.digest)
    return results


def prepare(*, spec, a26_spec, zip_path, inventory_dir, vocabulary, out, work, workers=1, tokenizer_dir=DEV.TOKENIZER,
            model_description=DEV.MODEL_DESCRIPTION, settings_path=SETTINGS_PATH, reuse=False, progress=print) -> dict:
    """reuse: `work` is an earlier attempt's work folder. Each of its scans is reused only if `_verify_reuse` passes;
    a scan that fails stops the run before anything is published. Scans it lacks are prepared fresh into it."""
    from ..inference.iref_vla.protocol import load_protocol
    from ..preparation.iref_vla_dev import _tokens
    from . import decoder as DC
    started = _now()
    t0 = time.perf_counter()
    spec_bytes = Path(spec).read_bytes()
    S = json.loads(spec_bytes)
    if S.get("record_type") != "a24_training_specification" or S.get("spec_id") != SPEC_ID:
        raise PrepError(f"{spec} is not an {SPEC_ID} specification")
    inv_bytes = (Path(inventory_dir) / "inventory.json").read_bytes()
    if _sha(inv_bytes) != S["inputs"]["inventory_sha256"]:
        raise PrepError("the inventory is not the one the specification froze")
    if L.spec_sha256() != S["inputs"]["language"]["sha256"] or _sha(Path(settings_path).read_bytes()) != S["inputs"]["settings"]["sha256"]:
        raise PrepError("the language or the settings differ from those the specification froze")
    vocab = Path(vocabulary).read_bytes()
    if _sha(vocab) != R.VOCABULARY["sha256"]:
        raise PrepError("the vocabulary is not the pinned NYU_Object_Classes.csv")
    work = Path(work)
    if Path(out).exists():
        raise PrepError("the output folder must be new")
    if reuse and not work.is_dir():
        raise PrepError(f"{work}: no such work folder to reuse")
    if not reuse and work.exists():
        raise PrepError("the work folder must be new (or pass reuse to verify and reuse an earlier attempt's)")
    inv = json.loads(inv_bytes)
    expected = {r["scene"]: r["input_sha256"] for r in inv["scenes"] if r["retained"]}
    texts = {r["scene"]: r["texts"] for r in inv["scenes"] if r["retained"]}
    settings = load_settings(settings_path)
    C = S["cohorts"]
    envs = {"fitting": list(S["environments"]["fitting"]), "validation": list(S["environments"]["validation"])}
    reserve = list(S["environments"]["reserve"])
    auth = R.TrainingAuthorization(a26_spec_bytes=Path(a26_spec).read_bytes(), a24_spec_bytes=spec_bytes,
                                   inventory_bytes=inv_bytes)
    work.mkdir(parents=True, exist_ok=reuse)
    results, rounds = [], []
    todo = envs["fitting"] + envs["validation"]
    cohort_of = {g: c for c in COHORTS for g in envs[c]}
    batch = 0
    while True:
        progress(f"round {batch}: {len(todo)} environments")
        done = _run_scenes(todo, cohort_of, S, zip_path, auth, expected, texts, vocab, work, tokenizer_dir,
                           model_description, workers, progress, reuse=reuse)
        refused = [res[0] for res in done if res[0]["status"] == "reuse_refused"]
        if refused:
            raise PrepError(f"{len(refused)} scan(s) of {work} failed the reuse checks; nothing was published: "
                            + "; ".join(f"{r['scene']}: {', '.join(map(str, r['problems']))[:300]}" for r in refused[:10]))
        results += done
        rows = [r for res in results for r in res[2]]
        sel = {c: sample([r for r in rows if cohort_of.get(r["group"]) == c], envs[c],
                         C["fitting_cap"] if c == "fitting" else C["validation_cap"])[1] for c in COHORTS}
        short = [c for c in COHORTS if len(sel[c]) < C[f"{c}_minimum"]]
        rounds.append({"round": batch, "environments": list(todo), "selected": {c: len(sel[c]) for c in COHORTS},
                       "short": short})
        if not short or batch == C["reserve_batches_at_most"] or not reserve:
            break
        batch += 1
        todo, reserve = reserve[:C["reserve_batch"]], reserve[C["reserve_batch"]:]
        auth = auth.activate(todo)
        envs[short[0]] += todo
        cohort_of.update({g: short[0] for g in todo})
    for r in rows:
        r["cohort"] = cohort_of.get(r["group"])
    strata, selected = {}, {}
    for c in COHORTS:
        strata[c], selected[c] = sample([r for r in rows if r["cohort"] == c], envs[c],
                                        C["fitting_cap"] if c == "fitting" else C["validation_cap"])
    proto = load_protocol()
    tok = _tokens.load(tokenizer_dir)
    constraint = DC.Constraint(L.load_spec(), DC.TokenTable(tok._backend))
    examples, failures = {}, []
    for c in COHORTS:
        d, f1 = direct_examples(selected[c], work, proto, tok, c)
        p, f2 = parser_examples(selected[c], settings, tok, constraint, c)
        examples[c] = {"direct": d, "parser": p}
        failures += [dict(x, cohort=c, kind="direct") for x in f1] + [dict(x, cohort=c, kind="parser") for x in f2]
    audit = audit_sample(rows, work)
    counts = count(rows, strata, selected, examples, [res[0] for res in results])
    files = {"spec.json": spec_bytes, "scenes.jsonl": DEV._jsonl([res[0] for res in results]),
             "rounds.json": DEV._dumps(rounds), "strata.jsonl": DEV._jsonl([dict(s, cohort=c) for c in COHORTS for s in strata[c]]),
             "cohort.json": DEV._dumps({c: [{"parent_command_id": r["parent_command_id"], "group": r["group"],
                                             "scene": r["scene"], "stratum": r["stratum"], "sampling_rank": r["sampling_rank"],
                                             "direct_views": r["direct_views"]} for r in selected[c]] for c in COHORTS}),
             "example-failures.jsonl": DEV._jsonl(failures),
             "reference_only/labels.jsonl": DEV._jsonl(rows),
             "reference_only/label-strings.jsonl": DEV._jsonl([{"label": s} for s in sorted({r["label"] for r in rows if r["label"]})]),
             "audit/audit.jsonl": DEV._jsonl(audit), "audit/audit.md": audit_markdown(audit).encode("utf-8")}
    for c in COHORTS:
        for f in FORMATS:
            files[f"training_data/{c}-direct-{f}.jsonl"] = DEV._jsonl([e for e in examples[c]["direct"] if e["format"] == f])
        files[f"training_data/{c}-parser.jsonl"] = DEV._jsonl(examples[c]["parser"])
    manifest = {"format_version": 1, "record_type": "a24_preparation_manifest", "spec_id": SPEC_ID,
                "spec_sha256": _sha(spec_bytes), "started_utc": started, "finished_utc": _now(),
                "elapsed_s": round(time.perf_counter() - t0, 1), "workers": workers, "counts": counts,
                "reuse": {"work": str(work), "scans_reused": sum(bool(res[0].get("reused")) for res in results),
                          "scans_prepared_fresh": sum(not res[0].get("reused") and res[0]["status"] == "prepared"
                                                      for res in results),
                          "checks": "per scan, in scenes.jsonl (reuse_checks)"} if reuse else None,
                "files": {n: _sha(d) for n, d in sorted(files.items())}, "code": code_hashes(),
                "status": "prepared; for the Stage A review before any fit"}
    files["manifest.json"] = DEV._dumps(manifest)
    files["pilot-spec.md"] = pilot_spec(S, settings, manifest, examples, rounds,
                                        scene_problems([res[0] for res in results])).encode("utf-8")
    folder = _publish(Path(out), files)
    return dict(manifest, folder=str(folder))


def scene_problems(scenes) -> list:
    """Every scan that was not prepared, and every scan that needed a lock retry, with its error and retries."""
    return [{"scene": r["scene"], "group": r["group"], "cohort": r.get("cohort"), "status": r["status"],
             "error": (r.get("error") or "")[:1500], "retries": r.get("retries") or []}
            for r in scenes if r["status"] != "prepared" or r.get("retries")]


def summary(manifest, scenes) -> dict:
    """What `prepare` prints: scans by status, every scan not prepared (with its error), the counts and the cohorts."""
    c = manifest["counts"]
    return {"folder": manifest.get("folder"), "elapsed_s": manifest["elapsed_s"], "scans": c["scenes"],
            "scans_reused_after_verification": sum(bool(r.get("reused")) for r in scenes),
            "scans_not_prepared": [x for x in scene_problems(scenes) if x["status"] != "prepared"],
            "scans_with_lock_retries": [x["scene"] for x in scene_problems(scenes) if x["retries"]],
            "counts": {k: c[k] for k in ("parents", "eligible_parents", "retained_parents")},
            "fitting": c["fitting"]["parents"], "validation": c["validation"]["parents"]}


def _jl(p) -> list:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def report(*, prep, out, settings_path=SETTINGS_PATH) -> dict:
    """The counts and pilot-spec.md regenerated from a saved preparation, into a new folder (fix 2). Nothing is rerun
    and the preparation is not changed: its files are verified against its manifest first, and the new manifest records
    the hashes of the report it supersedes."""
    prep, out = Path(prep), Path(out)
    if out.exists():
        raise PrepError("the output folder must be new")
    man_bytes = (prep / "manifest.json").read_bytes()
    man = json.loads(man_bytes)
    if man.get("record_type") != "a24_preparation_manifest":
        raise PrepError(f"{prep} is not an A2.4 preparation")
    changed = [n for n, h in man["files"].items() if not (prep / n).is_file() or _sha_file(prep / n) != h]
    if changed:
        raise PrepError(f"the preparation's files differ from its manifest: {changed[:5]}")
    S = json.loads((prep / "spec.json").read_bytes())
    if _sha(Path(settings_path).read_bytes()) != S["inputs"]["settings"]["sha256"]:
        raise PrepError("the settings differ from the ones the specification froze")
    settings = load_settings(settings_path)
    rows = _jl(prep / "reference_only" / "labels.jsonl")
    srows = _jl(prep / "strata.jsonl")
    strata = {c: [x for x in srows if x["cohort"] == c] for c in COHORTS}
    cohort = json.loads((prep / "cohort.json").read_text(encoding="utf-8"))
    examples = {c: {"direct": [e for f in FORMATS for e in _jl(prep / "training_data" / f"{c}-direct-{f}.jsonl")],
                    "parser": _jl(prep / "training_data" / f"{c}-parser.jsonl")} for c in COHORTS}
    scenes = _jl(prep / "scenes.jsonl")
    rounds = json.loads((prep / "rounds.json").read_text(encoding="utf-8"))
    counts = count(rows, strata, {c: cohort[c] for c in COHORTS}, examples, scenes)
    spec_md = pilot_spec(S, settings, dict(man, counts=counts), examples, rounds, scene_problems(scenes))
    files = {"counts.json": DEV._dumps(counts), "pilot-spec.md": spec_md.encode("utf-8")}
    rec = {"format_version": 1, "record_type": "a24_preparation_report", "made_utc": _now(),
           "preparation": str(prep), "preparation_manifest_sha256": _sha(man_bytes),
           "supersedes": {"pilot-spec.md": _sha_file(prep / "pilot-spec.md"), "manifest.json (its counts)": _sha(man_bytes)},
           "reason": "fix 2: an environment contributes nothing when its retained parents sum to zero; the first report "
                     "listed every environment with any relation stratum lacking a retained parent",
           "files": {n: _sha(d) for n, d in sorted(files.items())}, "code": code_hashes()}
    files["manifest.json"] = DEV._dumps(rec)
    folder = _publish(out, files)
    return dict(rec, folder=str(folder), counts=counts)


def count(rows, strata, selected, examples, scenes) -> dict:
    def by(rs, f):
        return dict(sorted(Counter(f(r) for r in rs).items()))
    elig = [r for r in rows if r["eligible"]]
    out = {"scenes": by(scenes, lambda r: r["status"]), "parents": len(rows), "eligible_parents": len(elig),
           "excluded_by_a26_eligibility": by([r for r in rows if not r["eligible"]],
                                             lambda r: "; ".join(sorted({x.split(': ')[-1] for x in r["eligibility_reasons"]}))),
           "quarantine_reasons_by_parent": by([x for r in elig for x in r["parent_reasons"]], lambda x: x),
           "retained_parents": sum(r["retained"] for r in rows),
           "view_outcomes_among_eligible": {v: by([r for r in elig if r["views"].get(v)],
                                                  lambda r, v=v: "supported" if r["views"][v]["supported"]
                                                  else r["views"][v].get("reason", "not_materialized")) for v in VIEWS},
           "view_outcomes_among_retained": {v: by([r for r in rows if r["retained"]],
                                                  lambda r, v=v: "supported" if r["views"][v]["supported"]
                                                  else r["views"][v].get("reason")) for v in VIEWS},
           "retained_relations": by([r for r in rows if r["retained"]], lambda r: r["stratum"]),
           "between_anchor_order_in_text": by([r for r in rows if r["retained"] and " | between | " in r["label"]],
                                              lambda r: text_order(r["text"], r["label"])),
           "retained_with_colour_word_category": sum(colour_word_category(r["label"]) for r in rows if r["retained"])}
    for c in COHORTS:
        d = [e for e in examples[c]["direct"] if e["format"] == FORMATS[0]]
        sums = defaultdict(lambda: [0, 0])
        for s_ in strata[c]:
            sums[s_["group"]][0] += s_["N"]
            sums[s_["group"]][1] += s_.get("N_eligible", 0)
        out[c] = {"environments": len(sums),
                  "environments_contributing": len({s["group"] for s in strata[c] if s["n"]}),
                  "environments_without_retained_parents": sorted(g for g, (n, _) in sums.items() if n == 0),
                  "environments_without_eligible_parents": sorted(g for g, (_, e) in sums.items() if e == 0),
                  "parents": len(selected[c]), "relations": by(selected[c], lambda r: r["stratum"]),
                  "direct_rows_per_format": len(d), "direct_rows_by_view": by(d, lambda e: e["view"]),
                  "target_letters": {v: by([e for e in d if e["view"] == v], lambda e: e["answer_code"]) for v in VIEWS},
                  "objects_per_row": by(d, lambda e: len(e["codes"]) - 1),
                  "parser_examples": len(examples[c]["parser"]),
                  "tokens_per_epoch": {f"direct.{f}": sum(e["input_tokens"] + 1 for e in examples[c]["direct"] if e["format"] == f)
                                       for f in FORMATS} | {"parser": sum(e["input_tokens"] + e["output_tokens"]
                                                                          for e in examples[c]["parser"])},
                  "supervised_tokens_per_epoch": {"direct (each format)": len(d),
                                                  "parser": sum(e["output_tokens"] for e in examples[c]["parser"])}}
    return out


def pilot_spec(S, settings, manifest, examples, rounds, problems=()) -> str:
    c, st = manifest["counts"], settings
    fx = examples["fitting"]
    d0 = next((e for e in fx["direct"] if e["format"] == FORMATS[0]), None)
    p0 = fx["parser"][0] if fx["parser"] else None
    L_ = ["# A2.4 pilot specification (draft from the preparation; for the Stage A review)", "",
          f"Training specification `{manifest['spec_sha256']}`; A2.6 specification `{S['inputs']['a26_spec_sha256'][:12]}...`; "
          f"inventory `{S['inputs']['inventory_sha256']}`; language `{S['inputs']['language']['sha256'][:12]}...`; "
          f"settings `{S['inputs']['settings']['sha256'][:12]}...`. Prepared {manifest['started_utc']} to "
          f"{manifest['finished_utc']} ({manifest['elapsed_s']} s, {manifest['workers']} workers).", "",
          "## Cohorts", ""]
    for k in COHORTS:
        x = c[k]
        none_ret, none_el = x["environments_without_retained_parents"], x["environments_without_eligible_parents"]
        quar = sorted(set(none_ret) - set(none_el))
        L_.append(f"- **{k}:** {x['parents']} parents from {x['environments_contributing']} of its {x['environments']} "
                  f"environments; {len(none_ret)} contribute none: {len(none_el)} with no eligible parent under A2.6b"
                  + (f" {none_el}" if none_el else "") + f", and {len(quar)} whose eligible parents were all quarantined"
                  + (f" {quar}" if quar else "") + f". {x['direct_rows_per_format']} direct rows per format "
                  f"({x['direct_rows_by_view']}); {x['parser_examples']} parser examples.")
    L_ += [f"- Rounds (reserve use): {json.dumps(rounds, sort_keys=True)}", "", "## Scans", "",
           f"- By status: {json.dumps(c['scenes'])}."]
    bad = [x for x in problems if x["status"] != "prepared"]
    L_ += [f"- **Not prepared:** `{x['scene']}` ({x['cohort']}): {x['status']}; {x['error'][:300]}" for x in bad] or \
        ["- Every drawn scan was prepared."]
    L_ += [f"- Lock retries: `{x['scene']}`, {len(x['retries'])} retried" for x in problems if x["retries"]]
    if manifest.get("reuse"):
        r = manifest["reuse"]
        L_.append(f"- Reused from an earlier attempt's work folder after verification: {r['scans_reused']} scans "
                  f"(prepared fresh: {r['scans_prepared_fresh']}); every check per scan is in `scenes.jsonl`.")
    L_ += ["", "## Supervision coverage and exclusions", "",
           f"- Parents prepared: {c['parents']}; eligible under A2.6b: {c['eligible_parents']}; retained: {c['retained_parents']}.",
           f"- Excluded by A2.6b's eligibility (by parent): {json.dumps(c['excluded_by_a26_eligibility'])}",
           f"- Quarantined among eligible (by parent; a parent may have several): {json.dumps(c['quarantine_reasons_by_parent'])}",
           f"- View checks among eligible: {json.dumps(c['view_outcomes_among_eligible'])}",
           f"- View checks among retained: {json.dumps(c['view_outcomes_among_retained'])}",
           f"- Retained relations: {json.dumps(c['retained_relations'])}",
           f"- Between labels whose anchors appear in the text in the label's order (an independent whole-word search; a "
           f"finding on this data, not a release-wide guarantee): {json.dumps(c['between_anchor_order_in_text'])}",
           f"- Retained parents with a category containing a colour word: {c['retained_with_colour_word_category']}",
           "- Per environment and stratum, eligible (A2.6b), retained and selected counts, zero-selected strata included: "
           "`strata.jsonl`.",
           "- Inner-validation metric: unweighted accuracy over the retained validation rows (settings `selection`); "
           "A2.6's N/n weights are kept for the development benchmark only.", "",
           "## Exposure (per epoch; three epochs)", ""]
    for k in COHORTS:
        L_.append(f"- {k}: tokens {json.dumps(c[k]['tokens_per_epoch'])}; supervised tokens "
                  f"{json.dumps(c[k]['supervised_tokens_per_epoch'])}; target letters {json.dumps(c[k]['target_letters'])}; "
                  f"objects per row {json.dumps(c[k]['objects_per_row'])}")
    L_ += ["", "## Input and output examples", ""]
    if d0:
        L_.append(f"- Direct ({d0['format']}, {d0['view']}): a D104 request of {d0['input_tokens']} tokens (prompt "
                  f"`{d0['prompt_sha256'][:12]}...`); the answer is one token, `{d0['answer_code']}` "
                  f"(id {d0['answer_token_id']}).")
    if p0:
        L_.append(f"- Parser: \"{p0['text']}\" -> `{p0['label']}` ({p0['input_tokens']} prompt tokens, "
                  f"{p0['output_tokens']} output tokens with the end token).")
    L_ += ["", "## The comparison matrix", "",
           "| Configuration | Learned input and output | Scene evidence at inference |", "|---|---|---|",
           "| direct.coordinates_v2 | D104 prompt -> one letter | the frozen view's coordinates_v2 document |",
           "| direct.coordinates_relations_v2 | D104 prompt -> one letter | the same view's relations document |",
           "| parser | command text -> v1 query string | the same view, through the accepted resolver |", "",
           "Three paired seeds each, nine fits. The direct formats share rows, order, initialization and exposure; the "
           "parser trains once per parent (its exposure differs, and is reported).", "",
           "## Training settings (frozen; `settings.v1.json`)", "",
           f"- LoRA: {json.dumps(st['lora'])}", f"- Optimizer: {json.dumps(st['optimizer'])}",
           f"- Schedule: {json.dumps(st['schedule'])}; batch: {json.dumps(st['batch'])}; epochs {st['epochs']}",
           f"- Precision: {json.dumps(st['precision'])}; context: {json.dumps(st['context'])}",
           f"- Supervision: {json.dumps(st['supervision'])}", f"- Seeds: {st['seeds']}; data order: {st['data_order']}",
           f"- Checkpoints: {json.dumps(st['checkpoints'])}", f"- Selection: {json.dumps(st['selection'])}",
           f"- Continuation after seed 1: {json.dumps(st['continuation'])}", f"- Merge: {json.dumps(st['merge'])}",
           "", "## Decoding constraints (parser)", "",
           f"- Language `{L.LANGUAGE_ID}`, greedy, constrained to it over every tokenization; the end token only at a "
           f"complete string; at most {L.load_spec()['max_new_tokens']} new tokens; outcomes completed, "
           "length_exhausted, empty_allowed_set, premature_eos, each distinct from ASK; the text validated independently.",
           "", "## Evaluation (after all nine fits and their selections)", "",
           "- Direct: each selected checkpoint, merged, through A2.6c's preflight-bound runner over all 10,860 frozen "
           "requests (its readback requires every request); each configuration is reported on its own format.",
           "- Parser: one decode per parent (2,715), resolved in both views; outcomes per parent and view.",
           f"- Analysis: {json.dumps(st['analysis'])}", "",
           "## Time estimates", "", "From the Stage A smoke (`smoke-report.md`): measured steps, validation, decoding and "
           "merge, extrapolated to the nine fits and nine evaluations.", ""]
    return "\n".join(L_) + "\n"
