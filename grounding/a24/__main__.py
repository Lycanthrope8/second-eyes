"""A2.4's commands (D109). Training environment (.venv-train): freeze, prepare, gbnf-check, smoke, fit, select, merge,
continuation. Accepted evaluation environment (.venv): eval-direct, eval-parser, score, analyze.

    python -m grounding.a24 <command> --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _pairs(values) -> dict:
    out = {}
    for v in values or []:
        k, _, p = v.partition("=")
        if not p:
            raise SystemExit(f"expected NAME=PATH, got {v!r}")
        out[k] = p
    return out


def locate(root) -> dict:
    """A2.6c's folders under root (three levels), identified by their own records, never by name: run folders by their
    result manifest's model key, the rules by their manifest, the preflight by its receipt, scores by their manifest."""
    out = {"runs": {}, "rules": [], "preflight": [], "scores": []}
    root = Path(root)
    for p in sorted(list(root.glob("*/*.json")) + list(root.glob("*/*/*.json")) + list(root.glob("*.json"))):
        if p.name not in ("manifest.json", "receipt.json"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rt = rec.get("record_type")
        if rt == "iref_compare_result_manifest":
            out["runs"].setdefault(rec.get("model_key"), []).append(str(p.parent))
        elif rt == "a26c_rules_manifest":
            out["rules"].append(str(p.parent))
        elif rt == "a26c_preflight":
            out["preflight"].append(str(p.parent))
        elif rt == "a26c_scores_manifest":
            out["scores"].append(str(p.parent))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.a24")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("freeze", help="freeze the training specification (environments, cohorts, reserve)")
    s.add_argument("--a26-spec", required=True)
    s.add_argument("--inventory", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("prepare", help="the authorized chain, labels, cohorts, examples, audit, specification draft")
    for a in ("--spec", "--a26-spec", "--zip", "--inventory", "--vocabulary", "--out"):
        s.add_argument(a, required=True)
    w = s.add_mutually_exclusive_group(required=True)
    w.add_argument("--work", help="a new work folder")
    w.add_argument("--reuse-work", help="an earlier attempt's work folder: each scan is verified, then reused")
    s.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("report", help="the counts and pilot-spec.md regenerated from a saved preparation (new folder)")
    s.add_argument("--prep", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("audit-export", help="the same audit examples re-exported with full geometry and linked evidence")
    for x in ("--prep", "--work", "--out"):
        s.add_argument(x, required=True)
    s = sub.add_parser("gbnf-check", help="the language against the pinned llama.cpp's grammar validator")
    s.add_argument("--validator", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--labels-file", default=None)
    s = sub.add_parser("smoke", help="Stage A: short fits, resume, validation, merge, identity, measured estimates")
    for a in ("--prep", "--work", "--base", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--a26-index", default=None)
    s.add_argument("--device", default="cuda")
    s = sub.add_parser("fit", help="one planned fit (resumable)")
    for a in ("--prep", "--work", "--config", "--base", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--seed-index", type=int, required=True, choices=(1, 2, 3))
    s.add_argument("--resume", action="store_true")
    s.add_argument("--device", default="cuda")
    s = sub.add_parser("select", help="apply the frozen selection rule to a completed fit")
    s.add_argument("--fit", required=True)
    s = sub.add_parser("merge", help="merge the selected checkpoint, check it, write the fine-tuned identity")
    for a in ("--fit", "--prep", "--base", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--device", default="cuda")
    s = sub.add_parser("continuation", help="the mechanical continuation check after seed 1")
    s.add_argument("--fit", action="append", required=True, help="CONFIG=FOLDER")
    s.add_argument("--merged", action="append", required=True, help="CONFIG=FOLDER")
    s.add_argument("--runner-smoke", action="append", required=True, help="CONFIG=FILE")
    s.add_argument("--out", required=True)
    s = sub.add_parser("eval-direct", help="a merged direct model through A2.6c's preflight-bound runner")
    for a in ("--prep", "--work", "--preflight", "--model", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--smoke-only", action="store_true")
    s.add_argument("--device", default="cuda")
    s = sub.add_parser("eval-parser", help="a merged parser model: one decode per parent, resolved in both views")
    for a in ("--prep", "--work", "--model", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--device", default="cuda")
    s = sub.add_parser("score", help="A2.6c's scoring for direct runs; the rules' outcome rule for the parser")
    for a in ("--prep", "--rules", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--base-run", action="append", default=[], help="MODEL_KEY=A2.6c RUN FOLDER")
    s.add_argument("--direct", action="append", default=[])
    s.add_argument("--parser", action="append", default=[])
    s = sub.add_parser("locate", help="find A2.6c's run, rules and preflight folders under a root by their records")
    s.add_argument("--root", required=True)
    s = sub.add_parser("analyze", help="per-seed estimates, seed means and spread, paired differences, supplementary")
    for a in ("--prep", "--scores", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--supersedes", help="an earlier analysis folder this one replaces (kept unchanged; its hash recorded)")
    s = sub.add_parser("analysis-diff", help="what changed between two analyses: intervals, zero crossings, other numbers")
    s.add_argument("--old", required=True, help="the earlier analysis folder")
    s.add_argument("--new", required=True, help="the later analysis folder")
    a = ap.parse_args(argv)
    from ..evaluation.iref_vla.protocol import EvaluationInputError
    if a.cmd == "locate":
        print(json.dumps(locate(a.root), indent=1))
        return 0
    try:
        if a.cmd == "audit-export":
            from . import prep as P
            r = P.audit_export(prep=a.prep, work=a.work, out=a.out)
            print(json.dumps({k: r[k] for k in ("folder", "counts", "supersedes", "derived_records_match_bundle_index",
                                                "letters_match_original_audit")}, indent=1))
            return 0
        if a.cmd == "report":
            from . import prep as P
            r = P.report(prep=a.prep, out=a.out)
            print(json.dumps({"folder": r["folder"], "supersedes": r["supersedes"],
                              **{c: {k: r["counts"][c][k] for k in ("parents", "environments", "environments_contributing",
                                                                    "environments_without_retained_parents",
                                                                    "environments_without_eligible_parents")}
                                 for c in P.COHORTS}}, indent=1))
            return 0
        if a.cmd in ("freeze", "prepare"):
            from . import prep as P
            if a.cmd == "freeze":
                r = P.freeze(a26_spec=a.a26_spec, inventory_dir=a.inventory, out=a.out)
                print(json.dumps({"folder": r["folder"], "spec_sha256": r["spec_sha256"],
                                  "fitting": len(r["environments"]["fitting"]),
                                  "validation": len(r["environments"]["validation"]),
                                  "reserve": len(r["environments"]["reserve"])}, indent=1))
            else:
                r = P.prepare(spec=a.spec, a26_spec=a.a26_spec, zip_path=a.zip, inventory_dir=a.inventory,
                              vocabulary=a.vocabulary, out=a.out, work=a.reuse_work or a.work, reuse=bool(a.reuse_work),
                              workers=a.workers)
                scenes = [json.loads(x) for x in (Path(r["folder"]) / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
                          if x.strip()]
                print(json.dumps(P.summary(r, scenes), indent=1))
        elif a.cmd == "gbnf-check":
            from . import gbnf_check as G
            print(json.dumps(G.check(validator=a.validator, out=a.out, labels_file=a.labels_file), indent=1))
            return 0
        elif a.cmd in ("smoke", "fit", "select", "merge", "continuation"):
            from . import train as T
            if a.cmd == "smoke":
                r = T.smoke(prep=a.prep, work=a.work, base_dir=a.base, out=a.out, device=a.device, a26_index=a.a26_index)
                print(json.dumps(r["stage_b"], indent=1))
            elif a.cmd == "fit":
                r = T.fit(prep=a.prep, work=a.work, config=a.config, seed_index=a.seed_index, base_dir=a.base, out=a.out,
                          device=a.device, resume=a.resume)
                print(json.dumps({k: r[k] for k in ("model_name", "optimizer_steps", "planned_optimizer_steps")}, indent=1))
            elif a.cmd == "select":
                print(json.dumps(T.select(a.fit), indent=1))
            elif a.cmd == "merge":
                r = T.merge(fit_dir=a.fit, prep=a.prep, base_dir=a.base, out=a.out, device=a.device)
                ident = Path(a.out) / "a24-identity.json"
                print(json.dumps({"model_name": r["model_name"],
                                  "identity_sha256": hashlib.sha256(ident.read_bytes()).hexdigest(),
                                  "check": r["merge"]["check"]}, indent=1))
            else:
                r = T.continuation(fits=_pairs(a.fit), merged=_pairs(a.merged), runner_smokes=_pairs(a.runner_smoke))
                Path(a.out).write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
                print(json.dumps(r, indent=1))
                return 0 if r["passed"] else 3
        else:
            from . import evaluate as E
            if a.cmd == "eval-direct":
                r = E.direct(prep=a.prep, work=a.work, preflight=a.preflight, model_dir=a.model, out=a.out,
                             device=a.device, resume=a.resume, smoke_only=a.smoke_only)
                print(json.dumps({"model_name": r["model_name"], "smoke_only": a.smoke_only}, indent=1))
            elif a.cmd == "eval-parser":
                r = E.parser(prep=a.prep, work=a.work, model_dir=a.model, out=a.out, device=a.device, resume=a.resume)
                print(json.dumps({k: r[k] for k in ("model_name", "parents", "statuses")}, indent=1))
            elif a.cmd == "score":
                r = E.score(prep=a.prep, rules=a.rules, base_runs=_pairs(a.base_run), direct_evals=a.direct,
                            parser_evals=a.parser, out=a.out)
                print(json.dumps(r["counts"], indent=1))
            elif a.cmd == "analysis-diff":
                print(json.dumps(E.analysis_diff(Path(a.old) / "analysis.json", Path(a.new) / "analysis.json"), indent=1))
            else:
                r = E.analyze(prep=a.prep, scores=a.scores, out=a.out, supersedes=a.supersedes)
                print(f"analysis written to {a.out}")
                if a.supersedes:
                    print(json.dumps({"supersedes": {k: v for k, v in r["supersedes"].items() if k != "changes"},
                                      "changes": r["supersedes"]["changes"]}, indent=1))
    except (EvaluationInputError, ValueError, RuntimeError, OSError) as e:
        print(f"REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
