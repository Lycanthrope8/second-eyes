"""python -m grounding.inference.iref_vla_dev COMMAND ... (A2.6c: baseline execution and the frozen analysis).

Exit codes, as A2.3d's:

- 0: complete;
- 1: complete, but with requests over the ceiling or failed (kept, and counted);
- 2: an input refused (the preflight's problems are listed, and its receipt is written);
- 3: an output or technical failure. Nothing is published; partial runs stay resumable, and failed evidence is kept.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError, encode_json
from . import preflight as PF


def _tok_dirs(a) -> dict:
    out = {PF.MODEL_KEYS[0]: a.tokenizer_small}
    if a.tokenizer_large:
        out[PF.MODEL_KEYS[1]] = a.tokenizer_large
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m grounding.inference.iref_vla_dev")
    sub = p.add_subparsers(dest="command", required=True)

    def inputs(x, tokens=True):
        x.add_argument("--prep", required=True, help="the accepted A2.6b preparation folder (with requests.jsonl)")
        x.add_argument("--work", required=True, help="its work folder (the per-scene bundles)")
        if tokens:
            x.add_argument("--tokenizer-small", required=True, help="the 0.5B's pinned tokenizer folder")
            x.add_argument("--tokenizer-large", help="the 7B's acquired model folder (its pinned tokenizer files)")
    pf = sub.add_parser("preflight", help="verify the frozen requests read-only; write the receipt and token hashes")
    inputs(pf)
    pf.add_argument("--workers", type=int, default=8, help="worker processes, by scene (default 8)")
    pf.add_argument("--out", required=True, help="the new preflight folder (receipt.json, tokens-<model>.jsonl)")
    for name in ("smoke", "run"):
        x = sub.add_parser(name, help=f"A2.3d's accepted {name}, after the preflight")
        inputs(x)
        x.add_argument("--preflight", required=True, help="the passed preflight folder this step must bind to")
        x.add_argument("--model", required=True, choices=PF.MODEL_KEYS)
        x.add_argument("--model-dir", required=True)
        x.add_argument("--device", default="cuda")
        x.add_argument("--out", required=True)
        if name == "run":
            x.add_argument("--smoke", required=True, help="the accepted smoke record")
            x.add_argument("--resume", action="store_true")
    r = sub.add_parser("rules", help="the accepted parser and resolver, one result per parent and view")
    inputs(r, tokens=False)
    r.add_argument("--out", required=True)
    de = sub.add_parser("desktop-export", help="the frozen 64 desktop requests (on the RTX PC)")
    de.add_argument("--prep", required=True)
    de.add_argument("--out", required=True)
    dr = sub.add_parser("desktop-run", help="path U on the desktop build (on the machine with the host library)")
    dr.add_argument("--export", required=True)
    dr.add_argument("--out", required=True)
    dr.add_argument("--model")
    dr.add_argument("--library")
    dc = sub.add_parser("desktop-compare", help="the desktop rows beside the matching 0.5B float32 rows")
    dc.add_argument("--desktop", required=True)
    dc.add_argument("--run-small", required=True, help="the 0.5B float32 run folder")
    dc.add_argument("--out", required=True)
    sc = sub.add_parser("score", help="the established outcomes (targets read here only)")
    sc.add_argument("--prep", required=True)
    sc.add_argument("--rules", required=True)
    sc.add_argument("--run-small", required=True)
    sc.add_argument("--run-large", required=True)
    sc.add_argument("--out", required=True)
    an = sub.add_parser("analyze", help="the frozen analysis: weights, paired environment-cluster bootstrap, report")
    an.add_argument("--prep", required=True)
    an.add_argument("--scores", required=True)
    an.add_argument("--desktop-compare")
    an.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        if a.command == "preflight":
            try:
                res = PF.preflight(prep=a.prep, work=a.work, tokenizer_dirs=_tok_dirs(a), workers=a.workers)
            except PF.PreflightError as e:
                rec = getattr(e, "receipt", None) or {"passed": False, "problems": e.issues}
                Path(a.out).mkdir(parents=True, exist_ok=True)
                (Path(a.out) / "receipt-refused.json").write_bytes(encode_json(rec))
                raise
            PF.write_preflight(res, a.out)
            rec = res["receipt"]
            print(f"preflight passed in {rec['duration_s']} s with {rec['workers']} workers: {rec['counts']}")
            for k, v in rec["models"].items():
                print(f"  {k}: tokenizer {v['tokenizer_identity']}; {v['requests']} requests; "
                      f"token IDs equal to the prepared ones: {v['same_token_ids_as_prepared']}")
            print(f"written to {a.out}")
            return 0
        if a.command in ("smoke", "run"):
            from . import runs as RN
            common = dict(prep=a.prep, work=a.work, preflight_dir=a.preflight, model_key=a.model, model_dir=a.model_dir,
                          tokenizer_dirs=_tok_dirs(a), device=a.device, out=a.out)
            if a.command == "smoke":
                s = RN.smoke(**common)
                print(f"{a.model}: smoke accepted {s['accepted']}; canaries {[c['accepted'] for c in s['canaries']]}; "
                      f"settings {s['settings']}")
                return 0 if s["accepted"] else 3
            x = RN.run(**common, smoke_record=a.smoke, resume=a.resume)
            c, f = x["counts"], x["timing_ms"]["forward"]
            print(f"{a.model}: completed {c['completed']} of {c['requests']}; over the ceiling {c['context_budget_exceeded']}; "
                  f"execution failures {c['execution_failed']}; sessions {c['sessions']}; canaries {c['canaries_accepted']}/{c['canaries']}")
            print(f"  forward median {f['median']:.1f} ms, p95 {f['p95']:.1f} ms; peak GPU bytes {x['peak_gpu_bytes']}")
            print(f"written to {a.out}")
            return 1 if c["context_budget_exceeded"] or c["execution_failed"] else 0
        if a.command == "rules":
            from . import runs as RN
            s = RN.run_rules(prep=a.prep, work=a.work, out=a.out)
            print(f"rules: {json.dumps(s.get('counts', {}))}; written to {a.out}")
            return 0
        if a.command == "desktop-export":
            from . import desktop as DK
            m = DK.export(prep=a.prep, out=a.out)
            print(f"exported {m['requests']} desktop requests (records {m['records_sha256'][:12]}...) to {a.out}")
            return 0
        if a.command == "desktop-run":
            from . import desktop as DK
            d = DK.run(export_dir=a.export, out=a.out, model=a.model, library=a.library)
            print(f"desktop path U: {d['completed']} of {d['requests']} completed; failed {d['failed']}; written to {a.out}")
            return 0 if not d["failed"] else 1
        if a.command == "desktop-compare":
            from . import desktop as DK
            s = DK.compare(desktop=a.desktop, run05=a.run_small, out=a.out)
            print(f"{s['label']}: {s['same_choice']} of {s['compared']} decisions equal; largest offered-logit difference "
                  f"{s['max_abs_logit_difference']}; not compared {len(s['not_compared'])}")
            return 0
        if a.command == "score":
            from . import score as SC
            m = SC.score(prep=a.prep, rules=a.rules, runs={PF.MODEL_KEYS[0]: a.run_small, PF.MODEL_KEYS[1]: a.run_large}, out=a.out)
            print(f"scored: {json.dumps(m['counts'])}; written to {a.out}")
            return 0
        from . import analysis as AN
        r = AN.analyze(prep=a.prep, scores=a.scores, out=a.out, desktop=a.desktop_compare)
        print((Path(a.out) / "report.md").read_text(encoding="utf-8"))
        return 0
    except EvaluationInputError as e:
        for i in e.issues:
            print(f"input error {i['code']} at {i['path']}: {i['message']}", file=sys.stderr)
        return 2
    except EvaluationOutputError as e:
        for i in e.issues:
            print(f"output error {i['code']} at {i['path']}: {i['message']}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        print(f"failure, nothing published: {type(e).__name__}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
