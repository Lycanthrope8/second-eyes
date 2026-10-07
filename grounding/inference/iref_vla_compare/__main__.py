"""Command line for A2.3d (D95): `prepare` and `rules` (laptop); `smoke` and `run` (RTX PC).

Exit 0 complete; 1 a published run with requests over the context ceiling or failed executions; 2 invalid input, an
identity or resume mismatch, or an existing destination; 3 a failed canary or smoke check, repeated execution failures or
an unexpected failure (nothing published; a partial run stays resumable)."""
from __future__ import annotations

import argparse
import sys

from ...evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grounding.inference.iref_vla_compare")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="freeze the selection, the mappings, the prompts and both tokenizers' token IDs")
    for n in ("--bundle", "--tokenizer-small", "--tokenizer-large", "--out"):
        p.add_argument(n, required=True)
    r = sub.add_parser("rules", help="the unchanged parser and resolver on the models' subscenes")
    for n in ("--requests", "--bundle", "--relation-config", "--direction-config", "--out"):
        r.add_argument(n, required=True)
    for name, helptext in (("smoke", "resource smoke check: canaries and the longest requests; writes the frozen settings record"),
                           ("run", "one model's run over every request, gated by the smoke record; resumable")):
        s = sub.add_parser(name, help=helptext)
        for n in ("--requests", "--model", "--model-dir", "--tokenizer-dir", "--out"):
            s.add_argument(n, required=True)
        s.add_argument("--device", required=True, choices=("cuda",))
        if name == "run":
            s.add_argument("--smoke", required=True)
            s.add_argument("--resume", action="store_true")
    sc = sub.add_parser("score", help="score both runs and the rules against the pinned annotations (laptop)")
    for n in ("--requests", "--rules", "--small-run", "--large-run", "--bundle", "--annotations", "--out"):
        sc.add_argument(n, required=True)
    a = ap.parse_args(argv)
    try:
        if a.command == "prepare":
            from .prepare import prepare_compare
            m = prepare_compare(bundle=a.bundle, tokenizer_small=a.tokenizer_small, tokenizer_large=a.tokenizer_large, out=a.out)
            c = m["counts"]
            print(f"prepared {c['requests']} requests: {c['parents']} parents x 2 views x 2 formats; "
                  f"selection {m['selection_sha256']}")
            print(f"  requests by object count {c['requests_by_object_count']}")
            for key, st in m["token_stats"].items():
                print(f"  {key}: " + "; ".join(f"{v}/{fm} median {s['median']} max {s['max']}" for v, d in st.items()
                                                for fm, s in d.items()))
            print(f"  over the context limit: {c['context_budget_exceeded']}; identical token IDs across the two "
                  f"tokenizers: {m['identical_token_ids_across_tokenizers']} of {c['requests']}")
            print(f"written to {a.out}")
            return 0
        if a.command == "smoke":
            from .run import smoke
            x = smoke(requests=a.requests, model_key=a.model, model_dir=a.model_dir, tokenizer_dir=a.tokenizer_dir,
                      device=a.device, out=a.out)
            print(f"smoke check of {a.model}: canaries " + ", ".join(f"{c['request_id']} max difference {c['max_abs_difference']}"
                                                              for c in x["canaries"]))
            for p in x["longest_requests"]:
                print(f"  {p['request_id']}: {p['input_tokens']} tokens, forward {p['forward_ms']:.1f} ms")
            print(f"  load {x['load_ms'] / 1000:.1f} s; peak GPU bytes {x['peak_gpu_bytes']}; settings {x['settings']}")
            print(f"written to {a.out}")
            return 0
        if a.command == "run":
            from .run import run_compare
            x = run_compare(requests=a.requests, model_key=a.model, model_dir=a.model_dir, tokenizer_dir=a.tokenizer_dir,
                            device=a.device, smoke_record=a.smoke, out=a.out, resume=a.resume)
            c, f = x["counts"], x["timing_ms"]["forward"]
            print(f"{a.model}: completed {c['completed']} of {c['requests']}; over the ceiling {c['context_budget_exceeded']}; "
                  f"execution failures {c['execution_failed']}; sessions {c['sessions']}; canaries {c['canaries_accepted']}/{c['canaries']}")
            print(f"  forward median {f['median']:.1f} ms, p95 {f['p95']:.1f} ms; peak GPU bytes {x['peak_gpu_bytes']}")
            print(f"written to {a.out}")
            return 1 if c["context_budget_exceeded"] or c["execution_failed"] else 0
        if a.command == "score":
            from .score import score_compare
            x = score_compare(requests=a.requests, rules=a.rules, small_run=a.small_run, large_run=a.large_run,
                              bundle=a.bundle, annotations=a.annotations, out=a.out)
            for v, b in x["by_view"].items():
                for fm, y in b["by_format"].items():
                    s_ = y["systems"]
                    def pc(f):
                        return "-" if f is None else f"{f['numerator']}/{f['denominator']}"
                    print(f"  {v}/{fm}: 0.5B {pc(s_['qwen2.5-0.5b-instruct']['correct_over_planned'])}, "
                          f"7B {pc(s_['qwen2.5-7b-instruct']['correct_over_planned'])}, rules {pc(s_['rules']['correct_over_planned'])}, "
                          f"always B {pc(y['references']['always_B'])}, second position {pc(y['references']['always_second_list_position'])}")
            print(f"scored {x['counts']['requests']} requests; written to {a.out}")
            return 0
        from .rules import run_compare_rules
        s = run_compare_rules(requests=a.requests, bundle=a.bundle, relation_config=a.relation_config,
                              direction_config=a.direction_config, out=a.out)
        for v, d in s["views"].items():
            print(f"  {v}: {d['records']} records; outcomes {d['outcomes']}; technical failures {d['technical_failures']}")
        print(f"rules on {s['counts']['parents']} parents; written to {a.out}")
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
