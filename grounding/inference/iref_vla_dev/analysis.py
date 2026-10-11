"""A2.6c frozen analysis (the specification's analysis plan, and ChatGPT's A2.6c brief, item 5).

**The value per sampled parent.** y = 1 when its outcome is correct. Every other outcome counts as y = 0: wrong object,
ASK, over the ceiling, execution failure, or for the rules any unresolved bin. Nothing leaves the denominator. Each
outcome category is also reported as its own weighted rate.

**The weights.** For environment e and relation stratum s, N_es is the number of eligible parents and n_es the number
sampled (`strata.jsonl`).

- **Command-weighted:** sum_e sum_s (N_es/n_es) * sum of y in (e, s), divided by sum_e N_e. It estimates over the 26,672
  eligible development commands.
- **Environment-weighted:** acc_e = sum_s (N_es/N_e) * mean of y in (e, s), which restores each environment's eligible
  relation proportions. The result is the mean of acc_e over the contributing environments, each with equal weight.
- **Unweighted:** the mean of y over the sampled parents. It is labelled as the stratified sample.

**The bootstrap: paired, by environment cluster.**

- The generator is Python's `random.Random(20261010)`, with B = 10,000 replicates.
- Each replicate draws E of the E contributing development environments (sorted IDs) by `randrange(E)`, with
  multiplicity: an environment drawn twice counts twice.
- Each replicate recomputes every estimate from per-environment aggregates, with the same draws for every system,
  format and view. The paired differences are therefore computed within each replicate.
- Intervals are 95% percentile intervals, interpolated linearly between order statistics (Hyndman-Fan type 7). Sums use
  `math.fsum`.

**The rules** give one result per parent and view. They are reported per view, and serve both formats in the paired
differences without becoming two observations. **The legacy environment** is reported separately: one environment, so
point estimates only. **The zero-eligible environment** has no data and is reported, not resampled. **"above" and
"in"** have no eligible case: their performance is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from ...evaluation.iref_vla.protocol import EvaluationInputError, encode_json, issue, runtime
from . import preflight as PF

B, SEED, LEVEL = 10000, 20261010, 0.95
MODEL_KEYS = PF.MODEL_KEYS
VIEWS, FORMATS = PF.VIEWS, PF.FORMATS
UNAVAILABLE = ("above", "in")
SCOPE = [
    "The development population is the 26,672 eligible development parents, sampled to 2,651 parents across 46 "
    "contributing environments; one assigned environment (scannet:scene0414) has none eligible.",
    "The two formats and two views of one parent are four variants of one parent observation, not four independent "
    "observations; the environment is the independent unit for the intervals.",
    "Eligibility is the accepted parser's and the 3-to-10-object selection's: 'above' and 'in' have no eligible case "
    "(their performance is unavailable), and 'near', 'below' and 'on' are largely excluded (see the coverage table).",
    "ASK outputs and annotation disagreements are not evidence of ambiguity or of unsupported commands; nothing here "
    "infers either.",
    "The desktop Q8_0/U results are a desktop comparison: they establish no Quest agreement and do not close A2.5's "
    "unresolved numerical acceptance.",
    "Calibration, the project test and the final lab commands stayed closed. No tuning, no Gate B.",
]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _jl(p) -> list:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def quantile(sorted_xs, p) -> float:
    """Hyndman-Fan type 7: linear interpolation between order statistics."""
    h = (len(sorted_xs) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (h - lo) * (sorted_xs[hi] - sorted_xs[lo])


# --------------------------------------------------------------------------------------------- aggregates
def aggregates(obs, strata) -> dict:
    """obs: {(env, stratum): [outcome, ...]} for one system, view and format (one outcome per sampled parent);
    strata: {(env, stratum): (N, n)}. Per environment: num, den, acc, sy, n, and the weighted numerator per outcome."""
    agg = {}
    for (e, s), outs in obs.items():
        N, n = strata[(e, s)]
        if len(outs) != n:
            raise ValueError(f"{e} {s}: {len(outs)} outcomes for n = {n}")
        a = agg.setdefault(e, {"num": [], "den": 0, "parts": [], "sy": 0, "n": 0, "cats": defaultdict(list)})
        y = sum(o == "correct" for o in outs)
        a["num"].append(N / n * y)
        a["den"] += N
        a["parts"].append((N, y / n))
        a["sy"] += y
        a["n"] += n
        for k, c in Counter(outs).items():
            a["cats"][k].append(N / n * c)
    out = {}
    for e, a in agg.items():
        out[e] = {"num": math.fsum(a["num"]), "den": float(a["den"]), "sy": float(a["sy"]), "n": float(a["n"]),
                  "acc": math.fsum(N / a["den"] * m for N, m in a["parts"]),
                  "cats": {k: math.fsum(v) for k, v in a["cats"].items()}}
    return out


def estimate(agg, counts=None) -> dict:
    """The three estimators over environments with multiplicities (counts: env -> count; None: each once)."""
    envs = sorted(agg)
    c = {e: (counts.get(e, 0) if counts is not None else 1) for e in envs}
    den = math.fsum(c[e] * agg[e]["den"] for e in envs)
    w = math.fsum(c[e] for e in envs)
    n = math.fsum(c[e] * agg[e]["n"] for e in envs)
    return {"command_weighted": math.fsum(c[e] * agg[e]["num"] for e in envs) / den if den else None,
            "environment_weighted": math.fsum(c[e] * agg[e]["acc"] for e in envs) / w if w else None,
            "unweighted": math.fsum(c[e] * agg[e]["sy"] for e in envs) / n if n else None}


def categories(agg) -> dict:
    den = math.fsum(a["den"] for a in agg.values())
    keys = sorted({k for a in agg.values() for k in a["cats"]})
    return {k: math.fsum(a["cats"].get(k, 0.0) for a in agg.values()) / den for k in keys}


def draws(envs, b=B, seed=SEED):
    """The replicates' multiplicities: for each replicate, env -> count, from randrange over the sorted environments."""
    rng, envs = random.Random(seed), sorted(envs)
    for _ in range(b):
        cnt = Counter()
        for _ in range(len(envs)):
            cnt[envs[rng.randrange(len(envs))]] += 1
        yield cnt


def interval(values) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"low": None, "high": None, "replicates": 0}
    return {"low": quantile(v, (1 - LEVEL) / 2), "high": quantile(v, 1 - (1 - LEVEL) / 2), "replicates": len(v)}


# --------------------------------------------------------------------------------------------- the analysis
def _combos():
    out = [("rules", v, None) for v in VIEWS]
    out += [(k, v, f) for k in MODEL_KEYS for v in VIEWS for f in FORMATS]
    return out


def _name(c):
    return f"{c[0]} | {c[1]}" + (f" | {c[2]}" if c[2] else "")


def differences():
    """The paired differences reported (first minus second)."""
    out = []
    for v in VIEWS:
        for f in FORMATS:
            out.append(((MODEL_KEYS[1], v, f), (MODEL_KEYS[0], v, f)))
            for k in MODEL_KEYS:
                out.append(((k, v, f), ("rules", v, None)))
    for k in MODEL_KEYS:
        for v in VIEWS:
            out.append(((k, v, FORMATS[1]), (k, v, FORMATS[0])))
        for f in FORMATS:
            out.append(((k, VIEWS[1], f), (k, VIEWS[0], f)))
    return out


def observations(scores, rule_scores, partition) -> dict:
    """combo -> {(env, stratum): [outcome per sampled parent]}"""
    obs = defaultdict(lambda: defaultdict(list))
    for r in rule_scores:
        if r["partition"] == partition:
            obs[("rules", r["view"], None)][(r["group"], r["stratum"])].append(r["outcome"])
    for r in scores:
        if r["partition"] == partition:
            for k in MODEL_KEYS:
                if k in r["models"]:
                    obs[(k, r["view"], r["format"])][(r["group"], r["stratum"])].append(r["models"][k]["outcome"])
    return obs


def coverage(eligibility, sampling) -> list:
    dev = [r for r in eligibility if r["partition"] == "development"]
    sel = Counter(x["stratum"] for x in sampling["selected"] if x["partition"] == "development")
    tot, ok, why = Counter(), Counter(), defaultdict(Counter)
    for r in dev:
        tot[r["stratum"]] += 1
        ok[r["stratum"]] += r["eligible"]
        for reason in {x.split(": ")[-1] for x in r["reasons"]}:
            why[r["stratum"]][reason] += 1
    return [{"relation": s, "parents": n, "eligible": ok[s], "selected": sel[s],
             "excluded_by_reason": dict(why[s]), "performance": "unavailable: no eligible case" if ok[s] == 0 else "reported"}
            for s, n in tot.most_common()]


def analyze(*, prep, scores, out, b=B, seed=SEED, desktop=None) -> dict:
    prep, scores, out = Path(prep), Path(scores), Path(out)
    if out.exists():
        raise EvaluationInputError([issue(str(out), "E_A26C_OUTPUT_EXISTS", "the output folder exists")])
    sman = json.loads((scores / "manifest.json").read_text(encoding="utf-8"))
    for name, h in sman["outputs"].items():
        if _sha((scores / name).read_bytes()) != h:
            raise EvaluationInputError([issue(str(scores / name), "E_A26C_ANALYSIS", "changed since scoring")])
    pman = json.loads((prep / "manifest.json").read_text(encoding="utf-8"))
    for name in ("strata.jsonl", "sampling.json", "reference_only/eligibility.jsonl"):
        if _sha((prep / name).read_bytes()) != pman["files"][name]:
            raise EvaluationInputError([issue(str(prep / name), "E_A26C_ANALYSIS", "changed since the preparation")])
    srows, rrows = _jl(scores / "scores.jsonl"), _jl(scores / "rules-scores.jsonl")
    strata = {(s["group"], s["stratum"]): (s["N"], s["n"]) for s in _jl(prep / "strata.jsonl") if s["stratum"]}
    sampling = json.loads((prep / "sampling.json").read_text(encoding="utf-8"))
    result = {"format_version": 1, "record_type": "a26c_analysis", "run_id": PF.RUN_ID,
              "configuration": {"replicates": b, "seed": seed, "generator": "Python random.Random(seed), randrange over "
                                "the sorted contributing development environments, E draws per replicate",
                                "interval": "95% percentile, Hyndman-Fan type 7", "sums": "math.fsum",
                                "estimators": ["command_weighted (N/n)", "environment_weighted (eligible relation "
                                               "proportions restored, equal environments)", "unweighted (stratified sample)"]},
              "scope": SCOPE, "coverage": coverage(_jl(prep / "reference_only" / "eligibility.jsonl"), sampling)}
    for part in ("development", "legacy-development"):
        obs = observations(srows, rrows, part)
        aggs = {c: aggregates(obs[c], strata) for c in _combos() if c in obs}
        block = {"systems": {}, "environments": sorted({e for a in aggs.values() for e in a})}
        for c, agg in aggs.items():
            block["systems"][_name(c)] = {"estimate": estimate(agg), "outcome_rates_command_weighted": categories(agg),
                                          "parents": int(sum(a["n"] for a in agg.values())),
                                          "observations": "one per parent and view (both formats share it)" if c[0] == "rules" else "one per request"}
        if part == "development":
            envs = block["environments"]
            reps = {c: {k: [] for k in ("command_weighted", "environment_weighted", "unweighted")} for c in aggs}
            per_rel = {(c, s): [] for c in aggs for s in sorted({k[1] for k in strata})}
            rel_aggs = {}
            for c in aggs:
                for (e, s), outs in obs[c].items():
                    N, n = strata[(e, s)]
                    rel_aggs.setdefault((c, s), {})[e] = (N / n * sum(o == "correct" for o in outs), N)
            for cnt in draws(envs, b, seed):
                for c, agg in aggs.items():
                    for k, v in estimate(agg, cnt).items():
                        reps[c][k].append(v)
                for (c, s), ea in rel_aggs.items():
                    den = math.fsum(cnt.get(e, 0) * d for e, (_, d) in ea.items())
                    per_rel[(c, s)].append(math.fsum(cnt.get(e, 0) * x for e, (x, _) in ea.items()) / den if den else None)
            for c in aggs:
                block["systems"][_name(c)]["interval"] = {k: interval(v) for k, v in reps[c].items()}
            diffs = []
            for a_, b_ in differences():
                if a_ in reps and b_ in reps:
                    pt = {k: block["systems"][_name(a_)]["estimate"][k] - block["systems"][_name(b_)]["estimate"][k]
                          for k in reps[a_]}
                    diffs.append({"first": _name(a_), "second": _name(b_), "difference": pt,
                                  "interval": {k: interval([x - y for x, y in zip(reps[a_][k], reps[b_][k])]) for k in reps[a_]}})
            block["paired_differences"] = diffs
            by_rel = []
            for (c, s), ea in sorted(rel_aggs.items(), key=lambda x: (_name(x[0][0]), x[0][1])):
                num, den = math.fsum(x for x, _ in ea.values()), math.fsum(d for _, d in ea.values())
                by_rel.append({"system": _name(c), "relation": s, "command_weighted": num / den,
                               "interval": interval(per_rel[(c, s)]), "environments": len(ea)})
            for s in UNAVAILABLE:
                by_rel.append({"system": "all", "relation": s, "command_weighted": None, "interval": None,
                               "note": "unavailable: no eligible case"})
            block["by_relation_command_weighted"] = by_rel
        else:
            block["note"] = "one environment: point estimates only, no environment-level interval"
        result[part] = block
    if desktop is not None:
        result["desktop_comparison"] = json.loads((Path(desktop) / "summary.json").read_text(encoding="utf-8"))
    result["inputs"] = {"scores_manifest_sha256": _sha((scores / "manifest.json").read_bytes()),
                        "preparation_manifest_sha256": _sha((prep / "manifest.json").read_bytes()),
                        "frozen": dict(PF.FROZEN)}
    result["code"] = {p.name: _sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    result["runtime"] = runtime()
    out.mkdir(parents=True)
    (out / "analysis.json").write_bytes(encode_json(result))
    (out / "report.md").write_text(render(result), encoding="utf-8")
    return result


def _p(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render(r) -> str:
    L = ["# A2.6c baseline results (development and legacy reported separately)", ""]
    L += [f"- {s}" for s in r["scope"]] + [""]
    d = r["development"]
    L += ["## Development: accuracy by system, view and format", "",
          "| System | Command-weighted (95% CI) | Environment-weighted (95% CI) | Unweighted sample | Parents |",
          "|---|---|---|---|---|"]
    for name, s in sorted(d["systems"].items()):
        e, i = s["estimate"], s.get("interval", {})
        ci = lambda k: f"{_p(e[k])} ({_p(i[k]['low'])} to {_p(i[k]['high'])})" if i else _p(e[k])  # noqa: E731
        L.append(f"| {name} | {ci('command_weighted')} | {ci('environment_weighted')} | {_p(e['unweighted'])} | {s['parents']} |")
    L += ["", "## Development: paired differences (first minus second, command-weighted; 95% CI)", "",
          "| First | Second | Difference (95% CI) |", "|---|---|---|"]
    for x in d["paired_differences"]:
        i = x["interval"]["command_weighted"]
        L.append(f"| {x['first']} | {x['second']} | {_p(x['difference']['command_weighted'])} ({_p(i['low'])} to {_p(i['high'])}) |")
    L += ["", "## Coverage of the development commands by relation", "",
          "| Relation | Parents | Eligible | Selected | Performance |", "|---|---|---|---|---|"]
    L += [f"| {c['relation']} | {c['parents']} | {c['eligible']} | {c['selected']} | {c['performance']} |" for c in r["coverage"]]
    lg = r.get("legacy-development", {})
    if lg.get("systems"):
        L += ["", "## Legacy (scannet:scene0010; one environment, point estimates)", "",
              "| System | Command-weighted | Environment-weighted | Unweighted |", "|---|---|---|---|"]
        L += [f"| {n} | {_p(s['estimate']['command_weighted'])} | {_p(s['estimate']['environment_weighted'])} | "
              f"{_p(s['estimate']['unweighted'])} |" for n, s in sorted(lg["systems"].items())]
    if "desktop_comparison" in r:
        dc = r["desktop_comparison"]
        L += ["", f"## Desktop Q8_0/U comparison ({dc['label']})", "",
              f"{dc['same_choice']} of {dc['compared']} decisions equal the 0.5B float32's; largest offered-logit difference "
              f"{dc['max_abs_logit_difference']}; not compared: {len(dc['not_compared'])}."]
    return "\n".join(L) + "\n"
