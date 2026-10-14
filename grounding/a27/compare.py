"""A2.7 comparison (D115): the desktop Q8_0 decodes against the float32 reference, on the PC that holds the scenes.

    python -m grounding.a27 compare --bundle BUNDLE --desktop DESKTOP --prep PREP --work WORK --out FOLDER

Judges the desktop halves of two frozen criteria:
- criterion 4: the native allowed tokens equal the Python decoder's at every checked prefix (no difference);
- criterion 7: at least 99% of the 2,715 parents decode to exactly the reference's tokens and status, every difference
  listed beside both outputs and both views' outcomes. A technical failure is a failure, listed apart, never agreement.
Every Q8_0 output is resolved through A2.4's own path (the decoder's validator, the language's interpretation, the
resolver's checker). Wherever the output equals the reference, its two views must reproduce A2.4's recorded outcomes
exactly: if one does not, the comparison stops, since the comparison itself would be wrong.
Writes compare.json and report.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..a24 import decoder as DC
from ..a24 import evaluate as E
from ..a24 import language as L
from ..evaluation.iref_vla.protocol import encode_json

POLICY = "a27.compare.v1"
AGREEMENT_BAR = 0.99


class CompareError(RuntimeError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _jl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def text_of(token_ids, table) -> str:
    return b"".join(table.bytes_of[t] for t in token_ids if t != table.eos_id).decode("ascii", errors="replace")


def resolve_views(res, parent, work, maps, checker) -> dict:
    """Both views' outcomes for a validated decoder result, by the same steps as A2.4's parser evaluation."""
    b = Path(work) / parent["scene"] / "bundle"
    if parent["scene"] not in maps:
        maps[parent["scene"]] = json.loads((b / "model_records" / "category-map.json").read_text(encoding="utf-8"))
    views = {}
    for v in E.VIEWS:
        vp = parent["view_paths"][v]
        if not res["valid"]:
            views[v] = {"outcome": f"decode:{res['status']}" if res["status"] != "completed" else "invalid_output",
                        "target_id": None, "technical": True}
            continue
        scene = json.loads((b / vp["scene_path"]).read_text(encoding="utf-8"))
        command = json.loads((b / vp["command_path"]).read_text(encoding="utf-8"))
        rr = checker.resolve(L.interpretation(res["parsed"]), scene, command, maps[parent["scene"]])
        views[v] = {"outcome": rr["outcome"], "target_id": rr.get("target_id"), "technical": False,
                    "candidate_ids": rr.get("candidate_ids")}
    return views


def judge(rows, results, masks, resolve) -> dict:
    """The verdicts from the bundle rows, the desktop results and masks; resolve(row, result) gives a result's views."""
    by = {r["parent_command_id"]: r for r in results}
    same, differ, failures, mismatch_self = 0, [], [], []
    for r in rows:
        d = by.get(r["parent_command_id"])
        ref = r["reference"]
        if d is None or d["status"] == "technical_failure":
            failures.append({"parent_command_id": r["parent_command_id"], "stratum": r["stratum"],
                             "error": None if d is None else d.get("error")})
            continue
        views = resolve(r, d)
        if d["token_ids"] == ref["token_ids"] and d["status"] == ref["status"]:
            same += 1
            if views != ref["views"]:
                mismatch_self.append(r["parent_command_id"])
        else:
            differ.append({"parent_command_id": r["parent_command_id"], "stratum": r["stratum"],
                           "headset": r["headset"], "reference": {"status": ref["status"], "text": ref["text"]},
                           "q8_0": {"status": d["status"], "text": d.get("text")},
                           "outcomes": {v: {"reference": ref["views"][v]["outcome"], "q8_0": views[v]["outcome"]}
                                        for v in views}})
    if mismatch_self:
        raise CompareError(f"{len(mismatch_self)} unchanged outputs did not reproduce A2.4's recorded outcomes, e.g. "
                           f"{mismatch_self[0]}: the comparison's resolution path is wrong")
    n = len(rows)
    agreement = same / n if n else 0.0
    return {"parents": n, "identical": same, "agreement": agreement, "bar": AGREEMENT_BAR,
            "criterion_7_desktop": "met" if agreement >= AGREEMENT_BAR else "not met",
            "differences": differ, "technical_failures": failures,
            "outcome_changes": sum(1 for x in differ for v in x["outcomes"].values() if v["reference"] != v["q8_0"]),
            "criterion_4_desktop": "met" if masks["differing"] == 0 else "not met",
            "masks": {k: masks[k] for k in ("prefixes", "differing")}, "reproduced_unchanged": same}


def compare(*, bundle, desktop, prep, work, out) -> dict:
    from ..a24 import labels as LB
    from ..a24 import prep as PR
    from ..preparation.iref_vla import tokens as T
    bundle, desktop, out = Path(bundle), Path(desktop), Path(out)
    bman_bytes = (bundle / "manifest.json").read_bytes()
    bman = json.loads(bman_bytes)
    dman = json.loads((desktop / "manifest.json").read_text(encoding="utf-8"))
    if dman["lock"]["bundle_manifest_sha256"] != _sha(bman_bytes):
        raise CompareError("the desktop run used another bundle")
    body = (desktop / "results.jsonl").read_bytes()
    if dman["outputs"]["results.jsonl"] != _sha(body):
        raise CompareError("the desktop results changed since the run")
    if _sha((bundle / "requests.jsonl").read_bytes()) != bman["files"]["requests.jsonl"]:
        raise CompareError("the bundle's requests changed")
    if not dman["tokenizer_against_base"]["equal"]:
        raise CompareError("the export's tokenizer differs from the base GGUF's")
    rows = _jl(bundle / "requests.jsonl")
    results = [json.loads(x) for x in body.decode("utf-8").splitlines() if x.strip()]
    masks = json.loads((desktop / "masks.json").read_text(encoding="utf-8"))
    spec = L.load_spec()
    table = DC.TokenTable(T.load_pinned_tokenizer(PR.DEV.TOKENIZER)._backend)
    parents = {p["parent_command_id"]: p for p in E._parents(prep, work)}
    checker, maps = LB.Checker(spec), {}
    for d in results:
        if d["status"] != "technical_failure":
            d["text"] = text_of(d["token_ids"], table)

    def resolve(row, d):
        res = DC.validate({"status": d["status"], "token_ids": d["token_ids"], "text": d["text"]}, spec)
        return resolve_views(res, parents[row["parent_command_id"]], work, maps, checker)
    verdict = judge(rows, results, masks, resolve)
    rec = {"format_version": 1, "record_type": "a27_desktop_comparison", "policy": POLICY,
           "candidate": bman["candidate"], "gguf_sha256": dman["export"]["gguf_sha256"],
           "library": {"sha256": dman["library"]["sha256"], "version": dman["library"]["version"]},
           "inputs": {"bundle_manifest_sha256": _sha(bman_bytes),
                      "desktop_manifest_sha256": _sha((desktop / "manifest.json").read_bytes())},
           "tokenizer_against_base": dman["tokenizer_against_base"], "decodes": dman["decodes"],
           "subset": {k: bman["subset"][k] for k in ("size", "allocation", "ids_sha256", "mask_sample_sha256")},
           **verdict}
    out.mkdir(parents=True, exist_ok=False)
    (out / "compare.json").write_bytes(encode_json(rec))
    (out / "report.md").write_text(render(rec), encoding="utf-8")
    return rec


def _cell(text) -> str:
    """Table-safe: a parser output's separators would otherwise end the cell."""
    return (text or "").replace("|", "\\|")


def render(r) -> str:
    pct = f"{r['agreement'] * 100:.2f}%"
    lines = ["# A2.7 Delivery 1: the desktop comparison", "",
             f"Candidate `{r['candidate']['model_name']}` (identity `{r['candidate']['identity_sha256'][:12]}`), GGUF "
             f"`{r['gguf_sha256'][:12]}`, libse_llama `{r['library']['sha256'][:12]}` ({r['library']['version']}).", "",
             "Desktop halves only: the headset halves of criteria 4 and 7 come in Delivery 3. Nothing here is Quest "
             "numerical acceptance.", "",
             "| Criterion | Result | Verdict |", "|---|---|---|",
             f"| 4, desktop: allowed tokens equal the Python decoder's | {r['masks']['differing']} of "
             f"{r['masks']['prefixes']} prefixes differ | {r['criterion_4_desktop']} |",
             f"| 7, desktop: identical to float32 on at least 99% | {r['identical']} of {r['parents']} ({pct}) | "
             f"{r['criterion_7_desktop']} |",
             f"| The export's tokenizer equals the base GGUF's | {r['tokenizer_against_base']['fields']} fields "
             f"compared | {'equal' if r['tokenizer_against_base']['equal'] else 'differs'} |", "",
             f"Technical failures: {len(r['technical_failures'])}. Outcome changes among the differences: "
             f"{r['outcome_changes']}. Every unchanged output reproduced A2.4's recorded outcomes in both views.", "",
             f"Decode statuses: {r['decodes']['statuses']}; median {r['decodes']['median_ms']} ms per command on the "
             "desktop (not a headset figure).", ""]
    if r["differences"]:
        lines += ["## Differences from the float32 reference", "",
                  "| Parent | Relation | Headset subset | float32 | Q8_0 | Outcomes (full inventory; source-known) |",
                  "|---|---|---|---|---|---|"]
        for x in r["differences"]:
            o = x["outcomes"]
            lines.append(f"| `{x['parent_command_id'][-12:]}` | {x['stratum']} | {'yes' if x['headset'] else 'no'} | "
                         f"`{_cell(x['reference']['text'])}` | `{_cell(x['q8_0']['text'])}` | "
                         + "; ".join(f"{o[v]['reference']} to {o[v]['q8_0']}" for v in o) + " |")
        lines.append("")
    if r["technical_failures"]:
        lines += ["## Technical failures", ""] + [f"- `{x['parent_command_id']}`: {x['error']}"
                                                   for x in r["technical_failures"]] + [""]
    lines += [f"Headset subset (frozen): {r['subset']['size']} parents, allocation {r['subset']['allocation']}, "
              f"IDs `{r['subset']['ids_sha256'][:12]}`, mask sample `{r['subset']['mask_sample_sha256'][:12]}`.", ""]
    return "\n".join(lines)
