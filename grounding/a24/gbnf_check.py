"""The v1 language against the pinned llama.cpp's own grammar engine (D109): language compatibility only.

A deterministic corpus covers every construction (each relation form, with and without a target colour and an anchor
colour, both anchors of `between`) with the labels most likely to break a grammar (the colour-word labels, digits,
multiword and prefix-sharing labels), plus near-misses that must be refused. Optionally it adds every retained label
string of a preparation. Each string goes to llama.cpp's `test-gbnf-validator` (built from b11277 with
LLAMA_BUILD_TESTS=ON), and its verdict is compared with the Python automaton and the independent reader.

This establishes that the grammar text means the same language in llama.cpp. It does not establish Quest execution,
token-level behaviour or numerical agreement: those belong to A2.7.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from . import language as L

TRICKY = ("orange", "orange juicer", "orange plastic cap", "yellow pepper", "magic 8ball", "chair", "table", "table lamp",
          "night stand", "air conditioner", "mortar and pestle", "door", "window", "shelves", "tv stand")


def _labels(spec):
    have = set(spec["categories"]["labels"])
    picked = [x for x in TRICKY if x in have]
    rest = spec["categories"]["labels"]
    picked += [rest[i] for i in range(0, len(rest), max(1, len(rest) // 25))]
    return list(dict.fromkeys(picked))


def corpus(spec) -> list:
    """[(string, expected)]: the valid constructions and the refusals, deterministic."""
    labs, cols = _labels(spec), spec["colours"]
    out, i = [], 0
    for form in L.relation_forms():
        for tcol in (None, cols[i % len(cols)]):
            for acol in (None, cols[(i + 3) % len(cols)]):
                t, a, b = labs[i % len(labs)], labs[(i + 5) % len(labs)], labs[(i + 9) % len(labs)]
                anchors = [{"category": a, "colour": acol}]
                if form == L.BETWEEN:
                    anchors.append({"category": b, "colour": cols[(i + 7) % len(cols)] if acol else None})
                rel, k = (form.split(" ")[0], int(form.split(" ")[1])) if " " in form else (form, None)
                out.append((L.render({"target": {"category": t, "colour": tcol}, "relation": rel, "k": k,
                                      "anchors": anchors}), True))
                i += 1
    good = [s for s, _ in out]
    bad = []
    for k, s in enumerate(good):
        variants = [s + " ", " " + s, s + "\n", s.upper(), s.replace(" | ", "|", 1), s.replace(" | ", " |  ", 1),
                    s[:-1], s.rsplit(" | ", 1)[0], s + " | door"]
        bad.append(variants[k % len(variants)])
    bad += ["chair | closest 0 | table", "chair | closest 4 | table", "chair | farthest | table", "chair | above | table",
            "chair | inside | box", "chair [grey] | near | table", "chair [White] | near | table", "chair [] | near | table",
            "chair | between | table", "chair | near | table | lamp", "| near | table", "chair | near |", "",
            "unknown | near | table", "white chair | near | table", "chair | near | table [white", "chair [white]] | near | table"]
    out += [(s, False) for s in dict.fromkeys(bad) if s not in set(good)]
    return out


def python_verdicts(spec, strings) -> list:
    A = L.Automaton(spec)
    rows = []
    for s in strings:
        try:
            L.parse_output(s, spec)
            reader = True
        except L.LanguageError:
            reader = False
        rows.append({"string": s, "automaton": A.accepts(s), "reader": reader})
    return rows


def run_validator(validator, grammar_text: str, strings, runner=subprocess.run) -> list:
    """llama.cpp's verdict per string: True, False, or None when its output is not one of its two messages."""
    out = []
    with tempfile.TemporaryDirectory() as td:
        g = Path(td) / "grammar.gbnf"
        g.write_bytes(grammar_text.encode("utf-8"))   # LF on every platform (Windows text mode would write CRLF)
        for k, s in enumerate(strings):
            f = Path(td) / f"input-{k}.txt"
            f.write_bytes(s.encode("utf-8"))
            p = runner([str(validator), str(g), str(f)], capture_output=True, text=True, timeout=60)
            text = (p.stdout or "") + (p.stderr or "")
            if "Input string is valid according to the grammar." in text:
                out.append(True)
            elif "Input string is invalid according to the grammar." in text:
                out.append(False)
            else:
                out.append(None)
    return out


def check(*, validator, out, labels_file=None, spec_path=L.SPEC_PATH, runner=subprocess.run) -> dict:
    """Writes out/ (corpus, verdicts, summary). labels_file: optional JSONL with a 'label' string per row (a
    preparation's retained label strings), all expected valid."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; each check writes a new folder")
    spec = L.load_spec(spec_path)
    items = corpus(spec)
    if labels_file:
        extra = [json.loads(x)["label"] for x in Path(labels_file).read_text(encoding="utf-8").splitlines() if x.strip()]
        seen = {s for s, _ in items}
        items += [(s, True) for s in dict.fromkeys(extra) if s not in seen]
    strings = [s for s, _ in items]
    py = python_verdicts(spec, strings)
    grammar = L.gbnf(spec)
    cpp = run_validator(validator, grammar, strings, runner=runner)
    rows = [dict(p, expected=e, llama_cpp=c,
                 agree=(c is not None and p["automaton"] == p["reader"] == c == e)) for (s, e), p, c in zip(items, py, cpp)]
    vb = Path(validator).read_bytes() if Path(validator).is_file() else b""
    summary = {"format_version": 1, "record_type": "a24_gbnf_check", "language_id": L.LANGUAGE_ID,
               "spec_sha256": L.spec_sha256(spec_path), "grammar_sha256": hashlib.sha256(grammar.encode("utf-8")).hexdigest(),
               "validator": {"path": str(validator), "sha256": hashlib.sha256(vb).hexdigest() if vb else None},
               "llama_cpp_pin": (Path(__file__).resolve().parents[2] / "native" / "llama.cpp.pin").read_text().strip(),
               "strings": len(rows), "expected_valid": sum(e for _, e in items),
               "expected_invalid": sum(not e for _, e in items), "from_labels_file": len(strings) - len(corpus(spec)),
               "disagreements": sum(not r["agree"] for r in rows),
               "unreadable_verdicts": sum(r["llama_cpp"] is None for r in rows),
               "scope": "language compatibility with llama.cpp's grammar engine; no Quest execution, token-level or "
                        "numerical claim"}
    out.mkdir(parents=True)
    (out / "grammar.gbnf").write_bytes(grammar.encode("utf-8"))
    (out / "verdicts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary
