"""Tests of A2.4's semantic language, its constrained decoder and its GBNF (D109).

    python grounding/tests/test_a24_language.py [--samples DIR] [--gbnf-validator PATH]

Without --samples the regeneration check of language.v1.json from the pinned vocabulary fails rather than skips; with
--gbnf-validator the corpus also goes through llama.cpp's own grammar engine. Everything here runs without a model:
the decoder is driven by labelled logit doubles. Real-model decoding is Stage A's smoke.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []
TOKENIZER = REPO / "quest-app" / "Assets" / "SecondEyes" / "Models" / "qwen2.5-0.5b-instruct"


PREFIXES = ["", "ch", "chair", "chair ", "chair |", "chair | ", "chair | be", "chair | between | ", "chair | between | door",
            "chair | between | door | ", "orange", "orange juicer [", "orange juicer [or", "chair [white]",
            "chair | near | table", "chair | closest ", "chair | closest 2", "magic 8", "chair | on | night stand [black",
            "chair | near | table ", "chair | farthest 3 | orange"]
FORMS = ["closest 1", "closest 2", "closest 3", "farthest 1", "farthest 2", "farthest 3", "near", "below", "on"]
EXPECTATIONS = Path(__file__).resolve().parent / "fixtures" / "a24" / "mask-expectations.v1.json"


class Oracle:
    """An independent definition of the language's prefixes, written for this test: splitting on the separator, set
    membership and prefix sets built from the specification's lists, with the relation forms written out literally.
    It shares no code with language.Automaton, language.parse_output or the decoder."""

    def __init__(self, spec):
        self.cats, self.cols = set(spec["categories"]["labels"]), list(spec["colours"])
        self.cat_prefixes = {c[:i] for c in self.cats for i in range(len(c) + 1)}
        self.alphabet = set("".join(self.cats) + "".join(self.cols) + "".join(FORMS) + "between | []")

    def obj_complete(self, t):
        if t in self.cats:
            return True
        return any(t.endswith(f" [{c}]") and t[:-len(c) - 3] in self.cats for c in self.cols)

    def obj_prefix(self, t):
        if t in self.cat_prefixes:
            return True
        return any(t[:i] in self.cats and any(f" [{c}]".startswith(t[i:]) for c in self.cols) for i in range(1, len(t)))

    def viable(self, s):
        if not set(s) <= self.alphabet:
            return False
        parts = s.split(" | ")
        done, tail = parts[:-1], parts[-1]
        n = len(done)
        if n > 3 or (n >= 1 and not self.obj_complete(done[0])) or (n >= 2 and done[1] not in FORMS + ["between"]):
            return False
        if n >= 3 and (done[1] != "between" or not self.obj_complete(done[2])):
            return False
        kind = "rel" if n == 1 else "obj"
        more = n in (0, 1) or (n == 2 and done[1] == "between")
        if kind == "rel":
            pre = any(f.startswith(tail) for f in FORMS + ["between"])
            comp = lambda x: x in FORMS + ["between"]   # noqa: E731
        else:
            pre, comp = self.obj_prefix(tail), self.obj_complete
        return pre or (more and any(tail.endswith(q) and comp(tail[:-len(q)]) for q in (" ", " |")))

    def complete(self, s):
        f = s.split(" | ")
        if len(f) == 3:
            return self.obj_complete(f[0]) and f[1] in FORMS and self.obj_complete(f[2])
        return len(f) == 4 and self.obj_complete(f[0]) and f[1] == "between" and all(map(self.obj_complete, f[2:]))

    def masks(self, hf, eos_id):
        """prefix -> sorted allowed IDs, from the tokenizer's own decoding of each ordinary token."""
        special = set(hf.all_special_ids) | {int(i) for i in hf.added_tokens_decoder}
        texts = {}
        for tid in range(len(hf)):
            if tid in special:
                continue
            s = hf.decode([tid])
            if s and set(s) <= self.alphabet:
                texts[tid] = s
        out = {}
        for p in PREFIXES:
            ids = [tid for tid, s in texts.items() if self.viable(p + s)]
            out[p] = sorted(ids + ([eos_id] if self.complete(p) else []))
        return out


def write_expectations(spec, hf, eos_id):
    import hashlib
    m = Oracle(spec).masks(hf, eos_id)
    doc = {"format_version": 1, "record_type": "a24_mask_expectations", "written_by": "the independent Oracle in "
           "test_a24_language.py (not the decoder)", "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct@7ae5576 (pinned files)",
           "language_spec_sha256": hashlib.sha256(Path(REPO / "grounding" / "a24" / "language.v1.json").read_bytes()).hexdigest(),
           "masks": {p: {"count": len(v), "eos": eos_id in v, "ids_sha256": hashlib.sha256(",".join(map(str, v)).encode()).hexdigest()}
                     for p, v in m.items()}}
    EXPECTATIONS.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc



def _safe_console():
    """Never crash on printing: a Windows console piped to a file may not encode every character (cp1252)."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def main() -> int:
    _safe_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.environ.get("SECOND_EYES_IREF_RELEASE_SAMPLES"))
    ap.add_argument("--gbnf-validator", default=None)
    ap.add_argument("--write-mask-expectations", action="store_true",
                    help="(re)write the frozen mask expectations from the independent oracle")
    a = ap.parse_args()
    L = importlib.import_module("grounding.a24.language")
    D = importlib.import_module("grounding.a24.decoder")
    G = importlib.import_module("grounding.a24.gbnf_check")
    T = importlib.import_module("grounding.preparation.iref_vla.tokens")
    PARSE = importlib.import_module("grounding.evaluation.iref_vla.parse")
    RV = importlib.import_module("grounding.resolution.validate")
    spec = L.load_spec()

    print("-- the specification")
    csv = Path(a.samples or "") / "data" / "NYU_Object_Classes.csv"
    if csv.is_file():
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            L.write_spec(csv, Path(td) / "language.v1.json")
            gen = [(Path(td) / n).read_bytes() for n in ("language.v1.json", "language.v1.gbnf")]
            have = [L.SPEC_PATH.read_bytes(), L.SPEC_PATH.with_suffix(".gbnf").read_bytes()]
            same = [g.replace(b"\r\n", b"\n") for g in gen] == [h.replace(b"\r\n", b"\n") for h in have]
        check("language.v1.json and its GBNF regenerate exactly from the pinned NYU_Object_Classes.csv (line endings "
              "aside: a Windows checkout may hold CRLF)", same)
        check("the generator writes LF on every platform", all(b"\r" not in g for g in gen))
    else:
        check("the pinned vocabulary is available for the regeneration check (pass --samples DIR)", False)
    check("893 categories, the category map's rule (every NYU class but 'unknown'), with their recorded hash",
          spec["categories"]["count"] == 893 and "unknown" not in spec["categories"]["labels"]
          and L.labels_sha256(spec["categories"]["labels"]) == spec["categories"]["labels_sha256"])
    for bad in ("a|b", "a [b]", " chair", "chair  lamp"):
        s2 = json.loads(json.dumps(spec))
        s2["categories"]["labels"] = sorted(s2["categories"]["labels"] + [bad])
        s2["categories"]["labels_sha256"] = L.labels_sha256(s2["categories"]["labels"])
        try:
            L.check_spec(s2)
            refused = False
        except L.LanguageError:
            refused = True
        check(f"a label that could make strings ambiguous is refused: {bad!r}", refused)

    print("-- one language: the automaton and the independent reader agree")
    items = G.corpus(spec)
    verdicts = G.python_verdicts(spec, [s for s, _ in items])
    agree = [v["automaton"] == v["reader"] == e for v, (_, e) in zip(verdicts, items)]
    check(f"automaton and reader agree with the expected verdict on all {len(items)} corpus strings "
          f"({sum(e for _, e in items)} valid, {sum(not e for _, e in items)} invalid)", all(agree),
          str([items[i][0] for i, ok in enumerate(agree) if not ok][:5]))
    forms = {L.parse_output(s, spec)["relation"] + str(L.parse_output(s, spec)["k"] or "") for s, e in items if e}
    check("the corpus holds every relation form with and without colours", len(forms) == 10
          and any("[" in s.split(" | ")[0] for s, e in items if e) and any(s.endswith("]") for s, e in items if e))
    rng = random.Random(7)
    labs = spec["categories"]["labels"]
    rand_ok = True
    A = L.Automaton(spec)
    for _ in range(400):
        rel = rng.choice(L.relation_forms())
        objs = [{"category": rng.choice(labs), "colour": rng.choice([None] + list(spec["colours"]))}
                for _ in range(3 if rel == "between" else 2)]
        r, k = (rel.split(" ")[0], int(rel.split(" ")[1])) if " " in rel else (rel, None)
        p = {"target": objs[0], "relation": r, "k": k, "anchors": objs[1:]}
        s = L.render(p)
        rand_ok &= A.accepts(s) and L.parse_output(s, spec) == p
    check("400 random structures render, are accepted, and read back to the same structure", rand_ok)

    print("-- conversion: the accepted query contract, A2.2b's layout")
    cmap_vocab, colours = spec["categories"]["labels"], spec["colours"]
    pairs = [("the white chair that is second closest to the table", "chair [white] | closest 2 | table"),
             ("the chair that is farthest from the black door", "chair | farthest 1 | door [black]"),
             ("the lamp that is near the night stand", "lamp | near | night stand"),
             ("the box that is on the table", "box | on | table"),
             ("the computer that is below the desk", "computer | below | desk"),
             ("the chair that is between the blue table and the window", "chair | between | table [blue] | window")]
    same = []
    for text, s in pairs:
        rp = PARSE.parse(text, categories=cmap_vocab, colours=colours)
        mine = L.interpretation(L.parse_output(s, spec))
        same.append(rp["parse_status"] == "parsed" and rp["interpretation"] == mine
                    and L.render(L.from_interpretation(rp["interpretation"])) == s)
    check("the converted interpretation equals the accepted rule parser's for the same meaning, and reads back",
          all(same), str(same))
    amb = PARSE.parse("the orange juicer that is third nearest to the orange", categories=cmap_vocab, colours=colours)
    check("why colours follow the category: the rule parser finds 'orange juicer' ambiguous (colour + 'juicer' or the "
          "label), while v1's 'orange juicer | closest 3 | orange' has one reading",
          amb["parse_reason"] == "ambiguous_parse" and L.parse_output("orange juicer | closest 3 | orange", spec)
          ["target"] == {"category": "orange juicer", "colour": None})
    scene = {"scene_id": "fixture.a24.scene", "scene_revision": 0, "evidence_profile": "annotated"}
    command = {"command_id": "fixture.a24.command"}
    q = L.query(L.interpretation(L.parse_output("chair [white] | between | table | window [black]", spec)), scene, command)
    found, _ = RV.check_query(q)
    check("the full query passes the accepted query validator (identity fields from code, frame null, INSPECT)",
          not found and q["query_id"] == "fixture.a24.command.q", str(found[:2]))
    try:
        L.from_interpretation({"interpretation_id": "main", "kind": "query", "action": "INSPECT", "root": "target",
                               "nodes": [{"node_id": "target", "category": "chair", "colours_all": [], "rank": None,
                                          "constraints": [{"relation": "left", "frame": "user_heading", "anchors": []}]}]})
        refused = False
    except L.LanguageError:
        refused = True
    check("an interpretation v1 cannot express (a directional relation) is refused, never approximated", refused)

    print("-- masks with the real pinned tokenizer")
    tok = T.load_pinned_tokenizer(TOKENIZER)
    hf = tok._backend
    table = D.TokenTable(hf)
    C = D.Constraint(spec, table)
    check("the table holds every ordinary token (151,643) and no special token; the end token is <|im_end|> (151645)",
          len(table.keys) == 151643 and table.eos_id == 151645 and table.eos_id not in table.bytes_of)
    import hashlib
    special = set(hf.all_special_ids) | {int(i) for i in hf.added_tokens_decoder}
    asc = [tid for tid, b in table.bytes_of.items() if b.isascii()]
    check(f"the decoder's byte table equals the tokenizer's own decoding for all {len(asc)} ASCII tokens",
          all(hf.decode([tid]) == table.bytes_of[tid].decode("ascii") for tid in asc))
    if a.write_mask_expectations:
        write_expectations(spec, hf, table.eos_id)
    oracle_masks = Oracle(spec).masks(hf, table.eos_id)
    mine = {p: sorted(C.allowed(C.A.walk(p))) for p in PREFIXES}
    check(f"for {len(PREFIXES)} prefixes, the decoder's masks equal an independently written oracle's over all ordinary "
          f"tokens (the oracle shares no code with the automaton, the reader or the decoder)", mine == oracle_masks,
          str([p for p in PREFIXES if mine[p] != oracle_masks[p]]))
    frozen = json.loads(EXPECTATIONS.read_text(encoding="utf-8")) if EXPECTATIONS.is_file() else None
    check("the masks also equal the frozen expectations file (prefix, count, end token, ID-list hash)",
          frozen is not None and all(frozen["masks"][p] == {"count": len(v), "eos": table.eos_id in v,
                                     "ids_sha256": hashlib.sha256(",".join(map(str, v)).encode()).hexdigest()}
                                     for p, v in mine.items()), "write it with --write-mask-expectations")
    states = [C.A.walk(p) for p in PREFIXES]
    al = C.allowed(C.A.walk("chair"))
    names = {hf.convert_ids_to_tokens(i) for i in al}
    check("after 'chair', tokens spanning the separator are allowed (the space-prefixed '|' and '['); the end token is "
          "not",
          {"Ġ|", "Ġ["} <= names and table.eos_id not in al, str(sorted(names)))
    al2 = C.allowed(C.A.walk("chair | near |"))
    check("after 'chair | near |', a token joining the space and a category word (the space-prefixed 'table') is "
          "allowed",
          hf.convert_tokens_to_ids("Ġtable") in al2)
    allowed_any = set().union(*(set(C.allowed(s)) for s in states))
    special = set(hf.all_special_ids) | {int(i) for i in hf.added_tokens_decoder}
    check("no special token is ever allowed except the end token; no token with a non-ASCII byte is allowed",
          not (allowed_any & special - {table.eos_id}) and all(table.bytes_of[i].isascii() for i in allowed_any - {table.eos_id}))
    lab_set = set(spec["categories"]["labels"])
    short = next(x for x in spec["categories"]["labels"] if any(y.startswith(x + " ") for y in lab_set))
    longer = next(y for y in spec["categories"]["labels"] if y.startswith(short + " "))
    al3 = C.allowed(C.A.walk(f"chair | near | {short}"))
    check(f"a complete string that can still grow ('{short}' to '{longer}') allows both the end token and the space",
          table.eos_id in al3 and hf.convert_tokens_to_ids("Ġ") in al3)
    canon = all(C.accepts_ids(tok.encode(s) + [table.eos_id]) == (True, len(tok.encode(s))) for s, e in items if e)
    check("every valid corpus string's own tokenization, then the end token, is a path the masks allow", canon)
    alt_ok = True
    for s, e in items:
        if not e:
            continue
        chars = [hf.convert_tokens_to_ids("Ġ" if c == " " else c) for c in s]
        alt_ok &= C.accepts_ids(chars + [table.eos_id])[0]
        mid = len(s) // 2
        split = tok.encode(s[:mid]) + tok.encode(s[mid:]) if s[mid] != " " else tok.encode(s)
        alt_ok &= C.accepts_ids(split + [table.eos_id])[0]
    check("alternative tokenizations stay possible: one token per character, and the string tokenized in two halves",
          alt_ok)
    early = all(not C.accepts_ids(tok.encode(s)[:-1] + [table.eos_id])[0] for s, e in items[:20] if e)
    check("the end token before a complete string is refused by the masks", early)

    print("-- the decoder, with labelled logit doubles")
    V = 151936

    def double(prefer):
        def step(prev):
            row = [0.0] * V
            for i, v in prefer(prev).items():
                row[i] = v
            return row
        return step
    target = tok.encode("chair [white] | closest 2 | table")
    seq = iter(target + [table.eos_id])
    r1 = D.decode(double(lambda prev: {next(seq): 5.0, 0: 9.0}), C, 64)
    v1 = D.validate(r1, spec)
    check("a model preferring a forbidden token (id 0, '!') still decodes the allowed path, then validates",
          r1["status"] == "completed" and r1["text"] == "chair [white] | closest 2 | table" and v1["valid"], str(r1))
    r2 = D.decode(double(lambda prev: {table.eos_id: -50.0}), C, 30)
    check("a model that never prefers the end token stops at the limit: length_exhausted, not ASK, not a result",
          r2["status"] == "length_exhausted" and r2["steps"] == 30 and not D.validate(r2, spec)["valid"])
    ties = D.decode(double(lambda prev: {}), C, 3)
    first = min(C.allowed(C.A.start))
    check("an exact tie goes to the lowest allowed token ID", ties["token_ids"][0] == first)
    Cbad = D.Constraint(spec, table)
    Cbad._allowed[Cbad.A.start] = {}
    check("an empty allowed set is reported as empty_allowed_set", D.decode(double(lambda p: {}), Cbad, 5)["status"]
          == "empty_allowed_set")
    Cbad2 = D.Constraint(spec, table)
    Cbad2._allowed[Cbad2.A.start] = {table.eos_id: None}
    r3 = D.decode(double(lambda p: {}), Cbad2, 5)
    check("an end token at an incomplete string is reported as premature_eos (a mask fault), never as a result",
          r3["status"] == "premature_eos" and not D.validate(r3, spec)["valid"])
    check("each result carries its mask and model times", r1["mask_ms"] >= 0 and r1["model_ms"] >= 0)

    print("-- GBNF")
    g = L.gbnf(spec)
    check("the GBNF has the root, relation, object, colour and category rules, one alternative per label",
          g.count("\ncategory ::= ") == 1 and g.split("category ::= ")[1].count(" | ") == 892 and "root ::= " in g)
    sp = json.loads(json.dumps(spec))
    sp["categories"]["labels"] = ["a\"b", "c\\d"]
    check("GBNF literals escape quotes and backslashes", '"a\\"b"' in L.gbnf(sp) and '"c\\\\d"' in L.gbnf(sp))

    class Fake:
        def __init__(self):
            self.A = L.Automaton(spec)

        def __call__(self, cmd, **k):
            s = Path(cmd[2]).read_bytes().decode("utf-8")
            ok = self.A.accepts(s)
            return type("P", (), {"stdout": "Input string is valid according to the grammar.\n" if ok else
                                  "Input string is invalid according to the grammar.\n", "stderr": ""})()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        fake_validator = Path(td) / "validator"
        fake_validator.write_bytes(b"labelled double")
        s = G.check(validator=fake_validator, out=Path(td) / "out", runner=Fake())
        check("the check's plumbing with a labelled double validator: verdicts read, no disagreement",
              s["disagreements"] == 0 and s["unreadable_verdicts"] == 0 and s["strings"] == len(items))
    if a.gbnf_validator:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            s = G.check(validator=a.gbnf_validator, out=Path(td) / "out")
            check(f"llama.cpp's own validator ({Path(a.gbnf_validator).name}) agrees on all {s['strings']} strings",
                  s["disagreements"] == 0 and s["unreadable_verdicts"] == 0, json.dumps(s)[:300])
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
