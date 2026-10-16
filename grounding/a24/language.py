"""A2.4's semantic output language, v1 (D109): what the text-only parser writes, one versioned specification for every
use of it.

**The form** (`language.v1.json`):

    TARGET | RELATION | ANCHOR              RELATION: closest N, farthest N (N = 1, 2, 3), near, below, on
    TARGET | between | ANCHOR | ANCHOR

Each of TARGET and ANCHOR is a category label, optionally followed by a colour in brackets: `chair [white]`. Colours
come after the category because four labels begin with or equal a colour word ("orange juicer", "orange plastic cap",
"yellow pepper", "orange"); a colour written first would make those strings ambiguous. No label contains `|`, `[` or
`]`, so every accepted string has exactly one reading.

**One specification, four uses**, all built from the same file:

1. `Automaton`: the character-level automaton the constrained decoder walks (an NFA, determinized lazily);
2. `parse_output`: an independent reader by splitting and set membership, used to validate decoded text. It shares no
   code with the automaton;
3. `interpretation` and `query`: deterministic structural conversion to the accepted `grounding_query` contract, the
   same node layout as A2.2b's rule parser. No repair, no rule-parser answer;
4. `gbnf`: the same language for llama.cpp's grammar engine (the pinned b11277).

**The vocabulary** is the pinned IRef-VLA `data/NYU_Object_Classes.csv` (commit 741cf02, SHA-256 3c65ff2e...) minus
`unknown`, exactly the category map's `model_vocabulary`: fixed before any scene, target, annotation or result is read,
and never narrowed per scene. Deterministic code, never the model, supplies the query's identity fields, the action
(INSPECT, the evaluation protocol's placeholder) and the frame (null: v1 has no directional relation).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "language.v1.json"
LANGUAGE_ID = "a24.semantic.v1"
SEP, OPEN, CLOSE = " | ", " [", "]"
RANKED = ("closest", "farthest")
SINGLE = ("near", "below", "on")
BETWEEN = "between"
RANKS = (1, 2, 3)
COLOURS = ("black", "blue", "brown", "gray", "maroon", "navy", "orange", "purple", "white", "yellow")
UNKNOWN = "unknown"
SOURCE = {"repository": "IRef-VLA", "commit": "741cf0289a6ee7c8f0763058f40bb3788b76f8e0",
          "path": "data/NYU_Object_Classes.csv", "sha256": "3c65ff2e4ad3a60a0a4ad87f01ef864f6848cdff8b8c7792e43e2389d9585f91"}


class LanguageError(ValueError):
    """A string outside the language, or a specification that does not hold together."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def labels_sha256(labels) -> str:
    return _sha("".join(x + "\n" for x in labels).encode("utf-8"))


def relation_forms() -> list:
    return [f"{r} {k}" for r in RANKED for k in RANKS] + list(SINGLE) + [BETWEEN]


# ------------------------------------------------------------------------------------------------ the specification
def build_spec(csv_bytes: bytes) -> dict:
    """The v1 specification from the pinned vocabulary file's bytes (the CSV is read with the release reader's rule)."""
    from ..adapters.iref_vla import release as R
    if _sha(csv_bytes) != SOURCE["sha256"]:
        raise LanguageError("the vocabulary is not the pinned NYU_Object_Classes.csv")
    labels = sorted(x for x in R.read_vocabulary(csv_bytes) if x != UNKNOWN)
    spec = {"format_version": 1, "record_type": "a24_semantic_language", "language_id": LANGUAGE_ID, "decision": "D109",
            "form": {"binary": "TARGET | RELATION | ANCHOR", "between": "TARGET | between | ANCHOR | ANCHOR",
                     "object": "CATEGORY, optionally followed by ' [COLOUR]'", "separator": SEP,
                     "colour_open": OPEN, "colour_close": CLOSE},
            "relations": {"ranked": list(RANKED), "ranks": list(RANKS), "single_anchor": list(SINGLE),
                          "two_anchors": BETWEEN, "forms": relation_forms()},
            "colours": list(COLOURS),
            "categories": {"source": dict(SOURCE), "rule": "every nyuClass of the pinned file except 'unknown', sorted by "
                           "code point: the category map's model_vocabulary", "count": len(labels),
                           "labels_sha256": labels_sha256(labels), "labels": labels},
            "supplied_by_code": {"identity": "query, scene and command IDs, revision and evidence profile from the derived "
                                            "scene and command", "action": "INSPECT (the evaluation protocol's placeholder)",
                                 "frame": "null for every constraint (v1 has no directional relation)",
                                 "nodes": "target, anchor_a, anchor_b in written order (A2.2b's layout)"},
            "outside_v1": ["above", "inside", "left", "right", "in front of", "behind", "size words", "plural references",
                           "coreference", "ASK or unsupported outputs (positive-reference pilot)"],
            "eos": "<|im_end|>", "max_new_tokens": 96}
    check_spec(spec)
    return spec


def check_spec(spec) -> None:
    """The properties every use relies on: one reading per string, and a vocabulary that is the pinned one."""
    labels = spec["categories"]["labels"]
    if labels != sorted(set(labels)) or labels_sha256(labels) != spec["categories"]["labels_sha256"]:
        raise LanguageError("the category list is not sorted, unique and equal to its recorded hash")
    for x in labels + list(spec["colours"]):
        if not x or not x.isascii() or any(c in x for c in "|[]\"\\\n\r\t") or x != x.strip() or "  " in x:
            raise LanguageError(f"label {x!r} could make the language ambiguous or unprintable")
    if spec["relations"]["forms"] != relation_forms() or tuple(spec["colours"]) != COLOURS:
        raise LanguageError("the relation forms or colours differ from v1")


def load_spec(path=SPEC_PATH) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("record_type") != "a24_semantic_language" or spec.get("language_id") != LANGUAGE_ID:
        raise LanguageError(f"{path} is not the {LANGUAGE_ID} specification")
    check_spec(spec)
    return spec


def spec_sha256(path=SPEC_PATH) -> str:
    return _sha(Path(path).read_bytes())


# ------------------------------------------------------------------------------------------------ the independent reader
def _object(text: str, sets, where: str) -> dict:
    cats, cols = sets
    if text.endswith(CLOSE):
        i = text.rfind(OPEN)
        if i <= 0:
            raise LanguageError(f"{where}: a closing bracket without an opening one")
        cat, col = text[:i], text[i + len(OPEN):-len(CLOSE)]
        if col not in cols:
            raise LanguageError(f"{where}: {col!r} is not one of the ten colours")
    else:
        cat, col = text, None
    if cat not in cats:
        raise LanguageError(f"{where}: {cat!r} is not a category label")
    return {"category": cat, "colour": col}


_SETS = {}


def _sets(spec) -> tuple:
    """(categories, colours) as sets, cached by the category list's hash; the specification is never modified."""
    key = (spec["categories"]["labels_sha256"], tuple(spec["colours"]))
    if key not in _SETS:
        _SETS[key] = (frozenset(spec["categories"]["labels"]), frozenset(spec["colours"]))
    return _SETS[key]


def parse_output(text: str, spec) -> dict:
    """The decoded string's structure, read by splitting on the separator: {target, relation, k, anchors}. Raises
    LanguageError naming the first problem. Independent of `Automaton` by construction."""
    sets = _sets(spec)
    if not isinstance(text, str) or not text:
        raise LanguageError("empty output")
    fields = text.split(SEP)
    if len(fields) not in (3, 4):
        raise LanguageError(f"{len(fields)} fields; the language has 3 (one anchor) or 4 (between)")
    rel = fields[1]
    if len(fields) == 4 and rel != BETWEEN:
        raise LanguageError("four fields need the relation 'between'")
    if len(fields) == 3 and rel == BETWEEN:
        raise LanguageError("'between' needs two anchors")
    if rel in SINGLE or rel == BETWEEN:
        name, k = rel, None
    else:
        parts = rel.split(" ")
        if len(parts) != 2 or parts[0] not in RANKED or parts[1] not in ("1", "2", "3"):
            raise LanguageError(f"{rel!r} is not a relation of v1")
        name, k = parts[0], int(parts[1])
    return {"target": _object(fields[0], sets, "target"), "relation": name, "k": k,
            "anchors": [_object(f, sets, f"anchor {i + 1}") for i, f in enumerate(fields[2:])]}


def render(parsed) -> str:
    """The canonical string of a structure (the inverse of parse_output)."""
    def obj(o):
        return o["category"] + (f"{OPEN}{o['colour']}{CLOSE}" if o["colour"] else "")
    rel = f"{parsed['relation']} {parsed['k']}" if parsed["relation"] in RANKED else parsed["relation"]
    return SEP.join([obj(parsed["target"]), rel] + [obj(a) for a in parsed["anchors"]])


# ------------------------------------------------------------------------------------------------ the query contract
def _node(node_id, o, constraints=(), rank=None) -> dict:
    return {"node_id": node_id, "category": o["category"], "colours_all": [o["colour"]] if o["colour"] else [],
            "constraints": list(constraints), "rank": rank}


def interpretation(parsed) -> dict:
    """Structural conversion to one grounding_query interpretation: A2.2b's node layout, INSPECT, null frames."""
    t, a, rel = parsed["target"], parsed["anchors"], parsed["relation"]
    if rel in RANKED:
        nodes = [_node("target", t, rank={"relation": rel, "anchor": "anchor_a", "k": parsed["k"]}), _node("anchor_a", a[0])]
    elif rel == BETWEEN:
        nodes = [_node("target", t, [{"relation": BETWEEN, "frame": None, "anchors": ["anchor_a", "anchor_b"]}]),
                 _node("anchor_a", a[0]), _node("anchor_b", a[1])]
    else:
        nodes = [_node("target", t, [{"relation": rel, "frame": None, "anchors": ["anchor_a"]}]), _node("anchor_a", a[0])]
    return {"interpretation_id": "main", "kind": "query", "action": "INSPECT", "root": "target", "nodes": nodes}


def query(interp, scene, command, suffix=".q") -> dict:
    """The full grounding_query: identity fields from the derived scene and command (deterministic code). A perceived
    scene (scene v3, A4) gets a format-2 query, the first format that takes its profile; every other scene, format 1."""
    return {"schema_version": 2 if scene["evidence_profile"] == "perceived" else 1, "record_type": "grounding_query", "query_id": command["command_id"] + suffix,
            "scene_id": scene["scene_id"], "scene_revision": scene["scene_revision"],
            "evidence_profile": scene["evidence_profile"], "command_id": command["command_id"], "interpretations": [interp]}


def from_interpretation(interp) -> dict:
    """The structure an interpretation in A2.2b's layout encodes, or LanguageError when v1 cannot express it."""
    if interp.get("kind") != "query" or interp.get("root") != "target":
        raise LanguageError(f"not a v1 query interpretation ({interp.get('kind')})")
    nodes = {n["node_id"]: n for n in interp["nodes"]}

    def obj(n):
        if len(n["colours_all"]) > 1 or n["constraints"] or (n["node_id"] != "target" and n["rank"]):
            raise LanguageError(f"node {n['node_id']} is outside v1")
        return {"category": n["category"], "colour": n["colours_all"][0] if n["colours_all"] else None}
    t = nodes["target"]
    if t["rank"]:
        if t["constraints"] or t["rank"]["anchor"] != "anchor_a" or set(nodes) != {"target", "anchor_a"}:
            raise LanguageError("a ranked target outside v1")
        return {"target": obj(dict(t, rank=None)), "relation": t["rank"]["relation"], "k": t["rank"]["k"],
                "anchors": [obj(nodes["anchor_a"])]}
    if len(t["constraints"]) != 1:
        raise LanguageError("the target needs exactly one constraint in v1")
    c = t["constraints"][0]
    if c["frame"] is not None:
        raise LanguageError("v1 has no frames")
    want = ["anchor_a", "anchor_b"] if c["relation"] == BETWEEN else ["anchor_a"]
    if c["anchors"] != want or set(nodes) != {"target", *want} or c["relation"] not in SINGLE + (BETWEEN,):
        raise LanguageError(f"relation {c['relation']} with anchors {c['anchors']} is outside v1")
    return {"target": obj(dict(t, constraints=[])), "relation": c["relation"], "k": None,
            "anchors": [obj(nodes[x]) for x in want]}


# ------------------------------------------------------------------------------------------------ the automaton
class Automaton:
    """The language as a character NFA, determinized lazily: a state is a frozenset of NFA states. Every live state can
    still reach acceptance (the grammar has no dead ends), so a non-empty state always has a continuation or is
    accepting."""

    def __init__(self, spec):
        self.trans, self.eps = [], []
        self._cache, self._chars = {}, {}
        cats, cols = spec["categories"]["labels"], spec["colours"]
        start, final = self._new(), self._new()

        def lit(s):
            a = cur = self._new()
            for ch in s:
                nxt = self._new()
                self.trans[cur].setdefault(ch, set()).add(nxt)
                cur = nxt
            return a, cur

        def trie(words):
            a = self._new()
            end = self._new()
            for w in words:
                cur = a
                for ch in w:
                    nxt = next(iter(self.trans[cur][ch])) if ch in self.trans[cur] else None
                    if nxt is None:
                        nxt = self._new()
                        self.trans[cur][ch] = {nxt}
                    cur = nxt
                self.eps[cur].add(end)
            return a, end

        def seq(*frags):
            for (_, e), (s, _) in zip(frags, frags[1:]):
                self.eps[e].add(s)
            return frags[0][0], frags[-1][1]

        def alt(*frags):
            a, b = self._new(), self._new()
            for s, e in frags:
                self.eps[a].add(s)
                self.eps[e].add(b)
            return a, b

        def opt(frag):
            a, b = self._new(), self._new()
            self.eps[a] |= {frag[0], b}
            self.eps[frag[1]].add(b)
            return a, b

        def obj():
            return seq(trie(cats), opt(seq(lit(OPEN), trie(cols), lit(CLOSE))))

        rel1 = alt(*(lit(f) for f in relation_forms() if f != BETWEEN))
        binary = seq(obj(), lit(SEP), rel1, lit(SEP), obj())
        between = seq(obj(), lit(SEP + BETWEEN + SEP), obj(), lit(SEP), obj())
        root = alt(binary, between)
        self.eps[start].add(root[0])
        self.eps[root[1]].add(final)
        self.final = final
        self.start = self._closure({start})

    def _new(self) -> int:
        self.trans.append({})
        self.eps.append(set())
        return len(self.trans) - 1

    def _closure(self, states) -> frozenset:
        stack, seen = list(states), set(states)
        while stack:
            for t in self.eps[stack.pop()]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    def step(self, state: frozenset, ch: str) -> frozenset:
        key = (state, ch)
        if key not in self._cache:
            nxt = set()
            for s in state:
                nxt |= self.trans[s].get(ch, set())
            self._cache[key] = self._closure(nxt) if nxt else frozenset()
        return self._cache[key]

    def chars(self, state: frozenset) -> list:
        """The characters that keep the string inside the language from this state, sorted."""
        if state not in self._chars:
            self._chars[state] = sorted({c for s in state for c in self.trans[s]})
        return self._chars[state]

    def accepting(self, state: frozenset) -> bool:
        return self.final in state

    def walk(self, text: str, state=None) -> frozenset:
        state = self.start if state is None else state
        for ch in text:
            state = self.step(state, ch)
            if not state:
                return state
        return state

    def accepts(self, text: str) -> bool:
        s = self.walk(text)
        return bool(s) and self.accepting(s)


# ------------------------------------------------------------------------------------------------ GBNF (llama.cpp)
def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def gbnf(spec) -> str:
    """The same language in llama.cpp's GBNF: rule names, literals and alternatives generated from the specification."""
    rel1 = " | ".join(_q(f) for f in relation_forms() if f != BETWEEN)
    lines = [f"# {LANGUAGE_ID}: generated from language.v1.json (categories {spec['categories']['labels_sha256'][:12]}...)",
             f"root ::= object {_q(SEP)} relation {_q(SEP)} object | object {_q(SEP + BETWEEN + SEP)} object {_q(SEP)} object",
             f"relation ::= {rel1}",
             f"object ::= category ({_q(OPEN)} colour {_q(CLOSE)})?",
             "colour ::= " + " | ".join(_q(c) for c in spec["colours"]),
             "category ::= " + " | ".join(_q(c) for c in spec["categories"]["labels"])]
    return "\n".join(lines) + "\n"


def write_spec(csv_path, out=SPEC_PATH) -> dict:
    """Writes language.v1.json and language.v1.gbnf from the pinned vocabulary file (a one-time build step)."""
    spec = build_spec(Path(csv_path).read_bytes())
    Path(out).write_bytes((json.dumps(spec, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))   # LF everywhere
    Path(out).with_suffix(".gbnf").write_bytes(gbnf(spec).encode("utf-8"))
    return spec
