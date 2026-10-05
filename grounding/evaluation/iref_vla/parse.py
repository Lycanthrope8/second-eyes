"""The A2.2b sample grammar (D75): command text to one grounding-query interpretation, without any scene.

    parse(text, categories=..., colours=...) -> {parse_status, parse_reason, features, analysis_count, interpretation}

Text is matched exactly as given: case-sensitive, single literal spaces, nothing trimmed, folded, normalized,
singularized, repaired or expanded. A command must be wholly one of

    the NP that is|are ALIAS the NP                  (binary and ranking aliases)
    the NP that is|are ALIAS the NP and the NP       (between aliases)
    the NP that is|are ALIAS both of the TAIL        (between aliases; recognized as unsupported plural)

where NP is [other] [big|small] [COLOUR] CATEGORY, a category being an exact label of the supplied vocabulary, possibly
of several words. Every complete analysis is enumerated: every target and anchor boundary, every reading of a noun
phrase (an exact category such as 'white chair' as well as 'white' + 'chair'), every alias. Identical analyses are
merged; anything else is decided by section 3.4's fixed order, never by preferring a reading. The relation aliases and
size words come from the tracked protocol; the parser sees no scene, object ID, source relation label or answer.
"""
from __future__ import annotations

import json

from .protocol import PROTOCOL_PATH, QUERY_REASON, load_protocol

_PROTOCOL = {}
WRAPPERS = (" that is ", " that are ")
BOTH = "both of the "


def _protocol():
    if "p" not in _PROTOCOL:
        _PROTOCOL["p"] = load_protocol(PROTOCOL_PATH)
    return _PROTOCOL["p"]


def _vocabulary(values, name):
    if isinstance(values, (str, bytes)) or values is None:
        raise TypeError(f"{name} must be a collection of strings, not {type(values).__name__}")
    out = frozenset(values)
    if not all(isinstance(v, str) for v in out):
        raise TypeError(f"every {name} entry must be a string")
    return out


def _readings(s, categories, colours, sizes):
    """Every reading of s as [other] [SIZE] [COLOUR] CATEGORY: (other, size, colour, category) tuples."""
    out = []
    for other in (False, True):
        if other and not s.startswith("other "):
            continue
        rest = s[6:] if other else s
        for size in (None,) + sizes:
            if size is not None and not rest.startswith(size + " "):
                continue
            rest2 = rest[len(size) + 1:] if size is not None else rest
            for colour in (None,) + colours:
                if colour is not None and not rest2.startswith(colour + " "):
                    continue
                category = rest2[len(colour) + 1:] if colour is not None else rest2
                if category in categories:
                    out.append((other, size, colour, category))
    return out


def _analyses(text, categories, colours, sizes, aliases):
    """The set of distinct complete analyses: (target NP, relation, k, anchor NPs, plural)."""
    found = set()
    if not text.startswith("the "):
        return found
    body = text[4:]
    for wrapper in WRAPPERS:
        at = body.find(wrapper)
        while at >= 0:
            targets = _readings(body[:at], categories, colours, sizes)
            rest = body[at + len(wrapper):]
            for alias in aliases if targets else ():
                if not rest.startswith(alias["phrase"] + " "):
                    continue
                tail = rest[len(alias["phrase"]) + 1:]
                if alias["relation"] == "between":
                    if tail.startswith(BOTH) and len(tail) > len(BOTH):
                        found.update((t, "between", None, (), True) for t in targets)
                    if tail.startswith("the "):
                        inner = tail[4:]
                        cut = inner.find(" and the ")
                        while cut >= 0:
                            for a in _readings(inner[:cut], categories, colours, sizes):
                                for b in _readings(inner[cut + 9:], categories, colours, sizes):
                                    found.update((t, "between", None, (a, b), False) for t in targets)
                            cut = inner.find(" and the ", cut + 1)
                elif tail.startswith("the "):
                    for a in _readings(tail[4:], categories, colours, sizes):
                        found.update((t, alias["relation"], alias["k"], (a,), False) for t in targets)
            at = body.find(wrapper, at + 1)
    return found


def _features(analysis):
    target, _, _, anchors, plural = analysis
    phrases = (target,) + anchors
    out = set()
    if plural:
        out.add("plural_reference")
    if any(p[0] for p in phrases):
        out.add("coreference")
    if any(p[1] is not None for p in phrases):
        out.add("size_comparison")
    return out


def _node(node_id, phrase, constraints=(), rank=None):
    return {"node_id": node_id, "category": phrase[3], "colours_all": [phrase[2]] if phrase[2] is not None else [],
            "constraints": list(constraints), "rank": rank}


def _interpretation(analysis):
    target, relation, k, anchors, _ = analysis
    if relation in ("closest", "farthest"):
        nodes = [_node("target", target, rank={"relation": relation, "anchor": "anchor_a", "k": k}),
                 _node("anchor_a", anchors[0])]
    elif relation == "between":
        nodes = [_node("target", target, [{"relation": "between", "frame": None, "anchors": ["anchor_a", "anchor_b"]}]),
                 _node("anchor_a", anchors[0]), _node("anchor_b", anchors[1])]
    else:
        nodes = [_node("target", target, [{"relation": relation, "frame": None, "anchors": ["anchor_a"]}]),
                 _node("anchor_a", anchors[0])]
    return {"interpretation_id": "main", "kind": "query", "action": "INSPECT", "root": "target", "nodes": nodes}


def _unsupported(reason, features, count):
    return {"parse_status": "unsupported", "parse_reason": reason, "features": sorted(features), "analysis_count": count,
            "interpretation": {"interpretation_id": "main", "kind": "unsupported", "action": "INSPECT",
                               "reason": QUERY_REASON[reason]}}


def parse(text, *, categories, colours) -> dict:
    """One command's parse: status, reason, features, analysis count and its single interpretation."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")
    if text == "":
        raise ValueError("empty text is invalid at the command contract boundary")
    protocol = _protocol()
    found = _analyses(text, _vocabulary(categories, "categories"), tuple(sorted(_vocabulary(colours, "colours"))),
                      tuple(protocol["size_words"]), protocol["relation_aliases"])
    analyses = sorted(found, key=lambda a: json.dumps(a))
    if not analyses:
        return _unsupported("unrecognized_text", (), 0)
    features = set().union(*(_features(a) for a in analyses))
    if len(analyses) > 1:
        return _unsupported("ambiguous_parse", features, len(analyses))
    for reason in ("plural_reference", "coreference", "size_comparison"):
        if reason in features:
            return _unsupported(reason, features, 1)
    return {"parse_status": "parsed", "parse_reason": None, "features": [], "analysis_count": 1,
            "interpretation": _interpretation(analyses[0])}


def parse_record(parent_command_id, text, *, categories, colours) -> dict:
    """The closed iref_parse record (section 3.4) for one parent command."""
    return {"format_version": 1, "record_type": "iref_parse", "parent_command_id": parent_command_id, "text": text,
            **parse(text, categories=categories, colours=colours)}
