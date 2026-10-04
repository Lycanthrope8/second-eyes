"""Input validation for the offline resolver (A2.1e, D71).

Every expected input problem becomes an issue {label, path, code, message}; nothing here raises for bad input. The
accepted contract validator checks scene, command and category-map records unchanged; query and resolver-config
records have their own schemas and are never registered in the contract's dispatch. JSON Schema patterns are
checked again as full-string matches, because Python's $ also matches before a final newline. Resolver limits and
rank k must also be Python integers (D72): JSON Schema's integer type admits whole-valued floats such as 10000.0,
and they are rejected, never converted.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from pathlib import Path

import jsonschema

from ..contract import validate as contract
from ..relations import directions as D
from ..relations import predicates as P

SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"
ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
LABEL = re.compile(r"\S(?:.*\S)?")
LOCAL_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")
_VALIDATORS = {}


def issue(label, path, code, message):
    return {"label": label, "path": path, "code": code, "message": message}


def canonical_sha256(record) -> str:
    """SHA-256 of sorted-key compact UTF-8 JSON; non-finite numbers are refused."""
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validator(name):
    if name not in _VALIDATORS:
        schema = json.loads((SCHEMAS / name).read_text(encoding="utf-8"))
        _VALIDATORS[name] = jsonschema.Draft202012Validator(schema)
    return _VALIDATORS[name]


def _path(parts) -> str:
    return "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in parts)


def _schema_issues(label, record, schema_name, code):
    out = []
    for e in sorted(_validator(schema_name).iter_errors(record), key=lambda e: [str(p) for p in e.absolute_path]):
        detail = jsonschema.exceptions.best_match(e.context) if e.context else e
        where = list(e.absolute_path)
        text = detail.message if len(detail.message) <= 240 else detail.message[:237] + "..."
        if detail is not e and list(detail.absolute_path) != where:
            text += f" (at {_path(detail.absolute_path)})"
        out.append(issue(label, _path(where), code, text))
    return out


def _full_matches(label, value, path, code, out):
    """Re-check every ID and label as a full-string match (the schema's patterns let a final newline through)."""
    if isinstance(value, dict):
        for key, item in value.items():
            _full_matches(label, item, path + [key], code, out)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _full_matches(label, item, path + [i], code, out)
    elif isinstance(value, str):
        key = next((p for p in reversed(path) if isinstance(p, str)), "")
        rule = {"query_id": ID, "scene_id": ID, "command_id": ID, "config_id": ID, "vocabulary_id": ID,
                "category": LABEL, "colours_all": LABEL, "category_labels": LABEL, "colour_labels": LABEL,
                "interpretation_id": LOCAL_ID, "node_id": LOCAL_ID, "root": LOCAL_ID, "anchors": LOCAL_ID,
                "anchor": LOCAL_ID}.get(key)
        if rule is not None and not rule.fullmatch(value):
            out.append(issue(label, _path(path), code, f"{value!r} must match the whole {key} pattern"))


# ------------------------------------------------------------------------------------------------- the options
def check_options(resolver_config, relation_config_path, direction_config_path, category_maps, trace):
    out = []
    if not isinstance(resolver_config, dict):
        out.append(issue("resolver_config", "$", "E_RESOLVER_CONFIG", "the resolver configuration must be an object"))
    for name, value in (("relation_config_path", relation_config_path),
                        ("direction_config_path", direction_config_path)):
        if not isinstance(value, (str, Path)):
            out.append(issue("options", f"$.{name}", "E_RESOLVER_CONFIG", f"{name} must be a path"))
    if not isinstance(category_maps, (list, tuple)) or not all(isinstance(m, dict) for m in category_maps):
        out.append(issue("options", "$.category_maps", "E_RESOLVER_CONFIG",
                         "category_maps must be a list or tuple of objects"))
    if not isinstance(trace, bool):
        out.append(issue("options", "$.trace", "E_RESOLVER_CONFIG", "trace must be true or false"))
    return out


def _not_integer(label, path, value, code):
    return issue(label, path, code, f"an integer value is required: {value!r} is a {type(value).__name__}, and a "
                                    "float is not accepted even when it is whole-valued")


def check_config(config):
    out = _schema_issues("resolver_config", config, "resolver-config.v1.json", "E_RESOLVER_CONFIG")
    if not out:
        _full_matches("resolver_config", config, [], "E_RESOLVER_CONFIG", out)
        for name, value in config["limits"].items():  # every limit; the schema has made access safe (D72)
            if type(value) is not int:
                out.append(_not_integer("resolver_config", f"$.limits.{name}", value, "E_RESOLVER_CONFIG"))
    return out


# --------------------------------------------------------------------------------------------------- the query
def _deps(node):
    deps = [a for c in node["constraints"] for a in c["anchors"]]
    if node["rank"] is not None:
        deps.append(node["rank"]["anchor"])
    return deps


def _constraint_key(c):
    anchors = sorted(c["anchors"]) if c["relation"] == "between" else list(c["anchors"])
    return json.dumps([c["relation"], c["frame"], anchors], separators=(",", ":")), anchors


def _check_graph(label, i, interp, out):
    """Graph rules for one QUERY interpretation; returns the topological order, or None after an issue."""
    base = f"$.interpretations[{i}]"
    nodes = interp["nodes"]
    ids = [n["node_id"] for n in nodes]
    found = len(out)
    for j, nid in enumerate(ids):
        if ids.index(nid) != j:
            out.append(issue(label, f"{base}.nodes[{j}].node_id", "E_QUERY_GRAPH", f"node ID {nid!r} is repeated"))
        if nid.startswith("obj_"):
            out.append(issue(label, f"{base}.nodes[{j}].node_id", "E_QUERY_GRAPH",
                             f"node ID {nid!r} is reserved: node IDs must not begin obj_ or name a scene object"))
    if interp["root"] not in ids:
        out.append(issue(label, f"{base}.root", "E_QUERY_GRAPH", f"root {interp['root']!r} is not a node"))
    for j, node in enumerate(nodes):
        refs = [(f"{base}.nodes[{j}].constraints[{k}].anchors[{m}]", a)
                for k, c in enumerate(node["constraints"]) for m, a in enumerate(c["anchors"])]
        if node["rank"] is not None:
            refs.append((f"{base}.nodes[{j}].rank.anchor", node["rank"]["anchor"]))
        for path, ref in refs:
            if ref not in ids:
                out.append(issue(label, path, "E_QUERY_GRAPH", f"{ref!r} is not a node of this interpretation"))
            elif ref == node["node_id"]:
                out.append(issue(label, path, "E_QUERY_GRAPH", f"node {ref!r} depends on itself"))
        seen = {}
        for k, c in enumerate(node["constraints"]):
            key, _ = _constraint_key(c)
            if key in seen:
                out.append(issue(label, f"{base}.nodes[{j}].constraints[{k}]", "E_QUERY_SCHEMA",
                                 f"constraint repeats constraints[{seen[key]}]: redundant input"))
            seen.setdefault(key, k)
    if len(out) > found:
        return None
    by_id = {n["node_id"]: n for n in nodes}
    deps = {nid: sorted(set(_deps(by_id[nid]))) for nid in ids}
    waiting = {nid: len(deps[nid]) for nid in ids}
    users = {nid: [] for nid in ids}
    for nid in ids:
        for d in deps[nid]:
            users[d].append(nid)
    ready = [nid for nid in ids if waiting[nid] == 0]
    heapq.heapify(ready)
    order = []
    while ready:  # Kahn's algorithm: dependencies first, the lexicographically smallest ready node next
        nid = heapq.heappop(ready)
        order.append(nid)
        for u in users[nid]:
            waiting[u] -= 1
            if waiting[u] == 0:
                heapq.heappush(ready, u)
    if len(order) < len(ids):
        cyclic = sorted(nid for nid in ids if waiting[nid] > 0)
        out.append(issue(label, f"{base}.nodes", "E_QUERY_GRAPH", f"dependency cycle through {', '.join(cyclic)}"))
        return None
    reach, todo = {interp["root"]}, [interp["root"]]
    while todo:  # iterative reachability along dependencies
        for d in deps[todo.pop()]:
            if d not in reach:
                reach.add(d)
                todo.append(d)
    unused = [nid for nid in ids if nid not in reach]
    if unused:
        out.append(issue(label, f"{base}.nodes", "E_QUERY_GRAPH",
                         f"not reachable from root {interp['root']!r}: {', '.join(sorted(unused))}"))
        return None
    return order


def check_query(query):
    """Issues for the query, and its normalized form when there are none."""
    label = "query"
    out = _schema_issues(label, query, "grounding-query.v1.json", "E_QUERY_SCHEMA")
    if out:
        return out, None
    _full_matches(label, query, [], "E_QUERY_SCHEMA", out)
    for i, interp in enumerate(query["interpretations"]):  # every ranked node, at its original input path (D72)
        if interp["kind"] == "query":
            for j, node in enumerate(interp["nodes"]):
                if node["rank"] is not None and type(node["rank"]["k"]) is not int:
                    out.append(_not_integer(label, f"$.interpretations[{i}].nodes[{j}].rank.k", node["rank"]["k"],
                                            "E_QUERY_SCHEMA"))
    if out:
        return out, None
    ids = [it["interpretation_id"] for it in query["interpretations"]]
    for i, iid in enumerate(ids):
        if ids.index(iid) != i:
            out.append(issue(label, f"$.interpretations[{i}].interpretation_id", "E_QUERY_SCHEMA",
                             f"interpretation ID {iid!r} is repeated"))
    orders = {}
    for i, interp in enumerate(query["interpretations"]):
        if interp["kind"] == "query":
            orders[interp["interpretation_id"]] = _check_graph(label, i, interp, out)
    if out:
        return out, None
    return [], normalize(query, orders)


def normalize(query, orders):
    """Interpretations sorted by ID; nodes with sorted colours and canonically ordered constraints."""
    result = []
    for interp in sorted(query["interpretations"], key=lambda it: it["interpretation_id"]):
        if interp["kind"] != "query":
            result.append(dict(interp))
            continue
        nodes = {}
        for n in interp["nodes"]:
            cons = []
            for c in n["constraints"]:
                key, anchors = _constraint_key(c)
                cons.append((key, {"relation": c["relation"], "frame": c["frame"], "anchors": anchors}))
            nodes[n["node_id"]] = {"node_id": n["node_id"], "category": n["category"],
                                   "colours_all": sorted(n["colours_all"]),
                                   "constraints": [c for _, c in sorted(cons, key=lambda kc: kc[0])],
                                   "rank": dict(n["rank"]) if n["rank"] is not None else None}
        between = sorted({tuple(c["anchors"]) for n in nodes.values() for c in n["constraints"]
                          if c["relation"] == "between"})
        result.append({"interpretation_id": interp["interpretation_id"], "kind": "query", "action": interp["action"],
                       "root": interp["root"], "order": orders[interp["interpretation_id"]], "nodes": nodes,
                       "between": between})
    return result


def request_size(query):
    """(interpretations, nodes, constraints) as the static limits count them."""
    queries = [it for it in query["interpretations"] if it["kind"] == "query"]
    return (len(query["interpretations"]), sum(len(it["nodes"]) for it in queries),
            sum(len(n["constraints"]) for it in queries for n in it["nodes"]))


# ------------------------------------------------------------------------------------------------- the records
def check_records(scene, command, maps):
    """(errors, warnings) for the scene, command and category maps, through the accepted validator unchanged."""
    errors = []
    for label, record, wanted in (("scene", scene, "scene"), ("command", command, "command_context")):
        if isinstance(record, dict) and record.get("record_type") not in (wanted, None):
            errors.append(issue(label, "$.record_type", "E_RECORD_ROLE",
                                f"the {label} argument holds a {record.get('record_type')!r} record"))
    for i, m in enumerate(maps):
        if m.get("record_type") not in ("category_map", None):
            errors.append(issue(f"category_map[{i}]", "$.record_type", "E_RECORD_ROLE",
                                f"category_maps[{i}] holds a {m.get('record_type')!r} record"))
    named = [("scene", scene), ("command", command)] + [(f"category_map[{i}]", m) for i, m in enumerate(maps)]
    found = [issue(i.file, i.path, i.code, i.message) for i in contract.validate(named)]
    errors += [x for x in found if x["code"].startswith("E_")]
    warnings = [x for x in found if not x["code"].startswith("E_")]
    if errors:
        return errors, warnings
    supplied = sorted(m["map_id"] for m in maps)
    if maps and scene["category_map"] not in supplied:  # D70's boundary rule, applied locally
        errors.append(issue("scene", "$.category_map", "E_CATEGORY_MAP_MISMATCH",
                            f"the scene needs map {scene['category_map']!r}; supplied: {', '.join(supplied)}"))
    if command["scene_id"] != scene["scene_id"]:
        errors.append(issue("command", "$.scene_id", "E_QUERY_CONTEXT",
                            f"command {command['command_id']!r} is for scene {command['scene_id']!r}, "
                            f"not {scene['scene_id']!r}"))
    return errors, warnings


def check_context(query, scene, command):
    out = []
    for key, want in (("scene_id", scene["scene_id"]), ("scene_revision", scene["scene_revision"]),
                      ("evidence_profile", scene["evidence_profile"]), ("command_id", command["command_id"])):
        if query[key] != want:
            out.append(issue("query", f"$.{key}", "E_QUERY_CONTEXT",
                             f"query {key} {query[key]!r} does not match the supplied record's {want!r}"))
    return out


def load_configs(relation_config_path, direction_config_path):
    """The accepted relation and direction configurations, loaded fresh into call-local objects."""
    out, loaded = [], {}
    for name, path, loader in (("relation_config", relation_config_path, P.load_config),
                               ("direction_config", direction_config_path, D.load_direction_config)):
        try:
            loaded[name] = loader(Path(path))
        except (OSError, ValueError, jsonschema.exceptions.ValidationError) as e:  # as the serializer's E_CONFIG
            out.append(issue(name, str(path), "E_CONFIG", f"{type(e).__name__}: {e}"))
    return out, loaded


def finite_seconds(value) -> float:
    return value if math.isfinite(value) and value >= 0 else 0.0
