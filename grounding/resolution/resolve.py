"""The offline structured resolver (A2.1e, D71).

resolve() finds the scene objects a structured grounding query describes. It reads no English and no model scores:
the query's complete alternative readings are executed as given, over every supplied object, with the accepted
relation and direction libraries. Uncertain anchors stop their branch (conservative early termination); readings
and branches are aggregated by the fixed precedence of the A2.1e brief, section 6.4. Every unit of work is charged
against the caller's limits before it is done, and a limit failure is a technical result with no target.
"""
from __future__ import annotations

import copy
import json
import time

from ..relations import directions as D
from ..relations import predicates as P
from . import validate as V

EVALUATOR_VERSION = "resolver.v1"
STATUSES = ("resolved", "ambiguous", "no_match", "insufficient_information", "unsupported")
PRIMARY = {"resolved": "unique_target_consensus", "ambiguous": "nonunique_target_or_interpretation",
           "no_match": "no_matching_reference", "insufficient_information": "required_information_unavailable",
           "unsupported": "unsupported_interpretation"}
RANK_STATUS = {"resolved": "resolved", "ambiguous": "ambiguous", "no_match": "no_match",
               "insufficient": "insufficient_information"}
TRUTH = {P.TRUE: "TRUE", P.FALSE: "FALSE", P.UNKNOWN: "UNKNOWN"}
WORK = ("interpretations", "nodes", "constraints", "candidate_checks", "binding_attempts", "pruned_bindings",
        "predicate_calls", "predicate_cache_hits", "rank_calls", "rank_cache_hits", "rank_subsets",
        "max_rank_preflight_subsets", "terminal_outcomes")
STATIC_LIMITS = ("max_interpretations", "max_nodes", "max_constraints")
DYNAMIC = {"candidate_checks": "max_candidate_checks", "binding_attempts": "max_binding_attempts",
           "predicate_calls": "max_predicate_calls", "rank_subsets": "max_rank_subset_evaluations"}
PRUNE_REASON = "between_anchors_not_distinct"


class _OverBudget(Exception):
    def __init__(self, name, limit, used, next_required):
        super().__init__(name)
        self.budget = {"name": name, "limit": limit, "used": used, "next_required": next_required}


def _trace_event(kind, interpretation_id, node_id, bindings, object_ids, relation, frame, outcome, reasons):
    """One closed trace record. Built only when tracing is on."""
    return {"kind": kind, "interpretation_id": interpretation_id, "node_id": node_id,
            "bindings": [{"node_id": k, "object_id": bindings[k]} for k in sorted(bindings)],
            "object_ids": list(object_ids), "relation": relation, "frame": frame, "outcome": outcome,
            "reasons": sorted(set(reasons))}


def _unknown_codes(result):
    """Detail codes for an UNKNOWN relation or direction result, from the library's own reason strings."""
    codes = set()
    for r in result.reasons:
        if r.startswith("missing_centre:"):
            codes.add("missing_centre")
        elif r.startswith("missing_size:"):
            codes.add("missing_size")
        elif r.startswith("missing_rotation:"):
            codes.add("missing_rotation")
        elif r.startswith("missing_semantic_front:"):
            codes.add("missing_semantic_front")
        elif r.startswith(("missing_user_position:", "missing_heading:")):
            codes.add("missing_pose")
        elif r in ("degenerate_viewer_anchor", "vertical_semantic_front"):
            codes.add("degenerate_frame")
        elif r.startswith("boundary:"):
            codes.add("boundary_relation")
    return codes or {"relation_unknown"}  # e.g. degenerate_anchors, support_not_represented: no invented field


def _rank_codes(result):
    codes = set()
    for r in result.reasons:
        if r.startswith("missing_centre:"):
            codes.add("missing_centre")
        elif r.startswith("possible_competitors:"):
            codes.add("possible_rank_competitors")
        elif r == "tie":
            codes.add("rank_tie")
        elif r == "too_few_candidates":
            codes.add("too_few_rank_candidates")
    if result.status == "insufficient" and not codes:
        codes.add("relation_unknown")
    return codes


def _classify(definite, possible):
    """Section 6.3's table, for one node in one binding environment."""
    if len(definite) >= 2:
        return "ambiguous"
    if possible:
        return "insufficient_information"
    return "resolved" if definite else "no_match"


class _Evaluation:
    """The state of one resolve() call: counters, caches, provenance and trace. Nothing survives the call."""

    def __init__(self, scene, command, config, configs, tracing):
        self.scene, self.command = scene, command
        self.by_id = {o["object_id"]: o for o in scene["objects"]}
        self.objects = sorted(self.by_id)
        self.limits = config["limits"]
        self.category_labels, self.colour_labels = set(config["category_labels"]), set(config["colour_labels"])
        self.relation_config, self.direction_config = configs["relation_config"], configs["direction_config"]
        self.scene_scope = (scene["scene_id"], str(scene["scene_revision"]), scene["evidence_profile"])
        self.work = dict.fromkeys(WORK, 0)
        self.counts = dict.fromkeys(STATUSES, 0)
        self.memo, self.rank_memo = {}, {}
        self.assumptions, self.conditional = set(), False
        self.tracing, self.trace, self.trace_bytes, self.truncated = tracing, [], 0, False
        self.targets, self.uncertain, self.rank_definite, self.rank_possible = set(), set(), set(), set()
        self.outcomes = []
        self._relations = self._directions = None

    # -- accounting
    def charge(self, counter):
        limit, used = self.limits[DYNAMIC[counter]], self.work[counter]
        if used + 1 > limit:
            raise _OverBudget(DYNAMIC[counter], limit, used, 1)
        self.work[counter] = used + 1

    def emit(self, *fields):
        if not self.tracing or self.truncated:  # call sites also check, so no arguments are built either
            return
        event = _trace_event(*fields)
        size = len(json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        if len(self.trace) + 1 > self.limits["max_trace_events"] or \
                self.trace_bytes + size > self.limits["max_trace_bytes"]:
            self.truncated = True  # diagnostic truncation only: evaluation carries on unchanged
            return
        self.trace.append(event)
        self.trace_bytes += size

    def consult_wrapper(self, w):
        if w["state"] == "known":
            evidence = w["evidence"]
            if evidence["kind"] == "assumed" or evidence["assumptions"]:
                self.conditional = True
            self.assumptions.update(("scene", self.scene_scope, a) for a in evidence["assumptions"])

    def consult_refs(self, refs):
        for r in refs:
            if r.state != "known":
                continue
            if r.kind == "assumed" or r.assumptions:
                self.conditional = True
            if isinstance(r, D.DirectionFieldRef):
                scope = (r.record_type, tuple(str(x) for x in r.record_id))
            else:
                scope = ("scene", self.scene_scope)
            self.assumptions.update((scope[0], scope[1], a) for a in r.assumptions)

    # -- accepted libraries, built on first use only
    def relations(self):
        if self._relations is None:
            self._relations = P.Relations(self.scene, self.relation_config)
        return self._relations

    def directions(self):
        if self._directions is None:
            self._directions = D.DirectionalRelations(self.scene, self.direction_config)
        return self._directions

    # -- one candidate
    def attributes(self, obj, node):
        value, why = P.TRUE, set()
        if node["category"] is not None:
            w = obj["category"]
            self.consult_wrapper(w)
            if w["state"] == "known":
                value = P.TRUE if w["value"]["model"] == node["category"] else P.FALSE
            else:
                value = P.UNKNOWN
                why.add("unknown_category")
            if value is P.FALSE:
                return value, why
        if node["colours_all"]:
            w = obj["attributes"]["colours"]
            self.consult_wrapper(w)
            if w["state"] == "known":
                colour = P.TRUE if set(node["colours_all"]) <= set(w["value"]) else P.FALSE
            else:
                colour = P.UNKNOWN
                why.add("unknown_colour")
            value = P.AND(value, colour)
        return value, why

    def predicate(self, iid, node_id, env, constraint, target):
        relation, frame = constraint["relation"], constraint["frame"]
        anchors = [env[a] for a in constraint["anchors"]]
        if relation == "near":
            key = (relation, None, tuple(sorted((target, anchors[0]))))
        elif relation == "between":
            key = (relation, None, (target,) + tuple(sorted(anchors)))
        else:
            key = (relation, frame, (target,) + tuple(anchors))
        result = self.memo.get(key)
        if result is not None:
            self.work["predicate_cache_hits"] += 1
        else:
            self.charge("predicate_calls")
            if frame is None:
                result = getattr(self.relations(), relation)(target, *anchors)
            elif frame == "user_heading":
                result = self.directions().evaluate(relation, target, frame=frame, command=self.command)
            elif frame == "user_to_anchor":
                result = self.directions().evaluate(relation, target, frame=frame, anchor_id=anchors[0],
                                                    command=self.command)
            else:
                result = self.directions().evaluate(relation, target, frame=frame, anchor_id=anchors[0])
            self.memo[key] = result
            if self.tracing:
                self.emit("predicate", iid, node_id, env, [target] + anchors, relation, frame, TRUTH[result.value],
                          result.reasons)
        self.consult_refs(result.inputs)
        return result

    def evaluate_node(self, iid, node, env):
        definite, possible, causes = [], [], set()
        for oid in self.objects:
            self.charge("candidate_checks")
            value, why = self.attributes(self.by_id[oid], node)
            if value is P.FALSE:
                continue
            for constraint in node["constraints"]:
                result = self.predicate(iid, node["node_id"], env, constraint, oid)
                if result.value is P.UNKNOWN:
                    why |= _unknown_codes(result)
                value = P.AND(value, result.value)
                if value is P.FALSE:
                    break  # a FALSE may end the conjunction; an UNKNOWN never does
            if value is P.TRUE:
                definite.append(oid)
            elif value is P.UNKNOWN:
                possible.append(oid)
                causes |= why
        return definite, possible, causes

    def rank(self, iid, node, env, definite, possible):
        spec = node["rank"]
        anchor = env[spec["anchor"]]
        dx, px = [x for x in definite if x != anchor], [x for x in possible if x != anchor]
        key = (spec["relation"], spec["k"], anchor, tuple(dx), tuple(px))
        result = self.rank_memo.get(key)
        if result is not None:
            self.work["rank_cache_hits"] += 1
        else:
            limit, used = self.limits["max_rank_subset_evaluations"], self.work["rank_subsets"]
            remaining = limit - used
            if len(px) >= remaining.bit_length():  # 2 ** len(px) > remaining, without building 2 ** len(px)
                raise _OverBudget("max_rank_subset_evaluations", limit, used, remaining + 1)
            bound = 1 << len(px)
            result = self.relations().rank(spec["relation"], anchor, candidates=definite, possible=possible,
                                           k=spec["k"])
            if not 0 <= result.combinations_checked <= bound:
                raise RuntimeError(f"rank checked {result.combinations_checked} subsets; its bound was {bound}")
            self.work["rank_subsets"] += result.combinations_checked
            self.work["rank_calls"] += 1
            self.work["max_rank_preflight_subsets"] = max(self.work["max_rank_preflight_subsets"], bound)
            self.rank_memo[key] = result
            if self.tracing:
                self.emit("rank", iid, node["node_id"], env, result.object_ids, spec["relation"], None,
                          RANK_STATUS[result.status], result.reasons)
        self.consult_refs(result.inputs)
        return result, dx, px

    # -- outcomes
    def terminal(self, iid, node_id, env, action, status, target, ids, codes, incomplete=False):
        self.work["terminal_outcomes"] += 1
        self.counts[status] += 1
        self.outcomes.append({"action": action, "status": status, "target": target, "codes": set(codes),
                              "incomplete": incomplete})
        if self.tracing:
            self.emit("terminal", iid, node_id, env, ids, None, None, status, codes)

    def run(self, interpretations):
        for it in interpretations:
            iid, action = it["interpretation_id"], it["action"]
            if it["kind"] == "unavailable":
                self.terminal(iid, None, {}, action, "insufficient_information", None, [], {it["reason"]}, True)
            elif it["kind"] == "unsupported":
                self.terminal(iid, None, {}, action, "unsupported", None, [], {it["reason"]}, True)
            else:
                outside = set()
                for node in it["nodes"].values():
                    if node["category"] is not None and node["category"] not in self.category_labels:
                        outside.add("unsupported_category_label")
                    if any(c not in self.colour_labels for c in node["colours_all"]):
                        outside.add("unsupported_colour_label")
                if outside:
                    self.terminal(iid, None, {}, action, "unsupported", None, [], outside, True)
                else:
                    self.run_query(it)

    def run_query(self, it):
        iid, action, root, order, nodes = it["interpretation_id"], it["action"], it["root"], it["order"], it["nodes"]
        before = len(self.outcomes)
        stack = [(None, {}, 0)]  # (assignment to make, environment before it, index of the node to evaluate next)
        while stack:
            assignment, env, index = stack.pop()
            if assignment is not None:
                variable, oid = assignment
                self.charge("binding_attempts")
                env = dict(env)
                env[variable] = oid
                if self.tracing:
                    self.emit("binding", iid, variable, env, [oid], None, None, "bound", [])
                if any(a in env and b in env and env[a] == env[b] for a, b in it["between"]):
                    self.work["pruned_bindings"] += 1  # an illegal binding, not a no_match reading
                    if self.tracing:
                        self.emit("prune", iid, variable, env, [oid], "between", None, "pruned", [PRUNE_REASON])
                    continue
            node = nodes[order[index]]
            nid = node["node_id"]
            definite, possible, causes = self.evaluate_node(iid, node, env)
            if self.tracing:
                self.emit("node", iid, nid, env, definite, None, None, _classify(definite, possible), causes)
            ranked = None if node["rank"] is None else self.rank(iid, node, env, definite, possible)
            if nid == root:
                self.finish_root(iid, nid, env, action, definite, possible, causes, ranked)
                continue
            if ranked is None:
                if possible:
                    self.terminal(iid, nid, env, action, "insufficient_information", None, [],
                                  {"uncertain_anchor"} | causes, True)
                    continue
                if not definite:
                    self.terminal(iid, nid, env, action, "no_match", None, [], {"anchor_not_found"})
                    continue
                choices = definite
            else:
                result = ranked[0]
                if result.status == "no_match":
                    self.terminal(iid, nid, env, action, "no_match", None, [],
                                  {"anchor_not_found"} | _rank_codes(result))
                    continue
                if result.status == "insufficient":
                    self.terminal(iid, nid, env, action, "insufficient_information", None, [],
                                  {"uncertain_anchor"} | _rank_codes(result) | causes, True)
                    continue
                choices = sorted(result.object_ids)  # a tie forks: no arbitrary winner
            for oid in reversed(choices):  # LIFO: ascending object IDs are visited first
                stack.append(((nid, oid), env, index + 1))
        if len(self.outcomes) == before:
            self.terminal(iid, None, {}, action, "no_match", None, [], {"no_legal_anchor_binding"})

    def finish_root(self, iid, nid, env, action, definite, possible, causes, ranked):
        if ranked is None:
            status = _classify(definite, possible)
            self.targets.update(definite)
            self.uncertain.update(possible)
            target = definite[0] if status == "resolved" else None
            ids = definite if status in ("resolved", "ambiguous") else []
            codes = {"resolved": set(), "ambiguous": {"multiple_definite_matches"} | causes,
                     "no_match": {"target_not_found"}, "insufficient_information": causes}[status]
        else:
            result, dx, px = ranked
            status = RANK_STATUS[result.status]
            self.targets.update(result.object_ids)
            self.rank_definite.update(dx)
            self.rank_possible.update(px)
            target = result.object_ids[0] if status == "resolved" else None
            ids = list(result.object_ids)
            codes = _rank_codes(result) | (causes if status == "insufficient_information" else set())
        self.terminal(iid, nid, env, action, status, target, ids, codes)

    def result(self, query):
        outs = self.outcomes
        statuses = [o["status"] for o in outs]
        resolved_targets = {o["target"] for o in outs if o["status"] == "resolved"}
        codes = set().union(*(o["codes"] for o in outs))
        if len(resolved_targets) > 1:
            codes.add("divergent_targets")
        if len({o["action"] for o in outs if o["status"] == "resolved"}) > 1:
            codes.add("divergent_actions")
        if "resolved" in statuses and "no_match" in statuses:
            codes.add("resolved_vs_no_match")
        actions = {o["action"] for o in outs}
        if "unsupported" in statuses:
            status = "unsupported"
        elif "insufficient_information" in statuses:
            status = "insufficient_information"
        elif all(s == "resolved" for s in statuses) and len(resolved_targets) == 1 and len(actions) == 1:
            status = "resolved"
        elif all(s == "no_match" for s in statuses):
            status = "no_match"
        else:
            status = "ambiguous"
        if self.conditional:
            codes.add("conditional_evidence")
        incomplete = any(o["incomplete"] for o in outs)
        return {"query_id": query["query_id"], "scene_id": query["scene_id"], "command_id": query["command_id"],
                "scene_revision": query["scene_revision"], "evidence_profile": query["evidence_profile"],
                "status": status,
                "action": next(iter(actions)) if len(actions) == 1 and None not in actions else None,
                "target_id": next(iter(resolved_targets)) if status == "resolved" else None,
                "reason_code": PRIMARY[status], "reason_codes": sorted(codes),
                "candidates": {"target_ids": sorted(self.targets), "uncertain_match_ids": sorted(self.uncertain),
                               "rank_definite_eligible_ids": sorted(self.rank_definite),
                               "rank_possible_eligible_ids": sorted(self.rank_possible),
                               "coverage": "incomplete_due_to_unresolved_reference" if incomplete
                               else "evaluated_branch_union"},
                "assumptions": [{"record_type": t, "record_id": list(rid), "assumption_id": a}
                                for t, rid, a in sorted(self.assumptions)],
                "conditional": self.conditional}


# ---------------------------------------------------------------------------------------------------- the record
def _record(status, result, issues, budget, context, identities, warnings, work, counts, timings, trace=(),
            truncated=False, trace_bytes=0):
    return {"schema_version": 1, "record_type": "resolution_run", "processing_status": status, "result": result,
            "issues": list(issues), "budget": budget,
            "diagnostics": {"evaluator_version": EVALUATOR_VERSION, "context": context, "identities": identities,
                            "warnings": list(warnings), "work": dict(work), "outcome_counts": dict(counts),
                            "timings_s": dict(timings), "trace": list(trace), "trace_truncated": truncated,
                            "trace_bytes": trace_bytes}}


def _check_output(record, scene_ids, limits, tracing):
    """Structure and the invariants the schema can't express. A failure here is a resolver bug, so it raises."""
    errors = list(V._validator("grounding-result.v1.json").iter_errors(record))
    if errors:
        raise RuntimeError(f"the resolver built an invalid resolution_run at {list(errors[0].absolute_path)}: "
                           f"{errors[0].message[:200]}")
    problems, result, diag = [], record["result"], record["diagnostics"]
    if result is not None:
        c = result["candidates"]
        for key in ("target_ids", "uncertain_match_ids", "rank_definite_eligible_ids", "rank_possible_eligible_ids"):
            if c[key] != sorted(set(c[key])) or not set(c[key]) <= scene_ids:
                problems.append(f"candidates.{key}")
        if result["reason_code"] != PRIMARY[result["status"]] or result["reason_codes"] != sorted(result["reason_codes"]):
            problems.append("reason codes")
        if (result["status"] == "resolved") != (result["target_id"] is not None):
            problems.append("target_id against status")
        if result["status"] == "resolved" and (result["action"] is None or result["target_id"] not in scene_ids):
            problems.append("resolved target or action")
        keys = [(a["record_type"], tuple(a["record_id"]), a["assumption_id"]) for a in result["assumptions"]]
        if keys != sorted(set(keys)):
            problems.append("assumptions not sorted and unique")
    if limits is not None:
        for counter, name in DYNAMIC.items():
            if diag["work"][counter] > limits[name]:
                problems.append(f"work.{counter} above {name}")
        if len(diag["trace"]) > limits["max_trace_events"] or diag["trace_bytes"] > limits["max_trace_bytes"]:
            problems.append("trace caps")
    if not tracing and (diag["trace"] or diag["trace_truncated"] or diag["trace_bytes"]):
        problems.append("trace data without tracing")
    if problems:
        raise RuntimeError(f"the resolver broke its result invariants: {', '.join(problems)}")


def _finish(record, scene_ids=frozenset(), limits=None, tracing=False):
    t = time.perf_counter()
    _check_output(record, set(scene_ids), limits, tracing)
    record["diagnostics"]["timings_s"]["result_validation"] = V.finite_seconds(time.perf_counter() - t)
    return record


def invalid_input_record(issues, warnings=(), context=None, validation_s=0.0):
    """The technical record for inputs that fail validation, including the CLI's parse and file errors."""
    record = _record("invalid_input", None, issues, None, context, None, warnings, dict.fromkeys(WORK, 0),
                     dict.fromkeys(STATUSES, 0), {"validation": V.finite_seconds(validation_s), "evaluation": 0.0,
                                                  "result_validation": 0.0})
    return _finish(record)


def resolve(scene_record, command_record, query_record, *, resolver_config, relation_config_path,
            direction_config_path, category_maps=(), trace=False) -> dict:
    """Resolve a structured grounding query against one scene and command; returns a resolution_run record.

    Inputs are decoded JSON records; private copies are evaluated, the caller's records are never changed. Expected
    input problems return processing_status invalid_input; a work limit returns budget_exceeded with no result;
    unexpected errors raise.
    """
    started = time.perf_counter()
    issues = V.check_options(resolver_config, relation_config_path, direction_config_path, category_maps, trace)
    if issues:
        return invalid_input_record(issues, validation_s=time.perf_counter() - started)
    scene, command, query, config = (copy.deepcopy(x) for x in (scene_record, command_record, query_record,
                                                                 resolver_config))
    maps = [copy.deepcopy(m) for m in category_maps]
    query_issues, normalized = V.check_query(query)
    record_issues, warnings = V.check_records(scene, command, maps)
    issues = V.check_config(config) + query_issues + record_issues
    context = None if query_issues else {k: query[k] for k in ("query_id", "scene_id", "scene_revision",
                                                                  "evidence_profile", "command_id")}
    if not issues:
        issues = V.check_context(query, scene, command)
    configs = {}
    if not issues:
        issues, configs = V.load_configs(relation_config_path, direction_config_path)
    if issues:
        return invalid_input_record(issues, warnings, context, time.perf_counter() - started)
    identities = {"scene_sha256": V.canonical_sha256(scene), "command_sha256": V.canonical_sha256(command),
                  "query_sha256": V.canonical_sha256(query), "resolver_config_sha256": V.canonical_sha256(config),
                  "relation_config_identity": configs["relation_config"].identity,
                  "direction_config_identity": configs["direction_config"].identity}
    validated = time.perf_counter()
    run = _Evaluation(scene, command, config, configs, trace)
    sizes = V.request_size(query)
    run.work["interpretations"], run.work["nodes"], run.work["constraints"] = sizes
    result, budget = None, None
    try:
        for name, count in zip(STATIC_LIMITS, sizes):  # before any relation work
            if count > config["limits"][name]:
                raise _OverBudget(name, config["limits"][name], 0, count)
        run.run(normalized)
        result = run.result(query)
    except _OverBudget as e:
        budget = e.budget
    evaluated = time.perf_counter()
    record = _record("completed" if budget is None else "budget_exceeded", result, [], budget, context, identities,
                     warnings, run.work, run.counts,
                     {"validation": V.finite_seconds(validated - started),
                      "evaluation": V.finite_seconds(evaluated - validated), "result_validation": 0.0},
                     run.trace, run.truncated, run.trace_bytes)
    return _finish(record, {o["object_id"] for o in scene["objects"]}, config["limits"], trace)
