"""The association guard (A4 Delivery 1; docs/resolution.md).

In a perceived scene (scene v3) the localization keeps uncertain observations apart instead of merging them, and marks
each such object possible_duplicate, naming the objects it may be the same as. The accepted resolver doesn't read that
mark, so a ranking could quietly pick one of a possible pair. This guard runs after the resolver and turns a resolved
answer that rests on a possible duplicate into the existing ambiguity outcome: status ambiguous, reason
nonunique_target_or_interpretation, detail code uncertain_association, and the pair (or group) as the target candidates.
It changes nothing else, and nothing at all for any other scene or outcome.

An answer rests on a possible duplicate when
- its target is one, or
- the query ranks with k of 2 or more and any object eligible for that ranking is one: whether two observations are one
  object changes the count. The result doesn't record the ranking's order (only the optional trace does), so every
  eligible object counts, a safe superset of the objects ranked ahead of the target.
A repeated anchor needs no rule here: the resolver already tries every object the anchor could be and answers only when
all of them give the same target.
"""
from __future__ import annotations

import copy

DETAIL = "uncertain_association"


def _ranks_past_first(query: dict) -> bool:
    for it in query.get("interpretations", []):
        for node in it.get("nodes", []) if it.get("kind") == "query" else []:
            rank = node.get("rank")
            if rank is not None and rank.get("k", 1) >= 2:
                return True
    return False


def guard(scene: dict, query: dict, run: dict) -> dict:
    """The resolution_run, its result made ambiguous when the answer rests on a possible duplicate; otherwise unchanged."""
    if scene.get("evidence_profile") != "perceived" or run.get("processing_status") != "completed":
        return run
    result = run["result"]
    if result["status"] != "resolved":
        return run
    dup = {o["object_id"]: o["observation"]["association"]["may_duplicate"] for o in scene["objects"]
           if o["observation"]["association"]["status"] == "possible_duplicate"}
    involved = set()
    if result["target_id"] in dup:
        involved |= {result["target_id"], *dup[result["target_id"]]}
    if _ranks_past_first(query):
        eligible = set(result["candidates"]["rank_definite_eligible_ids"]) | set(result["candidates"]["rank_possible_eligible_ids"])
        for oid in sorted(eligible & dup.keys()):
            involved |= {oid, *dup[oid]}
    if not involved:
        return run
    out = copy.deepcopy(run)
    r = out["result"]
    r["status"], r["target_id"], r["reason_code"] = "ambiguous", None, "nonunique_target_or_interpretation"
    r["reason_codes"] = sorted(set(r["reason_codes"]) | {DETAIL})
    r["candidates"]["target_ids"] = sorted(involved)
    return out
