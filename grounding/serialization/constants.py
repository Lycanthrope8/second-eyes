"""Fixed model-text constants of serializer version 2 (A2.1d, D69).

Copied, not imported, from Second_Eyes_A2_1d_Serializer_Implementation_Brief.md Appendix A, which equals the accepted
indexed audit (analysis/A2.1d_indexed_token_sizing). Changing any of these changes the model input: that needs a new
serializer version, not an edit here. Parameter and category values are not constants: they come from the loaded
configuration files on every call.
"""
from __future__ import annotations

SERIALIZER_VERSION = 2
COORDINATES = "coordinates_v2"
AUGMENTED = "coordinates_relations_v2"
FORMATS = (COORDINATES, AUGMENTED)

AXES = {"handedness": "right", "up": "+z", "units": "m"}
OBJECT_COLUMNS = ("id", "category", "colours", "center_m", "size_m", "rotation_xyzw", "rotation_support",
                  "semantic_front", "conditional_fields")
# Columns whose wrapper carries evidence of its own; rotation_support belongs to the rotation_xyzw wrapper.
EVIDENCE_COLUMNS = ("category", "colours", "center_m", "size_m", "rotation_xyzw", "semantic_front")
POSE_CONDITIONAL_FIELDS = ("position_m", "heading_xy")
UNKNOWN_TEXT = "null means unknown; [] means known-empty colours"
CONDITIONAL_FIELDS_TEXT = "named displayed fields depend on assumptions"
COVERAGE = (
    "Indices are zero-based positions in objects. Every declared domain is complete; T=true, F=false, U=unknown. A "
    "single T/F/U broadcasts over legal tuples; [] means an empty domain. objects uses one character per target. "
    "ordered_pairs uses target rows and anchor columns. unordered_pairs uses upper-triangle rows: row i lists j=i+1 "
    "onward. target_anchor_pairs uses [a,b,row] for every a<b in index order; character t describes target t. '-' "
    "marks excluded repeated-object cells, never unknown or false. Conditional membership is a Boolean broadcast or "
    "matching rows of 0/1; 1 means consulted assumptions. Distances use numeric upper-triangle rows, with null "
    "unknown. Undeclared or excluded tuples have no truth value. left=opposite(right) and behind=opposite(in_front_of), "
    "exchanging T/F and retaining U on legal tuples.")
DEFINITIONS = (
    "Centres and user positions are scene-frame metres; sizes are full local-axis lengths; rotations are xyzw "
    "local-to-scene quaternions; semantic front is separate. Directions project centre offsets onto horizontal axes. "
    "User-heading origin is user position, front is normalized heading, right=(front_y,-front_x). User-to-anchor "
    "origin is anchor centre, v=normalized XY(anchor-user), front=-v, right=(v_y,-v_x). Intrinsic origin is anchor "
    "centre, front=normalized XY(semantic_front), right=(front_y,-front_x). Direction score greater than band is T, "
    "below negative band F, otherwise U. Viewer-anchor horizontal length and semantic-front horizontal norm at or "
    "below their cutoffs make that frame U. Near uses shortest solid-box distance. Above uses target-bottom minus "
    "anchor-top gap and footprint overlap. On uses that gap within the support/penetration limits, overlap and "
    "eligible nearly upright anchor. Below is separated-below OR the eligible-furniture branch using bottom offset, "
    "top margin and overlap. Inside tests all target corners against eligible anchor outer-box bounds; this does not "
    "establish a cavity. Footprint overlap is intersection divided by smaller projected area. Between uses centre "
    "projection inside the anchor segment, lateral limit and each anchor overlap limit when vertically overlapping. "
    "Missing required evidence is U; existing relation gates and eligibility rules apply. Non-directional le(x,t,b): "
    "T if x<=t-b, F if x>t+b, else U; ge(x,t,b): T if x>=t+b, F if x<t-b, else U. Centre distances are 3D and "
    "support closest/farthest ranks 1-3 only after candidate eligibility is decided; adjacent gaps at or below "
    "rank_tie_m form linked ties. No ranking population or winner is supplied here. Derived states and distances "
    "remain conditional on their consulted assumptions. UNKNOWN is neither FALSE nor an executable target.")

# Line order (brief section 5): static derived blocks before pose, viewer-dependent blocks after it.
# (block name, relation, frame, domain)
STATIC_BLOCKS = (("near", "near", None, "unordered_pairs"), ("above", "above", None, "ordered_pairs"),
                 ("below", "below", None, "ordered_pairs"), ("on", "on", None, "ordered_pairs"),
                 ("inside", "inside", None, "ordered_pairs"), ("between", "between", None, "target_anchor_pairs"),
                 ("intrinsic.right", "right", "object_intrinsic", "ordered_pairs"),
                 ("intrinsic.in_front_of", "in_front_of", "object_intrinsic", "ordered_pairs"))
DYNAMIC_BLOCKS = (("user_heading.right", "right", "user_heading", "objects"),
                  ("user_heading.in_front_of", "in_front_of", "user_heading", "objects"),
                  ("user_to_anchor.right", "right", "user_to_anchor", "ordered_pairs"),
                  ("user_to_anchor.in_front_of", "in_front_of", "user_to_anchor", "ordered_pairs"))


def work_slots(n: int) -> int:
    """Planned augmented work: predicate calls plus distance slots (brief section 6)."""
    return 9 * n * (n - 1) + n * (n - 1) * (n - 2) // 2 + 2 * n
