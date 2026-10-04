"""Expected mappings for the indexed codec, fixed before the codec was written (indexed audit brief, section 8).

The literal examples are copied from section 4 of Second_Eyes_A2_1d_Indexed_Audit_Brief.md, with the mappings its text
states. Nothing here is produced by the codec; the audit checks the codec against these.
"""

O_ABC = ["A", "B", "C"]

# (literal block text, domain, {tuple of IDs: (state, conditional)}); between tuples are (target, anchor_a, anchor_b)
LITERAL_BLOCKS = [
    ('{"relation":"right","frame":"user_heading","domain":"objects","states":"TFU","conditional":false}',
     "objects", {("A",): ("T", False), ("B",): ("F", False), ("C",): ("U", False)}),
    ('{"relation":"right","frame":"user_to_anchor","domain":"ordered_pairs","states":["-TU","F-T","UF-"],'
     '"conditional":["-10","0-0","10-"]}',
     "ordered_pairs", {("A", "B"): ("T", True), ("A", "C"): ("U", False), ("B", "A"): ("F", False),
                       ("B", "C"): ("T", False), ("C", "A"): ("U", True), ("C", "B"): ("F", False)}),
    ('{"relation":"near","frame":null,"domain":"unordered_pairs","states":["TU","F"],"conditional":["01","0"]}',
     "unordered_pairs", {("A", "B"): ("T", False), ("A", "C"): ("U", True), ("B", "C"): ("F", False)}),
    ('{"relation":"between","frame":null,"domain":"target_anchor_pairs","states":[[0,1,"--T"],[0,2,"-U-"],'
     '[1,2,"F--"]],"conditional":[[0,1,"--0"],[0,2,"-1-"],[1,2,"0--"]]}',
     "target_anchor_pairs", {("C", "A", "B"): ("T", False), ("B", "A", "C"): ("U", True),
                             ("A", "B", "C"): ("F", False)}),
]
# Section 4.5: distances A/B = 0.1006, A/C = 0.1508 (not 0.101 and 0.151), B/C unknown (null, not 0)
LITERAL_DISTANCES = ('{"measure":"center_distance_m","domain":"unordered_pairs","values":[[0.1006,0.1508],[null]],'
                     '"conditional":false}',
                     {("A", "B"): (0.1006, False), ("A", "C"): (0.1508, False), ("B", "C"): (None, False)})

# Section 8.5: reverse retrieval. near and distances are symmetric; between's anchors are unordered; ordered
# relations are not symmetrized: right(A,B) = T but right(B,A) = F in the literal ordered example.
REVERSE = {"near": {("B", "A"): "T", ("C", "A"): "U", ("C", "B"): "F"},
           "between": {("C", "B", "A"): "T", ("B", "C", "A"): "U", ("A", "C", "B"): "F"},
           "ordered": {("A", "B"): "T", ("B", "A"): "F"}}

# Section 8.11: indices map through O, not through ID suffixes.
O_SUFFIX = ["obj_001", "obj_1000", "obj_999"]
SUFFIX_BLOCK = '{"relation":"right","frame":"user_heading","domain":"objects","states":"TFU","conditional":false}'
SUFFIX_EXPECTED = {"obj_001": "T", "obj_1000": "F", "obj_999": "U"}  # index 0, 1, 2 of O
# Wrong lookups the audit must show to be wrong: by numeric suffix (index 999 or 998 does not exist) and by sorting
# the suffixes numerically (that order puts obj_999 at index 1, which would read F for obj_999).
SUFFIX_WRONG = {"suffix_minus_one": {"obj_1000": 999, "obj_999": 998}, "numeric_order_index_1": "obj_999"}

# Section 8.7: target/anchor orientation in an audited input, worked by hand from a17's records before decoding.
# User at U = (0.4, 1.0), table obj_001 at (2.6, 1.4), chair obj_004 at (2.55, 2.18), chair obj_005 at (2.55, 0.62).
# right(obj_005, obj_001): w = A - U = (2.2, 0.4), |w| = sqrt(5) = 2.2360680, v = (0.9838699, 0.1788854),
#   right axis = (v_y, -v_x) = (0.1788854, -0.9838699); d = T - A = (-0.05, -0.78);
#   score = -0.05 * 0.1788854 + 0.78 * 0.9838699 = 0.7584742 > 0.02 -> T
# right(obj_004, obj_001): d = (-0.05, 0.78); score = -0.0089443 - 0.7674185 = -0.7763628 < -0.02 -> F
# right(obj_001, obj_005): anchor (2.55, 0.62), w = (2.15, -0.38), |w| = 2.1833231, v = (0.9847419, -0.1740473),
#   right axis = (-0.1740473, -0.9847419); d = (0.05, 0.78); score = -0.0087024 - 0.7680987 = -0.7768011 -> F
# So in the ordered layout row obj_005 / column obj_001 is T, and the transposed cell is F.
A17_ORIENTATION = {"input": "a17.annotated.coordinates_relations_v1", "relation": "right", "frame": "user_to_anchor",
                   "cells": {("obj_005", "obj_001"): "T", ("obj_004", "obj_001"): "F", ("obj_001", "obj_005"): "F"}}
