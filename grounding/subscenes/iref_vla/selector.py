"""The A2.2c core selector (D77): category-complete membership over a validated projection.

select() sees only object IDs with their category state, a model label or None for an unknown category, and the
required category set. It never receives a scene, a command's text, a relation, a colour or an answer: validation,
parsing and provenance belong to audit.py. It performs no geometry, no subset enumeration and no inference.
"""
from __future__ import annotations

from dataclasses import dataclass

POLICY_ID = "iref_category_complete.v1"
PRIMARY_OBJECT_LIMIT = 10
SENSITIVITY_LIMITS = tuple(range(3, 11))


@dataclass(frozen=True)
class ProjectedObject:
    """One object as the selector may see it: its ID, and its model category or None when the category is unknown."""
    object_id: str
    model_category: object


def select(objects, categories) -> tuple:
    """Every object whose category is unknown or among `categories`, as IDs in lexicographic string order."""
    if not isinstance(objects, tuple) or not all(isinstance(o, ProjectedObject) for o in objects):
        raise TypeError("objects must be a tuple of ProjectedObject")
    if not isinstance(categories, frozenset) or not all(isinstance(c, str) for c in categories):
        raise TypeError("categories must be a frozenset of model labels")
    if not all(isinstance(o.object_id, str) and (o.model_category is None or isinstance(o.model_category, str))
               for o in objects):
        raise TypeError("each object needs a string ID and a string model category or None")
    ids = [o.object_id for o in objects]
    if len(set(ids)) != len(ids):
        raise ValueError("an object ID is repeated in the projection")
    return tuple(sorted(o.object_id for o in objects if o.model_category is None or o.model_category in categories))


def primary_status(count) -> str:
    """fits at or under the provisional ten-object screening budget, otherwise over_budget; never truncated."""
    if type(count) is not int or count < 0:
        raise TypeError("count must be a nonnegative plain integer")
    return "fits" if count <= PRIMARY_OBJECT_LIMIT else "over_budget"
