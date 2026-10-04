"""The offline structured resolver (A2.1e, D71): grounding queries to scene-object IDs, deterministically.

    from grounding.resolution import resolve
    record = resolve(scene, command, query, resolver_config=config, relation_config_path=...,
                     direction_config_path=..., category_maps=[...])

The returned resolution_run record follows schemas/grounding-result.v1.json. See docs/resolution.md. The resolver
interprets no English and no model output; it executes supplied readings with the accepted relation libraries.
"""
from .resolve import EVALUATOR_VERSION, resolve

__all__ = ["EVALUATOR_VERSION", "resolve"]
