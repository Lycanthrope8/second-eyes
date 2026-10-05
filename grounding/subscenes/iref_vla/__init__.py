"""The A2.2c category-complete selection audit (D77) on the pinned IRef-VLA development sample.

    python -m grounding.subscenes.iref_vla --scene SCENE --commands COMMAND_DIR --category-map MAP
        --inventory-views VIEWS --out NEW_DIR

For each parsed command, every object of each inventory view whose category the accepted parse names, plus every
unknown-category object. Selection plans over the unchanged scene: no scene, prompt or training file is written. See
docs/iref-vla-subscenes.md.
"""
from .audit import Audit, audit, parent_scene_sha256, render_report, run_audit, selection_id, summarize
from .selector import POLICY_ID, PRIMARY_OBJECT_LIMIT, SENSITIVITY_LIMITS, ProjectedObject, primary_status, select

__all__ = ["Audit", "POLICY_ID", "PRIMARY_OBJECT_LIMIT", "ProjectedObject", "SENSITIVITY_LIMITS", "audit",
           "parent_scene_sha256", "primary_status", "render_report", "run_audit", "select", "selection_id", "summarize"]
