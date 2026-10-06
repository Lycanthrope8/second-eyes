"""A2.3b (D82): the saved A2.3a pilot choices scored against the source annotations, with a matched rules baseline.

    python -m grounding.evaluation.iref_vla_pilot baseline --requests DIR --bundle DIR --relation-config FILE
        --direction-config FILE --out NEW_DIR
    python -m grounding.evaluation.iref_vla_pilot score --pilot DIR --requests DIR --bundle DIR --rules DIR
        --annotations FILE --out NEW_DIR

`baseline` runs the accepted parser and resolver once per selected parent and inventory view and reads no annotation
or model output; `score` reads saved outputs and the annotations and never parses, resolves, serializes or runs a
model. See docs/iref-vla-pilot-scoring.md.
"""
from .baseline import load_rules_dir, run_baseline, summarize_rules, verify_rules_dir
from .inputs import load_annotations, score_problems
from .policy import POLICY_ID, POLICY_PATH, ScoringInternalError, load_policy
from .score import build_rows, render_report, run_score, summarize, verify_score_dir

__all__ = ["POLICY_ID", "POLICY_PATH", "ScoringInternalError", "build_rows", "load_annotations", "load_policy",
           "load_rules_dir", "render_report", "run_baseline", "run_score", "score_problems", "summarize",
           "summarize_rules", "verify_rules_dir", "verify_score_dir"]
