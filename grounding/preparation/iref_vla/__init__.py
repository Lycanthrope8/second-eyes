"""A2.2d (D78): materialize A2.2c's fitting selections and measure both accepted input formats with the pinned tokenizer.

    python -m grounding.preparation.iref_vla --scene ... --commands ... --category-map ... --inventory-views ...
        --selection-audit DIR --relation-config ... --direction-config ... --tokenizer-dir ... --model-description ... --out NEW

See docs/iref-vla-model-inputs.md. No model is run; rendered texts are sizing diagnostics, not inference requests.
"""
from .prepare import (OBJECT_LIMIT, POLICY, WORK_CAP, derived_command_id, line_names, materialize_command,
                      materialize_scene, measure, render_report, run_preparation, summarize, verify_bundle)
from .tokens import (CEILINGS, FORMATS, SYSTEM_MESSAGE, calibrate, ceiling_results, check_calibration,
                     check_model_description, common_prefix_length, load_pinned_tokenizer, prefix_reuse, wrap)

__all__ = ["CEILINGS", "FORMATS", "OBJECT_LIMIT", "POLICY", "SYSTEM_MESSAGE", "WORK_CAP", "calibrate", "ceiling_results",
           "check_calibration", "check_model_description", "common_prefix_length", "derived_command_id", "line_names",
           "load_pinned_tokenizer", "materialize_command", "materialize_scene", "measure", "prefix_reuse", "render_report",
           "run_preparation", "summarize", "verify_bundle", "wrap"]
