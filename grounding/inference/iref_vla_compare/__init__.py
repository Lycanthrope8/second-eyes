"""A2.3d (D95): the zero-shot comparison of rules, Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct on 256 more parents.

`prepare` (laptop) freezes the selection, one answer-blind letter assignment and list order per parent and view, the
prompts and both tokenizers' token IDs; `rules` (laptop) runs the unchanged parser and resolver on exactly the same
subscenes. Model runs and scoring are later deliveries. Commands load lazily.
"""
import importlib

_LAZY = {"prepare_compare": "prepare", "verify_compare_requests": "prepare", "run_compare_rules": "rules",
         "verify_compare_rules": "rules", "smoke": "run", "run_compare": "run", "verify_compare_results": "run", "score_compare": "score",
         "verify_compare_scores": "score"}
__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
