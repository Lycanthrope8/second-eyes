"""A2.3c (D94): crossed answer-code assignment and choices-list order diagnostic.

Every base request of the A2.3a pilot is re-presented in an n x n crossed cyclic grid: the answer letters assigned
to its objects rotate by a, and the order of the choices list rotates by p, while the scene document stays byte
identical. `prepare` freezes the 4,536 prompts on the laptop, `run` executes them on the RTX PC after canaries and
128 repeat controls, and `score` compares the choices with the accepted A2.3b source targets, on the laptop.

The commands are loaded lazily, so the RTX PC's `run` never imports the A2.3b scoring modules.
"""
import importlib

from .design import cells, load_policy, policy_sha256, transform, variant_id

_LAZY = {"prepare_ordering": "prepare", "verify_order_requests": "prepare", "run_ordering": "run",
         "verify_order_results": "run", "score_ordering": "score", "verify_order_scores": "score"}

__all__ = ["cells", "load_policy", "policy_sha256", "transform", "variant_id", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
