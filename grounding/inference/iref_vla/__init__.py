"""A2.3a (D81): a bounded, answer-blind zero-shot direct-selection pilot of Qwen2.5-0.5B-Instruct.

    python -m grounding.inference.iref_vla prepare --bundle DIR --tokenizer-dir DIR --model-description FILE --out NEW
    python -m grounding.inference.iref_vla run --requests DIR --model-dir DIR --device cpu|cuda --tokenizer-dir DIR --out NEW

See docs/iref-vla-zero-shot-pilot.md. PyTorch is imported only when a model is loaded.
"""
from .choices import (NonFiniteOutput, build_prompt, ceiling_results, check_boundary, choice_mapping, choices_line,
                      context_status, last_position, restricted_shares, score, select_parents)
from .model import TorchModel, checkpoint_evidence
from .prepare import prepare_requests, verify_request_dir
from .protocol import PROTOCOL_PATH, load_protocol, protocol_sha256
from .run import PilotFailure, run_pilot, verify_results

__all__ = ["NonFiniteOutput", "PROTOCOL_PATH", "PilotFailure", "TorchModel", "build_prompt", "ceiling_results",
           "check_boundary", "checkpoint_evidence", "choice_mapping", "choices_line", "context_status", "last_position",
           "load_protocol", "prepare_requests", "protocol_sha256", "restricted_shares", "run_pilot", "score",
           "select_parents", "verify_request_dir", "verify_results"]
