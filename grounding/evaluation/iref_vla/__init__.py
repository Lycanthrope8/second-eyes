"""The A2.2b text-only rules baseline on the pinned IRef-VLA development sample (D75).

    python -m grounding.evaluation.iref_vla predict --scene SCENE --commands COMMAND_DIR --category-map MAP
        --inventory-views VIEWS --relation-config REL --direction-config DIR --out NEW_DIR
    python -m grounding.evaluation.iref_vla score --predictions PREDICTION_DIR --annotations ANNOTATIONS --out NEW_DIR

Three separate operations: parse (text and vocabularies only), predict (no answers) and score (completed predictions
against the separate annotations). See docs/iref-vla-evaluation.md.
"""
from .parse import parse, parse_record
from .predict import Prediction, finalize, load_prediction, predict, publish_prediction, run_predict, summarize
from .protocol import (PROTOCOL_PATH, EvaluationInputError, EvaluationOutputError, load_protocol, semantic_hash,
                       validate_parse_record)
from .score import ScoreResult, run_score, score

__all__ = ["PROTOCOL_PATH", "EvaluationInputError", "EvaluationOutputError", "Prediction", "ScoreResult", "finalize",
           "load_prediction", "load_protocol", "parse", "parse_record", "predict", "publish_prediction", "run_predict",
           "run_score", "score", "semantic_hash", "summarize", "validate_parse_record"]
