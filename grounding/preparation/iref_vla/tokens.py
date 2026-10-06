"""A2.2d pinned tokenizer, sizing wrapper and token arithmetic (D78).

The tokenizer is Qwen/Qwen2.5-0.5B-Instruct at revision 7ae557604adf67be50417f59c2c2f167def9a775, loaded from exactly the
three verified local files, staged alone in a temporary folder so no other local file can change how it loads, under
transformers 4.57.6 and tokenizers 0.22.2. Counts are exact for that tokenizer and this sizing wrapper; parity with the
deployed GGUF model is not established. Test doubles (kind "test_double") may replace it in fixture tests only.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import tempfile
from pathlib import Path

from ... import scene as a1_scene
from ...evaluation.iref_vla.protocol import EvaluationInputError, issue

REPO = Path(__file__).resolve().parents[3]
MODEL_DESCRIPTION = REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct.json"
A1_PROMPTS = REPO / "grounding" / "prompts"
GOLDENS = REPO / "grounding" / "tests" / "fixtures" / "serialization" / "golden"

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PINNED_FILES = {"vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
                "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
                "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"}
VERSIONS = {"transformers": "4.57.6", "tokenizers": "0.22.2"}
TOKENIZER_CLASS = "Qwen2TokenizerFast"
FORMATS = ("coordinates_v2", "coordinates_relations_v2")
CEILINGS = (1024, 2048, 4096)
SYSTEM_MESSAGE = ("Use the supplied scene evidence to interpret the typed command. Unknown values are not false facts. "
                  "Respect the stated spatial definitions and frame labels. Do not invent missing evidence or silently "
                  "choose an unstated reference frame.")
WRAPPER_TEMPLATE = ("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{document}<|im_end|>\n"
                    "<|im_start|>assistant\n")
HEAD = "<|im_start|>system\n" + SYSTEM_MESSAGE + "<|im_end|>\n<|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n"
MEASUREMENT_LABEL = ("exact for the pinned Hugging Face tokenizer and this sizing wrapper; deployed-GGUF execution parity "
                     "not established by this increment")
A1_EXPECTED = {"a17-fixed": 230, "a17-front": 232, "a17-left": 230, "a17-table": 227}
A1_SCENE_EXPECTED = 191
GOLDEN_EXPECTED = {"a17.annotated": [1343, 2266], "a17.restricted": [1272, 2106],
                   "dirh10.annotated": [1443, 2998], "dirh10.restricted": [1413, 3625]}
A1_RAW_REFERENCES = "not supplied: token-ID parity with A1's recorded token IDs is not claimed"


def _h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def wrap(document: str) -> str:
    """The exact sizing wrapper; the document's own final LF stays immediately before <|im_end|>."""
    return HEAD + document + TAIL


def wrapper_identity() -> str:
    return f"sizing_wrapper@{_h(WRAPPER_TEMPLATE)[:12]}+system@{_h(SYSTEM_MESSAGE)[:12]}"


def ceiling_results(count: int) -> dict:
    """Each declared input-only ceiling; equality passes."""
    return {str(c): "within_input_ceiling" if count <= c else "exceeds_input_ceiling" for c in CEILINGS}


def common_prefix_length(a, b) -> int:
    """Length of the longest common token-ID prefix: compares IDs, never subtracts separately counted lengths."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def prefix_reuse(pairs) -> dict:
    """pairs: (prefix IDs, complete-input IDs) for one (selection, format) group; the group keeps its minimum."""
    compatible = [common_prefix_length(p, i) for p, i in pairs]
    return {"compatible": compatible, "group_reusable_prefix_tokens": min(compatible) if compatible else 0}


def check_model_description(desc) -> None:
    """The supplied description names the pinned model and revision, and its template equals the literal wrapper."""
    problems = []
    if not isinstance(desc, dict):
        problems.append(issue("model description", "E_PREP_MODEL", "the model description must be a JSON object"))
    else:
        if desc.get("hf_repo") != MODEL_REPO:
            problems.append(issue("model description", "E_PREP_MODEL", f"hf_repo must be {MODEL_REPO}"))
        if desc.get("hf_revision") != REVISION:
            problems.append(issue("model description", "E_PREP_MODEL", f"hf_revision must be {REVISION}"))
        t, probe = desc.get("chat_template"), "{\"probe\":1}\n"
        if not isinstance(t, str) or a1_scene.formatted(t, SYSTEM_MESSAGE, probe) != wrap(probe):
            problems.append(issue("model description", "E_PREP_MODEL", "chat_template, through grounding.scene.formatted, "
                                                                        "must equal the literal sizing wrapper"))
    if problems:
        raise EvaluationInputError(problems)


class PinnedTokenizer:
    kind = "exact"

    def __init__(self, backend, versions: dict, class_name: str):
        self._backend, self.versions, self.class_name = backend, versions, class_name
        self.identity = f"{MODEL_REPO}@{REVISION}"

    def encode(self, text: str) -> list:
        """One tokenizer call over the complete text; special tokens are already in the text, none are added."""
        return list(self._backend(text, add_special_tokens=False)["input_ids"])


def load_pinned_tokenizer(directory) -> PinnedTokenizer:
    d, problems, data = Path(directory), [], {}
    for name, want in PINNED_FILES.items():
        try:
            data[name] = (d / name).read_bytes()
        except OSError as e:
            problems.append(issue(str(d / name), "E_PREP_TOKENIZER", f"cannot read a pinned tokenizer file: {e}"))
            continue
        got = hashlib.sha256(data[name]).hexdigest()
        if got != want:
            problems.append(issue(str(d / name), "E_PREP_TOKENIZER", f"SHA-256 {got} is not the pinned {want}"))
    if problems:
        raise EvaluationInputError(problems)
    try:
        transformers, tokenizers = importlib.import_module("transformers"), importlib.import_module("tokenizers")
    except ImportError as e:
        raise EvaluationInputError([issue("environment", "E_PREP_TOKENIZER", f"the pinned measurement environment is not "
                                          f"installed ({e}); see grounding/requirements-token-audit.txt")]) from None
    got = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
    if got != VERSIONS:
        raise EvaluationInputError([issue("environment", "E_PREP_TOKENIZER", f"found {got}; the measurement environment "
                                                                             f"is pinned to {VERSIONS}")])
    stage = Path(tempfile.mkdtemp(prefix="iref-pinned-tokenizer-"))
    try:
        for name, b in data.items():
            (stage / name).write_bytes(b)
        backend = transformers.AutoTokenizer.from_pretrained(str(stage), local_files_only=True, trust_remote_code=False,
                                                             use_fast=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    if type(backend).__name__ != TOKENIZER_CLASS:
        raise EvaluationInputError([issue("tokenizer", "E_PREP_TOKENIZER", f"loaded {type(backend).__name__}; "
                                                                           f"{TOKENIZER_CLASS} is required")])
    return PinnedTokenizer(backend, got, type(backend).__name__)


def calibrate(tok, desc=None) -> dict:
    """A1's four tracked prompts through A1's own formatter and boundary helper, and the eight stored goldens wrapped."""
    desc = desc if desc is not None else json.loads(MODEL_DESCRIPTION.read_text(encoding="utf-8"))
    a1, prefix = {}, {}
    for pid in A1_EXPECTED:
        p = json.loads((A1_PROMPTS / f"{pid}.json").read_text(encoding="utf-8"))
        text = a1_scene.formatted(desc["chat_template"], p["system"], p["user"])
        a1[pid] = len(tok.encode(text))
        prefix[pid] = len(tok.encode(text[:a1_scene.scene_chars(text)]))
    goldens = {name: [len(tok.encode(wrap((GOLDENS / f"{name}.{fmt}.jsonl").read_bytes().decode("utf-8"))))
                      for fmt in FORMATS] for name in GOLDEN_EXPECTED}
    return {"a1_prompts": a1, "a1_scene_prefix": prefix, "goldens": goldens, "a1_raw_references": A1_RAW_REFERENCES}


def check_calibration(cal: dict) -> None:
    bad = []
    if cal["a1_prompts"] != A1_EXPECTED:
        bad.append(f"A1 prompts counted {cal['a1_prompts']}, expected {A1_EXPECTED}")
    if set(cal["a1_scene_prefix"].values()) != {A1_SCENE_EXPECTED}:
        bad.append(f"A1 scene prefixes counted {cal['a1_scene_prefix']}, expected {A1_SCENE_EXPECTED}")
    if cal["goldens"] != GOLDEN_EXPECTED:
        bad.append(f"goldens counted {cal['goldens']}, expected {GOLDEN_EXPECTED}")
    if bad:
        raise EvaluationInputError([issue("calibration", "E_PREP_CALIBRATION", m) for m in bad])
