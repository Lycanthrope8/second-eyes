"""A2.4's fine-tuned model identity (D109): a separate path beside the accepted base-model check, which is unchanged.

A merged checkpoint carries `a24-identity.json`, written by `train.merge`. It binds the model's name (for example
`a24.direct.coordinates_v2.s1`) to:

- the base: repository, pinned revision, and the accepted `checkpoint_evidence` record of the base folder it was
  trained from (its tie and digest);
- the tokenizer: the pinned files, byte for byte;
- the training: the training specification, settings, preparation manifest and data-file hashes, the fit's lock (seed,
  configuration, code and environment), and the selected checkpoint with its rule and metric;
- the adapter: its configuration and every weight file;
- the merge: procedure, software versions and the adapter-versus-merged check;
- every file of the merged folder, by SHA-256 and size.

`finetuned_evidence` is the accepted runner's `evidence_fn` for these folders. It re-hashes every file, refuses an
unlisted or missing file, checks the tokenizer and architecture as `checkpoint_evidence` does, and names the model in
its tie. A locally computed hash identifies a local artifact; it is never presented as a published checkpoint hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..evaluation.iref_vla.protocol import EvaluationInputError, issue

IDENTITY_FILE = "a24-identity.json"
RECORD_TYPE = "a24_finetuned_identity"
IGNORED = {IDENTITY_FILE}


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def folder_files(folder) -> dict:
    """{name: {sha256, bytes}} for every file directly in the folder except the identity record."""
    d = Path(folder)
    return {p.name: {"sha256": _sha_file(p), "bytes": p.stat().st_size}
            for p in sorted(d.iterdir()) if p.is_file() and p.name not in IGNORED}


def _fail(where, message):
    raise EvaluationInputError([issue(str(where), "E_A24_IDENTITY", message)])


def write_identity(model_dir, record: dict) -> dict:
    """Completes and writes the record: the merged folder's files are hashed now, after every file is in place."""
    d = Path(model_dir)
    if (d / IDENTITY_FILE).exists():
        _fail(d / IDENTITY_FILE, "an identity record exists; a merged folder is written once")
    rec = dict(record, format_version=1, record_type=RECORD_TYPE, merged_files=folder_files(d),
               status="a local fine-tuned artifact identity (A2.4); not a published checkpoint")
    (d / IDENTITY_FILE).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    load_identity(d)
    return rec


def load_identity(model_dir, protocol=None) -> dict:
    """The verified record: every listed file present with its hash and size, no other file, the pinned tokenizer files
    byte-identical when the protocol is given, the base revision pinned. Returns the record with its own hash."""
    d = Path(model_dir)
    p = d / IDENTITY_FILE
    if not p.is_file():
        _fail(d, f"no {IDENTITY_FILE}: not an A2.4 fine-tuned folder")
    raw = p.read_bytes()
    rec = json.loads(raw)
    if rec.get("record_type") != RECORD_TYPE or not rec.get("model_name", "").startswith("a24."):
        _fail(p, "not an A2.4 fine-tuned identity record")
    have = folder_files(d)
    if set(have) != set(rec["merged_files"]):
        _fail(d, f"the folder's files differ from the record: extra {sorted(set(have) - set(rec['merged_files']))}, "
                 f"missing {sorted(set(rec['merged_files']) - set(have))}")
    changed = [n for n, v in have.items() if v != rec["merged_files"][n]]
    if changed:
        _fail(d, f"files changed since the merge: {changed}")
    if protocol is not None:
        if rec["base"]["hf_revision"] != protocol["model"]["hf_revision"]:
            _fail(p, "the record's base revision is not the pinned one")
        for name, want in protocol["tokenizer"]["files"].items():
            if name in have and have[name]["sha256"] != want:
                _fail(d / name, "differs from the pinned tokenizer file")
            if name not in have:
                _fail(d, f"the pinned tokenizer file {name} is missing")
    return dict(rec, identity_sha256=hashlib.sha256(raw).hexdigest())


def finetuned_evidence(model_dir, protocol, desc=None, expected_weights_sha256=None) -> dict:
    """The accepted runner's evidence_fn for A2.4 folders (same signature as checkpoint_evidence)."""
    if expected_weights_sha256:
        _fail(model_dir, "a published-hash tie does not apply to a local fine-tuned artifact")
    rec = load_identity(model_dir, protocol)
    d = Path(model_dir)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    arch = (desc or {}).get("architecture", {})
    if arch:
        head_dim = cfg.get("hidden_size", 0) // max(cfg.get("num_attention_heads", 1), 1)
        if (cfg.get("num_hidden_layers"), cfg.get("num_key_value_heads"), head_dim) != (
                arch.get("max_layers"), arch.get("num_key_value_heads"), arch.get("head_dim")):
            _fail(d / "config.json", "layers, key-value heads or head size differ from the model description")
    if cfg.get("max_position_embeddings", 0) < protocol["context_limit_tokens"]:
        _fail(d / "config.json", "fewer positions than the context limit")
    w = [n for n in rec["merged_files"] if n.endswith(".safetensors")]
    tie = (f"A2.4 fine-tuned local artifact {rec['model_name']}: base {rec['base']['hf_repo']}@"
           f"{rec['base']['hf_revision'][:7]} ({rec['base']['evidence_tie']}); selected {rec['selection']['checkpoint']}; "
           f"merged weights {', '.join(rec['merged_files'][n]['sha256'][:12] for n in w)}; identity "
           f"{rec['identity_sha256'][:12]}; a local identity, not a published checkpoint")
    return {"revision": rec["base"]["hf_revision"], "tie": tie,
            "files": {n: {"sha256": v["sha256"], "bytes": v["bytes"], "metadata": None} for n, v in rec["merged_files"].items()},
            "max_position_embeddings": cfg.get("max_position_embeddings"), "a24_model": rec["model_name"],
            "a24_identity_sha256": rec["identity_sha256"]}
