"""A2.5 delivery 2 (D104): the PC's golden references for the headset's prompt builder.

**`build_goldens`**, command `golden-build`:

- reads the accepted A2.2d bundle (`a22d-20261005-153413`) after its own readback, with its index pinned by hash;
- chooses one annotated snapshot of each size (3, 6 and 10 objects) by a rule fixed in advance: among the materialized
  selections of that size, the lowest SHA-256 of `salt + LF + selection ID`;
- takes every derived command bound to each snapshot (dataset commands), plus two written fixtures per snapshot. The
  fixtures are labelled apart and serve as implementation checks only: the first dataset command with new text that
  exercises quotes, non-ASCII text and a tab;
- builds, with the accepted Python code:
  - the `coordinates_v2` document (serializer, A2.2d's configs and category map);
  - the serialized-order mapping (A2.3a's `choice_mapping`, D104(1)) and its hash;
  - the prompt (`build_prompt`) and its token IDs (the pinned tokenizer);
  - the two cache boundaries, before the last token and before the command line.

**The headset part** (`headset/`) holds:

- `prompt-asset.json`: the header and semantics lines, the wrapper strings, system message, codes and code token IDs,
  the configs' and serializer's hashes;
- the snapshot and command records, byte for byte as in the bundle;
- `goldens.jsonl`, one golden per request;
- `golden-manifest.json`, pinning every headset file by SHA-256.

The root `manifest.json` adds the provenance: the bundle, its index hash, the selection with its candidates, the
tokenizer's identity and the code that ran.

**`verify_goldens`** (`golden-verify`) rebuilds every document, mapping and prompt from the headset files and checks
every hash, and the tokenization too when a tokenizer is given.

**`golden_references`** (`golden-references`) writes fresh float32 references for every golden into a new folder:
offered logits, log-probabilities, shares, the choice (exact ties to K) and the margin. It uses A2.3a's `TorchModel`
(float32, eager attention, pinned checkpoint), on CPU or CUDA, with the device recorded.
"""
from __future__ import annotations

import base64
import copy
import datetime
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

from ..evaluation.iref_vla import output
from ..evaluation.iref_vla.protocol import EvaluationInputError, EvaluationOutputError, encode_json, encode_jsonl, issue, runtime, sha256
from ..inference.iref_vla.choices import build_prompt, choice_mapping
from ..inference.iref_vla.protocol import load_protocol
from ..preparation.iref_vla import prepare as P
from ..preparation.iref_vla import tokens as T
from ..serialization.serializer import serialize
from . import replay_inputs as RI
from .publish import publish
from .replay_bundle import boundary_before_command_line, boundary_before_last_token

REPO = Path(__file__).resolve().parents[2]
POLICY_ID = "a25.d2.goldens.v1"
SALT = "a25.d2.snapshots.v1"
SIZES = (3, 6, 10)
BUNDLE_NAME = "a22d-20261005-153413"
INDEX_SHA256_PREFIX = "44a994b4"          # A2.2d's recorded index hash (notes/phases/A2.2d_model_inputs.md)
FORMAT = "coordinates_v2"
REL = REPO / "grounding" / "relations" / "relations.v1.json"
DIR = REPO / "grounding" / "relations" / "directions.v1.json"
DEFAULT_HF = REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct" / "hf"
WRITTEN = (("w1", 'Hand me the "left" one: the café mug 😀'), ("w2", "which one?\tthe second, by the door"))
SERIALIZER_FILES = ("grounding/serialization/codec.py", "grounding/serialization/serializer.py",
                    "grounding/serialization/constants.py", "grounding/inference/iref_vla/choices.py")


def _fail(where, message):
    raise EvaluationInputError([issue(str(where), "E_GOLDENS", message)])


def select_snapshots(rows, sizes=SIZES, salt=SALT) -> list:
    """One materialized selection per size: the lowest SHA-256(salt + LF + selection ID) among those of that size."""
    count = {}
    for r in rows:
        if r.get("preparation_status") == "materialized":
            n = len(set(r["planned_object_ids"]))
            if count.setdefault(r["derived_scene_id"], n) != n:
                raise ValueError(f"selection {r['derived_scene_id']} has rows of different object counts")
    chosen = []
    for size in sizes:
        cands = sorted((hashlib.sha256(f"{salt}\n{sel}".encode("utf-8")).hexdigest(), sel) for sel, n in count.items() if n == size)
        if not cands:
            raise ValueError(f"no materialized selection has {size} objects")
        chosen.append({"objects": size, "selection_id": cands[0][1], "rank_sha256": cands[0][0], "candidates": len(cands)})
    return chosen


def _asset(proto, header, semantics) -> dict:
    w = proto["wrapper"]
    return {"format_version": 1, "record_type": "a25_prompt_asset", "policy_id": POLICY_ID, "format": FORMAT,
            "header_line": header, "semantics_line": semantics,
            "wrapper": {"before_system": w["before_system"], "between": w["between"], "after_user": w["after_user"]},
            "system_message": proto["system_message"], "object_codes": proto["object_codes"], "ask_code": proto["ask_code"],
            "ask_target": proto["ask_target"], "code_token_ids": proto["code_token_ids"],
            "configs": {"grounding/relations/relations.v1.json": sha256(REL.read_bytes()),
                        "grounding/relations/directions.v1.json": sha256(DIR.read_bytes())},
            "serializer": {f: sha256((REPO / f).read_bytes()) for f in SERIALIZER_FILES}}


def _golden(k, kind, sel, n, cid, scene_bytes, command_bytes, scene, command, cmap, proto, tok):
    res = serialize(scene, command, format=FORMAT, relation_config_path=REL, direction_config_path=DIR, category_maps=[cmap])
    if res.status != "ok":
        raise ValueError(f"{sel}/{cid}: the serializer did not render the request ({res.status})")
    doc = res.document
    ids = [o["object_id"] for o in sorted(scene["objects"], key=lambda o: o["object_id"])]
    mapping = choice_mapping(ids, proto)
    codes, targets = [m[0] for m in mapping], [m[1] for m in mapping]
    code_ids = [proto["code_token_ids"][c] for c in codes]
    prompt = build_prompt(proto, mapping, doc)
    token_ids = [int(x) for x in tok.encode(prompt)]
    if len(token_ids) + proto["continuation_tokens"] > proto["context_limit_tokens"]:
        raise ValueError(f"{sel}/{cid}: {len(token_ids)} tokens leave no room within the context limit")
    last = boundary_before_last_token(tok, prompt, token_ids)
    cmd = boundary_before_command_line(tok, prompt, token_ids, proto, mapping)
    pb = prompt.encode("utf-8")
    rec = {"format_version": 1, "record_type": "a25_prompt_golden", "request_id": f"g{k:02d}", "kind": kind,
           "snapshot_id": sel, "objects": n, "command_id": cid, "scene_file": f"snapshots/{sel}.json",
           "command_file": f"commands/{cid}.json", "scene_sha256": sha256(scene_bytes), "command_sha256": sha256(command_bytes),
           "document": doc, "document_sha256": sha256(doc.encode("utf-8")), "prompt_sha256": sha256(pb), "prompt_bytes": len(pb),
           "token_ids": token_ids, "input_tokens": len(token_ids), "token_ids_sha256": RI.token_ids_sha256(token_ids),
           "codes": codes, "targets": targets, "code_token_ids": code_ids,
           "mapping_sha256": RI.mapping_sha256(codes, targets, code_ids),
           "keep_before_last_token": last["keep_tokens"], "keep_before_command_line": cmd["keep_tokens"]}
    return rec, doc.splitlines(keepends=True)[:2]


def build_goldens(*, bundle, out, tokenizer_dir=None, tokenizer=None, protocol=None, sizes=SIZES,
                  index_sha256_prefix=INDEX_SHA256_PREFIX) -> dict:
    out = output.refuse_existing(out)
    b = Path(bundle)
    bad = P.verify_bundle(b)
    if bad:
        _fail(b, "the A2.2d bundle does not read back: " + "; ".join(bad[:3]))
    index_bytes = (b / "preparation-index.jsonl").read_bytes()
    if index_sha256_prefix and not sha256(index_bytes).startswith(index_sha256_prefix):
        _fail(b, f"the bundle's index is not the accepted one ({index_sha256_prefix}...)")
    rows = [json.loads(x) for x in index_bytes.decode("utf-8").splitlines() if x.strip()]
    cmap = json.loads((b / "model_records" / "category-map.json").read_text(encoding="utf-8"))
    proto = protocol if protocol is not None else load_protocol()
    tok = tokenizer if tokenizer is not None else T.load_pinned_tokenizer(tokenizer_dir or DEFAULT_HF)
    try:
        chosen = select_snapshots(rows, sizes)
    except ValueError as e:
        _fail(b, str(e))
    goldens, files, statics, k = [], {}, set(), 0
    for ch in chosen:
        sel = ch["selection_id"]
        scene_bytes = (b / "model_records" / "scenes" / f"{sel}.json").read_bytes()
        scene = json.loads(scene_bytes)
        mine = [r for r in rows if r.get("derived_scene_id") == sel and r.get("preparation_status") == "materialized"]
        if P.canonical_sha256(scene) != mine[0]["scene_sha256"] or len(scene["objects"]) != ch["objects"]:
            _fail(b, f"snapshot {sel} differs from its index row")
        files[f"snapshots/{sel}.json"] = scene_bytes
        cmds = []
        for r in sorted(mine, key=lambda r: r["derived_command_id"]):
            cb = (b / "model_records" / "commands" / f"{r['derived_command_id']}.json").read_bytes()
            c = json.loads(cb)
            if P.canonical_sha256(c) != r["command_sha256"]:
                _fail(b, f"command {r['derived_command_id']} differs from its index row")
            cmds.append(("dataset_command", c["command_id"], cb, c))
        for suffix, text in WRITTEN:
            c = copy.deepcopy(cmds[0][3])
            c["command_id"] = f"{c['command_id']}-{suffix}"
            c["text"] = text
            cmds.append(("written_fixture", c["command_id"], encode_json(c), c))
        for kind, cid, cb, c in cmds:
            files[f"commands/{cid}.json"] = cb
            k += 1
            try:
                rec, two = _golden(k, kind, sel, ch["objects"], cid, scene_bytes, cb, scene, c, cmap, proto, tok)
            except Exception as e:  # noqa: BLE001 - any refusal of the accepted code is reported as such
                _fail(b, f"{sel}/{cid}: {type(e).__name__}: {e}")
            statics.add(tuple(two))
            goldens.append(rec)
    if len(statics) != 1:
        _fail(b, "the header and semantics lines are not the same for every request")
    header, semantics = statics.pop()
    asset = encode_json(_asset(proto, header, semantics))
    head = {"prompt-asset.json": asset, "goldens.jsonl": encode_jsonl(goldens)}
    head.update(files)
    gman = {"format_version": 1, "record_type": "a25_golden_manifest", "policy_id": POLICY_ID,
            "prompt_asset_sha256": sha256(asset), "goldens": len(goldens), "files": {n: sha256(d) for n, d in sorted(head.items())}}
    selection = {"format_version": 1, "record_type": "a25_golden_selection", "policy_id": POLICY_ID, "salt": SALT,
                 "sizes": list(sizes), "chosen": chosen,
                 "rule": "per size, the lowest SHA-256(salt + LF + selection ID) among materialized selections of that size"}
    root = {"format_version": 1, "record_type": "a25_goldens_manifest", "policy_id": POLICY_ID,
            "source": {"bundle": BUNDLE_NAME, "bundle_folder": b.name, "index_sha256": sha256(index_bytes), "readback": "clean"},
            "selection_sha256": sha256(encode_json(selection)), "tokenizer": getattr(tok, "identity", None),
            "counts": {"snapshots": len(chosen), "goldens": len(goldens),
                       "dataset_commands": sum(g["kind"] == "dataset_command" for g in goldens),
                       "written_fixtures": sum(g["kind"] == "written_fixture" for g in goldens)},
            "code": {f"grounding/quest/{p.name}": sha256(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "runtime": runtime()}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        for name, data in head.items():
            p = staging / "headset" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        (staging / "headset" / "golden-manifest.json").write_bytes(encode_json(gman))
        (staging / "selection.json").write_bytes(encode_json(selection))
        root["files"] = {p.relative_to(staging).as_posix(): sha256(p.read_bytes()) for p in sorted(staging.rglob("*")) if p.is_file()}
        (staging / "manifest.json").write_bytes(encode_json(root))
        publish(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    root["folder"] = str(out)
    return root


def verify_goldens(folder, *, tokenizer=None, protocol=None) -> list:
    """Every hash, and every document, mapping and prompt rebuilt from the headset files (tokenization too if given)."""
    f, bad = Path(folder), []
    try:
        root = json.loads((f / "manifest.json").read_text(encoding="utf-8"))
        gman = json.loads((f / "headset" / "golden-manifest.json").read_text(encoding="utf-8"))
        asset = json.loads((f / "headset" / "prompt-asset.json").read_text(encoding="utf-8"))
        goldens = [json.loads(x) for x in (f / "headset" / "goldens.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    except (OSError, ValueError) as e:
        return [f"unreadable goldens folder: {e}"]
    for rel, want in root["files"].items():
        if not (f / rel).is_file() or sha256((f / rel).read_bytes()) != want:
            bad.append(f"{rel}: missing or changed")
    for rel, want in gman["files"].items():
        if not (f / "headset" / rel).is_file() or sha256((f / "headset" / rel).read_bytes()) != want:
            bad.append(f"headset/{rel}: missing or changed")
    if gman["prompt_asset_sha256"] != sha256((f / "headset" / "prompt-asset.json").read_bytes()):
        bad.append("the golden manifest pins another prompt asset")
    proto = protocol if protocol is not None else load_protocol()
    for g in goldens:
        rid = g["request_id"]
        scene = json.loads((f / "headset" / g["scene_file"]).read_text(encoding="utf-8"))
        command = json.loads((f / "headset" / g["command_file"]).read_text(encoding="utf-8"))
        res = serialize(scene, command, format=FORMAT, relation_config_path=REL, direction_config_path=DIR)
        doc = res.document if res.status == "ok" else None
        if doc != g["document"] or sha256((doc or "").encode("utf-8")) != g["document_sha256"]:
            bad.append(f"{rid}: the document does not rebuild")
            continue
        lines = doc.splitlines(keepends=True)
        if lines[0] != asset["header_line"] or lines[1] != asset["semantics_line"]:
            bad.append(f"{rid}: the static lines differ from the prompt asset")
        ids = [o["object_id"] for o in sorted(scene["objects"], key=lambda o: o["object_id"])]
        mapping = choice_mapping(ids, proto)
        codes, targets = [m[0] for m in mapping], [m[1] for m in mapping]
        if (codes, targets) != (g["codes"], g["targets"]) or \
                RI.mapping_sha256(codes, targets, [asset["code_token_ids"][c] for c in codes]) != g["mapping_sha256"]:
            bad.append(f"{rid}: the mapping or its hash does not rebuild")
        pb = build_prompt(proto, mapping, doc).encode("utf-8")
        if sha256(pb) != g["prompt_sha256"] or len(pb) != g["prompt_bytes"]:
            bad.append(f"{rid}: the prompt does not rebuild")
        if RI.token_ids_sha256(g["token_ids"]) != g["token_ids_sha256"] or len(g["token_ids"]) != g["input_tokens"]:
            bad.append(f"{rid}: the token IDs do not match their hash or count")
        if tokenizer is not None and [int(x) for x in tokenizer.encode(pb.decode("utf-8"))] != g["token_ids"]:
            bad.append(f"{rid}: the tokenizer gives other token IDs")
        if not 0 < g["keep_before_command_line"] < g["keep_before_last_token"] == g["input_tokens"] - 1:
            bad.append(f"{rid}: the cache boundaries are not ordered as expected")
    return bad


def _shares(logits):
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    z = math.fsum(e)
    return [v / z for v in e]


def _validated(row, offered_ids, where) -> list:
    """The final position's row, refused when empty, non-numeric, non-finite, or too short for an offered token ID."""
    from ..inference.iref_vla.choices import last_position
    row = last_position(row)
    if not row or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
        raise ValueError(f"{where}: an empty, non-numeric or non-finite row")
    if max(offered_ids) >= len(row):
        raise ValueError(f"{where}: an offered token ID lies outside the {len(row)}-value row")
    return [float(x) for x in row]


class _Checked:
    """The model, with both of its outputs validated before anything compares them (the pilot's canary calls these)."""

    def __init__(self, model, offered_ids):
        self._m, self._ids, self.rows = model, offered_ids, {}

    def forward_last(self, ids):
        self.rows["production"] = _validated(self._m.forward_last(ids), self._ids, "production")
        return self.rows["production"]

    def independent_last(self, ids):
        self.rows["independent"] = _validated(self._m.independent_last(ids), self._ids, "independent")
        if len(self.rows["independent"]) != len(self.rows.get("production", self.rows["independent"])):
            raise ValueError("the production and independent rows differ in length")
        return self.rows["independent"]

    def __getattr__(self, name):
        return getattr(self._m, name)


def _keep_failed(out, record):
    failed = out.parent / f"{out.name}.failed"
    k = 1
    while failed.exists():
        k += 1
        failed = out.parent / f"{out.name}.failed-{k}"
    failed.mkdir(parents=True)
    (failed / "failure.json").write_bytes(encode_json(record))
    return failed


def golden_references(*, goldens, out, model_dir=None, device="cpu", model_loader=None, protocol=None) -> dict:
    """Fresh float32 references for every golden, into a new folder, published only after the pilot protocol's canary
    accepts: every offered logit within atol + rtol x |independent| of the independent path, and the same choice, with
    exact ties to K. Both outputs are validated first. A failure refuses publication and keeps its diagnostics in
    <out>.failed."""
    from ..inference.iref_vla import run as PILOT
    out = output.refuse_existing(out)
    g = Path(goldens)
    bad = verify_goldens(g, protocol=protocol)
    if bad:
        _fail(g, "the goldens do not read back: " + "; ".join(bad[:3]))
    proto = protocol if protocol is not None else load_protocol()
    rows = [json.loads(x) for x in (g / "headset" / "goldens.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    model_dir = Path(model_dir) if model_dir is not None else DEFAULT_HF
    evidence = None
    if model_loader is None:
        from ..inference.iref_vla.model import TorchModel, checkpoint_evidence
        evidence = checkpoint_evidence(model_dir, proto)
        model = TorchModel.load(model_dir, device, proto)
    else:
        model = model_loader(model_dir, device, proto)
    info = model.info() if hasattr(model, "info") else {}
    base = {"goldens_manifest_sha256": sha256((g / "manifest.json").read_bytes()), "device_requested": device,
            "model_info": info, "checkpoint_evidence": evidence, "runtime": runtime()}
    t0 = time.perf_counter()
    first = rows[0]
    checked = _Checked(model, first["code_token_ids"])
    crow = {"request_id": first["request_id"],
            "mapping": [[c, tg, i] for c, tg, i in zip(first["codes"], first["targets"], first["code_token_ids"])]}
    try:
        canary, error = PILOT._canary(checked, crow, first["token_ids"], proto), None
    except Exception as e:  # noqa: BLE001 - a malformed or non-finite output, or a refused score
        canary, error = None, f"{type(e).__name__}: {e}"
    decision = "accepted" if canary is not None and canary["accepted"] else "refused"
    canary_record = {"rule": proto["canary"]["rule"], "decision": decision, "record": canary, "error": error}
    if decision != "accepted":
        kept = _keep_failed(out, dict(base, record_type="a25_golden_references_failure", canary=canary_record))
        _fail(g, f"the canary failed ({error or 'outside tolerance or another choice'}); nothing was published; "
                 f"diagnostics kept in {kept}")
    refs = []
    for r in rows:
        try:
            row = _validated(model.forward_last(r["token_ids"]), r["code_token_ids"], r["request_id"])
        except ValueError as e:
            kept = _keep_failed(out, dict(base, record_type="a25_golden_references_failure", canary=canary_record,
                                          error=str(e), references_so_far=refs))
            _fail(g, f"{e}; nothing was published; diagnostics kept in {kept}")
        offered = [row[i] for i in r["code_token_ids"]]
        m = max(row)
        lse = m + math.log(math.fsum(math.exp(x - m) for x in row))
        top = max(offered)
        tied = [c for c, x in zip(r["codes"], offered) if x == top]
        ordered = sorted(offered, reverse=True)
        refs.append({"format_version": 1, "record_type": "a25_golden_reference", "request_id": r["request_id"],
                     "kind": r["kind"], "prompt_sha256": r["prompt_sha256"], "token_ids_sha256": r["token_ids_sha256"],
                     "mapping_sha256": r["mapping_sha256"],
                     "scores": [{"code": c, "token_id": i, "logit": x, "log_prob": x - lse, "restricted_share": s}
                                for c, i, x, s in zip(r["codes"], r["code_token_ids"], offered, _shares(offered))],
                     "choice_code": proto["ask_code"] if len(tied) > 1 else tied[0],
                     "selection_reason": "exact_score_tie" if len(tied) > 1 else "max_offered_logit",
                     "tied_codes": tied if len(tied) > 1 else [], "margin": ordered[0] - ordered[1]})
    man = dict(base, format_version=1, record_type="a25_golden_references_manifest", policy_id=POLICY_ID, canary=canary_record,
               seconds=round(time.perf_counter() - t0, 3), references=len(refs),
               code={f"grounding/quest/{p.name}": sha256(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))})
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=str(out.parent)))
    try:
        (staging / "references.jsonl").write_bytes(encode_jsonl(refs))
        man["files"] = {"references.jsonl": sha256((staging / "references.jsonl").read_bytes())}
        (staging / "manifest.json").write_bytes(encode_json(man))
        publish(staging, out)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise EvaluationOutputError([issue(str(out), "E_EVAL_OUTPUT_IO", f"{type(e).__name__}: {e}")]) from e
    man["folder"] = str(out)
    return man
