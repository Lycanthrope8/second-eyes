#!/usr/bin/env python3
"""A2.1d exact-token sizing audit (proposal v0.2, section 12). A temporary offline probe, not the serializer.

    python analysis/A2.1d_token_sizing/token_sizing_audit.py --out OUT_DIR
        [--runtime-lib LIB --runtime-gguf GGUF]      # llama.cpp's tokenizer through the app's se_tokenize
        [--hf-reference DIR]                         # A1's Hugging Face download, e.g. grounding/models/<name>/hf
        [--deployed-gguf GGUF --release-vocab GGUF]  # compare the deployed model's tokenizer with the release's
        [--a1-references DIR ...]                    # A1 runs' raw folders holding reference_*_fp32.json
        [--compare-with RESULTS_JSON]                # check this run reproduces a delivered results.json

Run from the repository root. It renders coordinates_v1 and coordinates_relations_v1 for four fixed inputs with the
accepted relation libraries, wraps each with the model description's chat template (grounding/scene.formatted, as A1's
headset jobs did), and tokenizes with the pinned tokenizer files tracked in the Unity folder. With --runtime-lib and
--runtime-gguf it also tokenizes everything with llama.cpp through se_tokenize (special tokens parsed, none added), as
the app does, and requires identical token IDs. Writes the exact inputs, token IDs, results.json and report.md.
Nothing here changes repository files. No model inference: the runtime library is used only to tokenize.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import itertools
import json
import math
import platform
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from grounding import scene as a1_scene  # noqa: E402  (A1's prompt construction)
from grounding.relations import directions as D  # noqa: E402
from grounding.relations import geometry as g  # noqa: E402
from grounding.relations import predicates as P  # noqa: E402

MODEL_DESC = REPO / "grounding" / "models" / "qwen2.5-0.5b-instruct.json"
TOKENIZER_DIR = REPO / "quest-app" / "Assets" / "SecondEyes" / "Models" / "qwen2.5-0.5b-instruct"
PROVIDER_ASSET = TOKENIZER_DIR / "ChatProvider.asset"
FX = REPO / "grounding" / "tests" / "fixtures"
A1_PROMPTS = REPO / "grounding" / "prompts"
PIN = REPO / "native" / "llama.cpp.pin"

# ---- Section 12.5: fixed audit text (verbatim) -------------------------------------------------------------------
SYSTEM = ("Use the supplied scene evidence to interpret the typed command. Unknown values are not false facts. "
          "Respect the stated spatial definitions and frame labels. Do not invent missing evidence or silently choose "
          "an unstated reference frame.")
OBJECT_COLUMNS = ["id", "category", "colours", "center_m", "size_m", "rotation_xyzw", "rotation_support",
                  "semantic_front", "conditional_fields"]
EVIDENCE_COLUMNS = ["category", "colours", "center_m", "size_m", "rotation_xyzw", "semantic_front"]
COVERAGE = ("relation blocks fully cover declared distinct-ID domains; exceptions override default; undeclared or "
            "out-of-domain tuples have no truth value; conditional exceptions invert the Boolean default; "
            "left=opposite(right), behind=opposite(in_front_of), with T/F exchanged and U unchanged")
DEFINITIONS = (
    "Centres and user positions are scene-frame metres; sizes are full local-axis lengths; rotations are xyzw "
    "local-to-scene quaternions; semantic front is separate. Directions project centre offsets onto horizontal axes. "
    "User-heading origin is user position, front is normalized heading, right=(front_y,-front_x). User-to-anchor "
    "origin is anchor centre, v=normalized XY(anchor-user), front=-v, right=(v_y,-v_x). Intrinsic origin is anchor "
    "centre, front=normalized XY(semantic_front), right=(front_y,-front_x). Direction score greater than band is T, "
    "below negative band F, otherwise U. Viewer-anchor horizontal length and semantic-front horizontal norm at or "
    "below their cutoffs make that frame U. Near uses shortest solid-box distance. Above uses target-bottom minus "
    "anchor-top gap and footprint overlap. On uses that gap within the support/penetration limits, overlap and "
    "eligible nearly upright anchor. Below is separated-below OR the eligible-furniture branch using bottom offset, "
    "top margin and overlap. Inside tests all target corners against eligible anchor outer-box bounds; this does not "
    "establish a cavity. Footprint overlap is intersection divided by smaller projected area. Between uses centre "
    "projection inside the anchor segment, lateral limit and each anchor overlap limit when vertically overlapping. "
    "Missing required evidence is U; existing relation gates and eligibility rules apply. Non-directional le(x,t,b): "
    "T if x<=t-b, F if x>t+b, else U; ge(x,t,b): T if x>=t+b, F if x<t-b, else U. Centre distances are 3D and "
    "support closest/farthest ranks 1-3 only after candidate eligibility is decided; adjacent gaps at or below "
    "rank_tie_m form linked ties. No ranking population or winner is supplied here. Derived states and distances "
    "remain conditional on their consulted assumptions. UNKNOWN is neither FALSE nor an executable target.")

# Section 12.4: the four inputs
CASES = [
    ("a17.annotated", "contract/valid/scene.a17.annotated.json", "contract/valid/command.a17.c001.json", None),
    ("a17.restricted", "contract/valid/scene.a17.restricted.json", "contract/valid/command.a17.c001.json", None),
    ("dirh10.annotated", "directions/scene.h.annotated.json", "directions/command.h.base.json", 10),
    ("dirh10.restricted", "directions/scene.h.restricted.json", "directions/command.h.base.json", 10),
]
FORMATS = ["coordinates_v1", "coordinates_relations_v1"]

# A1's recorded calibration (docs/llama-cpp.md, from A1.8b): the four prompts and their cached scene
A1_RECORDED = {"a17-fixed": 230, "a17-front": 232, "a17-left": 230, "a17-table": 227}
A1_RECORDED_SCENE = 191
# A1.8b (run 20260930_A1_r027, adb shell, Q8_0, 2 threads): median uncached whole-prompt time over the four prompts
A1_WHOLE_PROMPT_S, A1_WHOLE_PROMPT_TOKENS = 1.39, 230
DEPLOYED_GGUF_SHA256 = "dd753cd62f163c8baa8d2e598e3b61385f31cd46ca04488cd88ba01a9c83eb18"  # runs r027 and r032
QWEN25_ADDED = range(151646, 151665)  # Qwen2.5's added tokens that llama.cpp's release Qwen2 vocabulary lacks

REPRESENTATIVE = [
    "obj_001", "obj_010", "obj_999", "obj_1000", '["obj_001","obj_002"]', '[["obj_001","obj_002","obj_003"]]',
    "-0.34", "0.7856844150166147", "-1.0000005", "2.0186381547964457", "1e-06", "-0.020000000000000018", "0",
    "[2.6,1.4,0.37]", "[0,0,0.7071068,0.7071068]",
    '{"relation":"near","frame":null,"domain":"unordered_pairs","default":"F","exceptions":{"T":[],"U":[]},'
    '"conditional":{"default":false,"exceptions":[]}}',
    '":[["', ']]},"', "{}[]:,", '"\\"quoted\\"\\nline"',
    "Inspect the caf\u00e9\u2019s box \u2014 2 m \u201cleft\u201d \u6771\u4eac \U0001F600",
    "caf\u00e9 (NFC)", "cafe\u0301 (NFD)", "\u00c5ngstr\u00f6m", "A\u030angstro\u0308m",
]


# ---- Section 3: spelling ----------------------------------------------------------------------------------------
def number(x) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        raise ValueError("non-finite number")
    if x == 0:
        return "0"
    if x.is_integer():
        return str(int(x))
    return repr(x)


def enc(v) -> str:
    """Compact JSON with insertion-ordered keys, UTF-8 (ensure_ascii=False) and the section 3 number spelling."""
    if v is None:
        return "null"
    if isinstance(v, (bool, int, float)):
        return number(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(enc(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + enc(x) for k, x in v.items()) + "}"
    raise TypeError(type(v))


def shown(w):
    return w["value"] if w["state"] == "known" else None


def conditional(w) -> bool:
    return w["state"] == "known" and (w["evidence"]["kind"] == "assumed" or bool(w["evidence"]["assumptions"]))


def consulted_conditional(result) -> bool:
    return any(r.state == "known" and (r.kind == "assumed" or bool(r.assumptions)) for r in result.inputs)


STATE = {P.TRUE: "T", P.FALSE: "F", P.UNKNOWN: "U"}


# ---- rendering (sections 3-6) -----------------------------------------------------------------------------------
def object_row(o):
    gm = o["geometry"]
    wrappers = {"category": o["category"], "colours": o["attributes"]["colours"], "center_m": gm["center_m"],
                "size_m": gm["size_m"], "rotation_xyzw": gm["rotation_xyzw"], "semantic_front": o["semantic_front"]}
    cat = shown(o["category"])
    colours = shown(o["attributes"]["colours"])
    rot = gm["rotation_xyzw"]
    return [o["object_id"], cat["model"] if cat else None, sorted(colours) if colours is not None else None,
            shown(gm["center_m"]), shown(gm["size_m"]), shown(rot), rot.get("support") if rot["state"] == "known"
            else None, shown(o["semantic_front"]), [c for c in EVIDENCE_COLUMNS if conditional(wrappers[c])]]


def pose_record(cmd):
    up = cmd["user_pose"]
    heading = up["heading_xy"]
    return {"pose_kind": up["pose_kind"], "position_m": shown(up["position_m"]), "heading_xy": shown(heading),
            "heading_source": heading.get("heading_source") if heading["state"] == "known" else None,
            "conditional_fields": [n for n in ("position_m", "heading_xy") if conditional(up[n])]}


def compress(relation, frame, domain, results):
    """results: {tuple: (state, conditional)} over the complete domain -> the section 6 block."""
    counts = Counter(s for s, _ in results.values())
    top = max(counts.values(), default=0)
    default = next(s for s in "UFT" if counts[s] == top) if results else "U"
    exceptions = {s: sorted(list(t) for t, (st, _) in results.items() if st == s) for s in "TFU" if s != default}
    n_true = sum(1 for _, c in results.values() if c)
    cond_default = n_true > len(results) - n_true
    cond_exc = sorted(list(t) for t, (_, c) in results.items() if c != cond_default)
    return {"relation": relation, "frame": frame, "domain": domain, "default": default, "exceptions": exceptions,
            "conditional": {"default": cond_default, "exceptions": cond_exc}}


def expand(block, domain_tuples):
    """Inverse of compress, for the self-check."""
    out = {}
    lists = {s: {tuple(t) for t in block["exceptions"][s]} for s in block["exceptions"]}
    cexc = {tuple(t) for t in block["conditional"]["exceptions"]}
    for t in domain_tuples:
        st = next((s for s, members in lists.items() if t in members), block["default"])
        out[t] = (st, (not block["conditional"]["default"]) if t in cexc else block["conditional"]["default"])
    return out


class Calls:
    def __init__(self):
        self.n = Counter()


def render(scene_rec, cmd, fmt, rel_cfg, dir_cfg, calls: Calls, checks: list):
    """Returns (lines, line_names, static_count): JSON-lines strings without their LF."""
    rel = P.Relations(scene_rec, rel_cfg)
    drel = D.DirectionalRelations(scene_rec, dir_cfg)
    objs = sorted(scene_rec["objects"], key=lambda o: o["object_id"])
    ids = [o["object_id"] for o in objs]
    header = {"serializer_version": 1, "format": fmt, "axes": {"handedness": "right", "up": "+z", "units": "m"},
              "object_columns": OBJECT_COLUMNS, "unknown": "null means unknown; [] means known-empty colours",
              "conditional_fields": "named displayed fields depend on assumptions", "coverage": COVERAGE}
    params = {"thresholds": dict(sorted(rel_cfg.thresholds.items())), "bands": dict(sorted(rel_cfg.bands.items())),
              "directions": dict(sorted({"direction_band_m": dir_cfg.direction_band_m,
                                         "viewer_anchor_min_horizontal_m": dir_cfg.viewer_anchor_min_horizontal_m,
                                         "semantic_front_min_horizontal_norm":
                                             dir_cfg.semantic_front_min_horizontal_norm}.items()))}
    cats = {k: sorted(rel_cfg.categories[k]) for k in ("support", "furniture", "container")}
    lines = [enc({"header": header}), enc({"semantics": {"definitions": DEFINITIONS, "parameters": params,
                                                         "categories": cats}}),
             enc({"objects": [object_row(o) for o in objs]})]
    names = ["header", "semantics", "objects"]
    pairs, ordered = list(itertools.combinations(ids, 2)), list(itertools.permutations(ids, 2))
    triples = [(t, a, b) for t in ids for a, b in itertools.combinations([x for x in ids if x != t], 2)]
    augmented = fmt == "coordinates_relations_v1"

    def evaluate(name, tuples, fn):
        res = {}
        for t in tuples:
            r = fn(*t)
            calls.n[name] += 1
            res[t] = (STATE[r.value], consulted_conditional(r))
        return res

    def emit(name, block, domain_tuples, res):
        if expand(block, domain_tuples) != res:
            checks.append(f"FAIL compression of {name}")
        lines.append(enc(block))
        names.append(name)

    if augmented:
        res = evaluate("near", pairs, rel.near)
        emit("near", compress("near", None, "unordered_pairs", res), pairs, res)
        for name in ("above", "below", "on", "inside"):
            res = evaluate(name, ordered, getattr(rel, name))
            emit(name, compress(name, None, "ordered_pairs", res), ordered, res)
        res = evaluate("between", triples, rel.between)
        emit("between", compress("between", None, "target_anchor_pairs", res), triples, res)
        for name in ("right", "in_front_of"):
            res = evaluate(f"intrinsic.{name}", ordered,
                           lambda t, a, n=name: drel.evaluate(n, t, frame="object_intrinsic", anchor_id=a))
            emit(f"intrinsic.{name}", compress(name, "object_intrinsic", "ordered_pairs", res), ordered, res)
            mirror_check(drel, name, ordered, res, "object_intrinsic", None, checks, calls)
        cen = {o["object_id"]: o["geometry"]["center_m"] for o in objs}
        values, cres = [], {}
        for a, b in pairs:
            ca, cb = shown(cen[a]), shown(cen[b])
            d = g.norm(g.sub(cb, ca)) if ca is not None and cb is not None else None
            if d is not None:
                calls.n["center_distance"] += 1
            values.append([a, b, d])
            cres[(a, b)] = ("U", conditional(cen[a]) or conditional(cen[b]))
        cblock = compress("x", None, "unordered_pairs", cres)["conditional"]
        lines.append(enc({"measure": "center_distance_m", "domain": "unordered_pairs", "values": values,
                          "conditional": cblock}))
        names.append("center_distance_m")
    static_count = len(lines)
    lines.append(enc({"pose": pose_record(cmd)}))
    names.append("pose")
    if augmented:
        objs1 = [(t,) for t in ids]
        for name in ("right", "in_front_of"):
            res = evaluate(f"user_heading.{name}", objs1,
                           lambda t, n=name: drel.evaluate(n, t, frame="user_heading", command=cmd))
            emit(f"user_heading.{name}", compress(name, "user_heading", "objects", res), objs1, res)
            mirror_check(drel, name, objs1, res, "user_heading", cmd, checks, calls)
        for name in ("right", "in_front_of"):
            res = evaluate(f"user_to_anchor.{name}", ordered,
                           lambda t, a, n=name: drel.evaluate(n, t, frame="user_to_anchor", anchor_id=a, command=cmd))
            emit(f"user_to_anchor.{name}", compress(name, "user_to_anchor", "ordered_pairs", res), ordered, res)
            mirror_check(drel, name, ordered, res, "user_to_anchor", cmd, checks, calls)
    lines.append(enc({"command": cmd["text"]}))
    names.append("command")
    for line in lines:
        if json.loads(line) is None or "\n" in line:
            checks.append("FAIL a line is not one JSON value")
    return lines, names, static_count


def mirror_check(drel, name, tuples, res, frame, cmd, checks, calls):
    """Left/behind reconstructed as the opposite of right/in_front_of must equal the evaluator's (not serialized)."""
    other = {"right": "left", "in_front_of": "behind"}[name]
    flip = {"T": "F", "F": "T", "U": "U"}
    for t in tuples:
        kw = {"frame": frame}
        if len(t) == 2:
            kw["anchor_id"] = t[1]
        if cmd is not None:
            kw["command"] = cmd
        r = drel.evaluate(other, t[0], **kw)
        calls.n["verification_only"] += 1
        if STATE[r.value] != flip[res[t][0]]:
            checks.append(f"FAIL mirror {frame} {name} {t}")


# ---- tokenizers -------------------------------------------------------------------------------------------------
class HFTokenizer:
    def __init__(self, folder):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(folder))
        shown_path = Path(folder).resolve()
        if REPO in shown_path.parents:
            shown_path = shown_path.relative_to(REPO)
        self.name = f"Hugging Face {type(self.tok).__name__} from {shown_path.as_posix()}"

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]


class RuntimeTokenizer:
    """llama.cpp through the app's se_tokenize (special tokens parsed, none added). Loads the model file, no inference."""

    def __init__(self, lib_path, gguf_path):
        lib = C.CDLL(str(lib_path))
        lib.se_load.restype, lib.se_load.argtypes = C.c_void_p, [C.c_char_p, C.c_int32, C.c_int32]
        lib.se_tokenize.restype = C.c_int32
        lib.se_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int32), C.c_int32]
        lib.se_llama_version.restype = C.c_char_p
        lib.se_last_error.restype = C.c_char_p
        self.lib, self.s = lib, lib.se_load(str(gguf_path).encode(), 512, 1)
        if not self.s:
            raise RuntimeError(lib.se_last_error().decode())
        self.name = f"llama.cpp {lib.se_llama_version().decode()} se_tokenize with {Path(gguf_path).name}"

    def ids(self, text):
        b = text.encode("utf-8")
        buf = (C.c_int32 * (len(b) + 16))()
        n = self.lib.se_tokenize(self.s, b, buf, len(buf))
        if n < 0:
            buf = (C.c_int32 * (-n))()
            n = self.lib.se_tokenize(self.s, b, buf, len(buf))
        return list(buf[:n])


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids) -> str:
    return sha256_bytes(",".join(map(str, ids)).encode())


def common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ---- identity, wrapper and calibration ----------------------------------------------------------------------------
def identity(args, tokenizers):
    desc = json.loads(MODEL_DESC.read_text(encoding="utf-8"))
    model_json = json.loads((TOKENIZER_DIR / "model.json").read_text(encoding="utf-8"))
    files = {}
    for f in model_json["tokenizer_files"]:
        actual = sha256_file(TOKENIZER_DIR / f["file"])
        files[f["file"]] = {"sha256": actual, "matches_model_json_fingerprint": actual == f["sha256"]}
    out = {"model_id": desc["hf_repo"], "hf_revision": desc["hf_revision"],
           "tokenizer_files_revision": model_json["hf_commit"], "tokenizer_files": files,
           "tokenizer_config": {k: v for k, v in json.loads((TOKENIZER_DIR / "tokenizer_config.json").read_text(
               encoding="utf-8")).items() if k in ("add_bos_token", "bos_token", "eos_token", "pad_token",
                                                   "tokenizer_class", "split_special_tokens", "add_prefix_space")},
           "chat_template": desc["chat_template"], "chat_template_sha256": sha256_bytes(desc["chat_template"].encode()),
           "model_description_sha256": sha256_file(MODEL_DESC),
           "wrapper_source": "grounding/scene.py formatted(): template with {1} = system, {0} = user (A1 headset jobs); "
                             "Meta's OnDeviceLlmConfig.ApplyChatTemplate is template.format(user, system)",
           "llama_cpp_pin": PIN.read_text(encoding="utf-8").split()[:2],
           "tokenizers": [t.name for t in tokenizers],
           "special_token_handling": "special tokens in the text are parsed; no BOS or EOS is added "
                                     "(se_tokenize add_special=false, parse_special=true; HF add_special_tokens=False)",
           "assistant_prefix": "<|im_start|>assistant\\n, once, from the template; no answer prefix",
           "python": platform.python_version()}
    try:
        import tokenizers as tk
        import transformers
        out["libraries"] = {"transformers": transformers.__version__, "tokenizers": tk.__version__}
    except ImportError:
        pass
    asset = PROVIDER_ASSET.read_text(encoding="utf-8") if PROVIDER_ASSET.exists() else ""
    out["provider_asset_has_template"] = "chatTemplateFormat" in asset
    return out, desc


def wrap(desc, user):
    a = a1_scene.formatted(desc["chat_template"], SYSTEM, user)
    b = desc["chat_template"].format(user, SYSTEM)  # Meta's runner and the app's provider
    if a != b:
        raise RuntimeError("the two A1 template paths disagree")
    return a


def calibration(desc, tokenizers, a1_reference_dirs):
    rows = {}
    for pid, recorded in A1_RECORDED.items():
        p = json.loads((A1_PROMPTS / f"{pid}.json").read_text(encoding="utf-8"))
        f = a1_scene.formatted(desc["chat_template"], p["system"], p["user"])
        cut = a1_scene.scene_chars(f)
        per = {}
        for t in tokenizers:
            ids, pre = t.ids(f), t.ids(f[:cut])
            per[t.name] = {"tokens": len(ids), "scene_tokens": len(pre), "ids_sha256": ids_hash(ids),
                           "scene_is_prefix": ids[:len(pre)] == pre}
        rows[pid] = {"recorded_tokens": recorded, "recorded_scene_tokens": A1_RECORDED_SCENE, "measured": per,
                     "formatted_sha256": sha256_bytes(f.encode())}
        for d in a1_reference_dirs:
            ref = Path(d) / f"reference_{pid}_fp32.json"
            if ref.exists():
                want = json.loads(ref.read_text(encoding="utf-8"))["prompt"]["token_ids"]
                rows[pid]["recorded_reference_ids"] = {t.name: t.ids(f) == want for t in tokenizers}
                rows[pid]["recorded_reference_file"] = ref.as_posix()
    return rows


def release_vs_deployed(deployed, release):
    sys.path.insert(0, str(Path(release).resolve().parents[1] / "gguf-py"))
    import gguf

    def fields(path):
        r = gguf.GGUFReader(str(path))
        out = {}
        for name, f in r.fields.items():
            if not name.startswith("tokenizer.ggml."):
                continue
            if f.types and f.types[0] == gguf.GGUFValueType.ARRAY:
                if f.types[-1] == gguf.GGUFValueType.STRING:
                    out[name] = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
                else:
                    out[name] = [int(f.parts[i][0]) for i in f.data]
            elif f.types and f.types[0] == gguf.GGUFValueType.STRING:
                out[name] = bytes(f.parts[f.data[0]]).decode("utf-8")
            else:
                out[name] = f.parts[f.data[0]].tolist()[0]
        return out

    a, b = fields(deployed), fields(release)
    digest = sha256_file(Path(deployed))
    report = {"deployed_sha256": digest, "deployed_sha256_matches_runs": digest == DEPLOYED_GGUF_SHA256, "keys": {}}
    for k in sorted(set(a) | set(b)):
        if isinstance(a.get(k), list) and isinstance(b.get(k), list):
            diff = [i for i in range(max(len(a[k]), len(b[k])))
                    if i >= len(a[k]) or i >= len(b[k]) or a[k][i] != b[k][i]]
            report["keys"][k] = {"equal": not diff, "lengths": [len(a[k]), len(b[k])], "first_differences": diff[:25],
                                 "n_differences": len(diff)}
        else:
            report["keys"][k] = {"equal": a.get(k) == b.get(k), "deployed": a.get(k), "release": b.get(k)}
    keys = report["keys"]
    tok_diffs = set(keys.get("tokenizer.ggml.tokens", {}).get("first_differences", []))
    confined = (keys.get("tokenizer.ggml.tokens", {}).get("n_differences", 1) <= len(QWEN25_ADDED)
                and tok_diffs <= set(QWEN25_ADDED))
    report["equivalent_for_audited_texts"] = bool(
        confined and keys.get("tokenizer.ggml.merges", {}).get("equal") and keys.get("tokenizer.ggml.pre", {}).get(
            "equal") and keys.get("tokenizer.ggml.model", {}).get("equal")
        and keys.get("tokenizer.ggml.tokens", {}).get("lengths", [0, 1])[0] == keys.get(
            "tokenizer.ggml.tokens", {}).get("lengths", [1, 0])[1])
    report["rule"] = ("equivalent for the audited texts if the BPE model, pre-tokenizer and merges are equal and the "
                      "token lists differ only at Qwen2.5's added tokens 151646-151664, none of which occurs in any "
                      "audited text; BOS/EOS IDs don't matter because none is added")
    return report


# ---- main -------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime-lib")
    ap.add_argument("--runtime-gguf")
    ap.add_argument("--hf-reference")
    ap.add_argument("--deployed-gguf")
    ap.add_argument("--release-vocab")
    ap.add_argument("--a1-references", nargs="*", default=[])
    ap.add_argument("--compare-with")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)

    skipped = {}

    def present(label, *paths):
        missing = [p for p in paths if p and not Path(p).exists()]
        if missing:
            skipped[label] = f"not found: {', '.join(missing)}"
        return all(paths) and not missing

    primary = HFTokenizer(TOKENIZER_DIR)
    tokenizers = [primary]
    if args.runtime_lib or args.runtime_gguf:
        if present("runtime tokenizer", args.runtime_lib, args.runtime_gguf):
            tokenizers.append(RuntimeTokenizer(args.runtime_lib, args.runtime_gguf))
    if args.hf_reference and present("A1 Hugging Face download", args.hf_reference):
        tokenizers.append(HFTokenizer(args.hf_reference))
    args.a1_references = [d for d in args.a1_references if present(f"A1 references {d}", d)]
    ident, desc = identity(args, tokenizers)
    rel_cfg, dir_cfg = P.load_config(), D.load_direction_config()
    results = {"identity": ident, "configs": {"relations": rel_cfg.identity, "directions": dir_cfg.identity},
               "calibration": calibration(desc, tokenizers, args.a1_references), "inputs": {}, "self_checks": [],
               "parity": {}, "extrapolation": {
                   "context": "A1.8b run 20260930_A1_r027: median uncached whole-prompt time of the four A1 prompts "
                              "(227-232 tokens) through se_llama_cli from adb shell, Q8_0, 2 threads, plugged in; "
                              "llama-bench pp230 at 2 threads gave 170 tokens/s",
                   "tokens_per_second": A1_WHOLE_PROMPT_TOKENS / A1_WHOLE_PROMPT_S,
                   "label": "linear extrapolation from A1, NOT measured Quest latency"}}
    if (args.deployed_gguf or args.release_vocab) and present("deployed GGUF check", args.deployed_gguf,
                                                               args.release_vocab):
        results["deployed_vs_release_tokenizer"] = release_vs_deployed(args.deployed_gguf, args.release_vocab)
    results["skipped"] = skipped

    token_ids, parity_texts = {}, {f"representative[{i}]": s for i, s in enumerate(REPRESENTATIVE)}
    for pid in A1_RECORDED:
        p = json.loads((A1_PROMPTS / f"{pid}.json").read_text(encoding="utf-8"))
        parity_texts[f"A1 {pid}"] = a1_scene.formatted(desc["chat_template"], p["system"], p["user"])
    chosen_by_case = {}
    for case, scene_file, cmd_file, first_n in CASES:
        scene_rec = json.loads((FX / scene_file).read_text(encoding="utf-8"))
        cmd = json.loads((FX / cmd_file).read_text(encoding="utf-8"))
        all_ids = sorted(o["object_id"] for o in scene_rec["objects"])
        chosen = all_ids[:first_n] if first_n else all_ids
        chosen_by_case[case] = chosen
        sub = dict(scene_rec, objects=[o for o in scene_rec["objects"] if o["object_id"] in chosen])
        for fmt in FORMATS:
            calls, checks = Calls(), []
            lines, names, n_static = render(sub, cmd, fmt, rel_cfg, dir_cfg, calls, checks)
            shuffled = dict(sub, objects=list(reversed(sub["objects"])))
            again, _, _ = render(shuffled, cmd, fmt, rel_cfg, dir_cfg, Calls(), [])
            if again != lines:
                checks.append("FAIL object order changed the bytes")
            doc = "".join(line + "\n" for line in lines)
            static, dynamic = "".join(x + "\n" for x in lines[:n_static]), "".join(x + "\n" for x in lines[n_static:])
            prompt = wrap(desc, doc)
            head = prompt[:prompt.index(doc)]
            tail = prompt[len(head) + len(doc):]
            key = f"{case}.{fmt}"
            (out / "inputs" / f"{key}.jsonl").write_bytes(doc.encode("utf-8"))
            (out / "inputs" / f"{key}.prompt.txt").write_bytes(prompt.encode("utf-8"))
            parity_texts[f"{key}.prompt"] = prompt
            parity_texts[f"{key}.document"] = doc
            parity_texts[f"{key}.static_prefix_input"] = head + static
            ids = primary.ids(prompt)
            doc_ids = primary.ids(doc)
            # marginal attribution over successive complete prefixes
            cum, marginal, prev = head, {"wrapper_head": len(primary.ids(head))}, len(primary.ids(head))
            for name, line in zip(names, lines):
                cum += line + "\n"
                now = len(primary.ids(cum))
                marginal[name] = now - prev
                prev = now
            marginal["wrapper_tail"] = len(ids) - prev
            prefix_ids = primary.ids(head + static)
            common = common_prefix(prefix_ids, ids)
            # conditional diagnostics (the delivered input keeps every field)
            stripped_blocks = []
            for line in lines:
                v = json.loads(line)
                if isinstance(v, dict) and "conditional" in v and ("relation" in v or "measure" in v):
                    v = {k: x for k, x in v.items() if k != "conditional"}
                stripped_blocks.append(enc(v))
            stripped_rows = []
            for name, line in zip(names, lines):
                v = json.loads(line)
                if name == "objects":
                    v = {"objects": [r[:-1] for r in v["objects"]]}
                elif name == "pose":
                    v = {"pose": {k: x for k, x in v["pose"].items() if k != "conditional_fields"}}
                stripped_rows.append(enc(v))
            doc_no_block_cond = "".join(x + "\n" for x in stripped_blocks)
            doc_no_row_cond = "".join(x + "\n" for x in stripped_rows)
            no_block_cond = len(primary.ids(wrap(desc, doc_no_block_cond)))
            no_row_cond = len(primary.ids(wrap(desc, doc_no_row_cond)))
            n = len(chosen)
            formula = 9 * n * (n - 1) + n * (n - 1) * (n - 2) // 2 + 2 * n
            serializer_calls = sum(v for k, v in calls.n.items() if k != "verification_only")
            if fmt == "coordinates_relations_v1" and serializer_calls != formula:
                checks.append(f"FAIL work {serializer_calls} != formula {formula}")
            if any(enc(json.loads(line)) != line for line in lines):
                checks.append("FAIL a line does not re-encode to itself")
            rate = results["extrapolation"]["tokens_per_second"]
            results["inputs"][key] = {
                "case": case, "format": fmt, "scene": scene_file, "command": cmd_file, "objects": chosen,
                "scene_id": sub["scene_id"], "scene_revision": sub["scene_revision"],
                "evidence_profile": sub["evidence_profile"], "command_id": cmd["command_id"],
                "bytes": {"wrapped": len(prompt.encode()), "document": len(doc.encode()),
                          "static_prefix": len(static.encode()), "dynamic_suffix": len(dynamic.encode()),
                          "wrapper_head": len(head.encode()), "wrapper_tail": len(tail.encode()),
                          "by_block": {nm: len((ln + "\n").encode()) for nm, ln in zip(names, lines)}},
                "tokens": {"wrapped": len(ids), "document_alone": len(doc_ids),
                           "static_lines": sum(marginal[x] for x in names[:n_static]),
                           "dynamic_lines": sum(marginal[x] for x in names[n_static:]),
                           "marginal": marginal, "marginal_sum_equals_total": sum(marginal.values()) == len(ids)},
                "warm_prefix": {"candidate_prefix": "wrapper head (system turn and '<|im_start|>user\\n') + static "
                                                    "prefix lines", "candidate_prefix_tokens": len(prefix_ids),
                                "wrapper_head_tokens": marginal["wrapper_head"], "common_prefix_tokens": common,
                                "tokens_after_reusable_prefix": len(ids) - common},
                "conditional_diagnostic": {
                    "tokens_without_block_conditional_maps": no_block_cond,
                    "block_conditional_tokens": len(ids) - no_block_cond,
                    "block_conditional_bytes": len(doc.encode()) - len(doc_no_block_cond.encode()),
                    "tokens_without_row_and_pose_conditional_fields": no_row_cond,
                    "row_and_pose_conditional_tokens": len(ids) - no_row_cond,
                    "row_and_pose_conditional_bytes": len(doc.encode()) - len(doc_no_row_cond.encode())},
                "work": {"n_objects": n, "formula_units": formula if fmt == "coordinates_relations_v1" else 0,
                         "serializer_calls_and_distances": serializer_calls, "by_block": dict(calls.n),
                         "verification_calls_not_serialized": calls.n.get("verification_only", 0)},
                "seconds_linear_extrapolation": {"cold": len(ids) / rate, "warm_best_case": (len(ids) - common) / rate},
                "ids_sha256": {"wrapped": ids_hash(ids), "document": ids_hash(doc_ids)},
                "text_sha256": {"wrapped": sha256_bytes(prompt.encode()), "document": sha256_bytes(doc.encode())},
                "self_checks": checks or ["all passed"]}
            results["self_checks"] += [f"{key}: {c}" for c in checks]
            token_ids[key] = {"wrapped": ids, "document": doc_ids}

    for a, b in (("a17.annotated", "a17.restricted"), ("dirh10.annotated", "dirh10.restricted")):
        if chosen_by_case[a] != chosen_by_case[b]:
            results["self_checks"].append(f"FAIL {a} and {b} hold different IDs")
    results["matched_profile_ids"] = {c: ids for c, ids in chosen_by_case.items()}
    # parity: every tokenizer must give the same IDs on every text
    for t in tokenizers[1:]:
        diffs = []
        for name, text in parity_texts.items():
            a, b = primary.ids(text), t.ids(text)
            if a != b:
                diffs.append({"text": name, "primary": len(a), "other": len(b), "first_difference": common_prefix(a, b)})
        results["parity"][t.name] = {"texts": len(parity_texts), "identical": len(parity_texts) - len(diffs),
                                     "differences": diffs}
    template_tokens = {"<|im_start|>", "<|im_end|>"}
    results["added_token_strings_in_texts"] = sorted({tok for tok in primary.tok.get_added_vocab()
                                                      for text in parity_texts.values()
                                                      if tok in text and tok not in template_tokens})
    results["nfc_note"] = {s: unicodedata.is_normalized("NFC", s) for s in REPRESENTATIVE if not s.isascii()}
    if args.compare_with:
        if present("comparison", args.compare_with):
            old = json.loads(Path(args.compare_with).read_text(encoding="utf-8"))
            results["comparison"] = {k: {"same_text": old["inputs"].get(k, {}).get("text_sha256") == v["text_sha256"],
                                         "same_token_ids": old["inputs"].get(k, {}).get("ids_sha256") == v["ids_sha256"]}
                                     for k, v in results["inputs"].items()}
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "token_ids.json").write_text(json.dumps(token_ids) + "\n", encoding="utf-8")
    (out / "report.md").write_text(report(results), encoding="utf-8")
    print(report(results))
    return 1 if any("FAIL" in c for c in results["self_checks"]) else 0


def report(r) -> str:
    rate = r["extrapolation"]["tokens_per_second"]
    out = ["# A2.1d token-sizing audit: results", "",
           f"Tokenizers: {'; '.join(r['identity']['tokenizers'])}. Configurations: {r['configs']['relations']}, "
           f"{r['configs']['directions']}.", "", "## Calibration against A1's records", "",
           "| A1 prompt | Recorded | Measured (each tokenizer) | Scene recorded | Scene measured |", "|---|---|---|---|---|"]
    for pid, row in r["calibration"].items():
        out.append(f"| {pid} | {row['recorded_tokens']} | " + ", ".join(str(m["tokens"]) for m in row["measured"].values())
                   + f" | {row['recorded_scene_tokens']} | " + ", ".join(str(m["scene_tokens"]) for m in
                                                                        row["measured"].values()) + " |")
    out += ["", "## Parity", ""]
    for name, p in r["parity"].items():
        out.append(f"- {name}: {p['identical']} of {p['texts']} texts give identical token IDs"
                   + (": differences in " + ", ".join(d["text"] for d in p["differences"]) if p["differences"] else ""))
    if not r["parity"]:
        out.append("- only one tokenizer ran: no parity measured")
    out += ["", "## Whole inputs", "",
            "| Input | Bytes | Tokens | Document alone | Static lines | Dynamic lines | After reusable prefix | "
            "Cold s* | Warm best-case s* | Work |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in r["inputs"].items():
        out.append(f"| {k} | {v['bytes']['wrapped']} | {v['tokens']['wrapped']} | {v['tokens']['document_alone']} | "
                   f"{v['tokens']['static_lines']} | {v['tokens']['dynamic_lines']} | "
                   f"{v['warm_prefix']['tokens_after_reusable_prefix']} | {v['seconds_linear_extrapolation']['cold']:.1f}"
                   f" | {v['seconds_linear_extrapolation']['warm_best_case']:.1f} | "
                   f"{v['work']['serializer_calls_and_distances']} |")
    out += ["", f"*{r['extrapolation']['label']}: {rate:.1f} tokens/s ({r['extrapolation']['context']}).", "",
            "## Marginal tokens by block", ""]
    keys = list(r["inputs"])
    blocks = []
    for k in keys:
        blocks += [b for b in r["inputs"][k]["tokens"]["marginal"] if b not in blocks]
    out.append("| Block | " + " | ".join(keys) + " |")
    out.append("|---" * (len(keys) + 1) + "|")
    for b in blocks:
        out.append(f"| {b} | " + " | ".join(str(r["inputs"][k]["tokens"]["marginal"].get(b, "")) for k in keys) + " |")
    out += ["", "## Conditional membership (diagnostic differences; the inputs keep these fields)", "",
            "| Input | Block conditional maps | Object and pose conditional_fields |", "|---|---|---|"]
    for k, v in r["inputs"].items():
        c = v["conditional_diagnostic"]
        out.append(f"| {k} | {c['block_conditional_tokens']} tokens, {c['block_conditional_bytes']} B | "
                   f"{c['row_and_pose_conditional_tokens']} tokens, {c['row_and_pose_conditional_bytes']} B |")
    if "deployed_vs_release_tokenizer" in r:
        d = r["deployed_vs_release_tokenizer"]
        out += ["", "## Deployed GGUF against the release vocabulary", "",
                f"SHA-256 matches runs r027/r032: {d['deployed_sha256_matches_runs']}; equivalent for the audited "
                f"texts: {d['equivalent_for_audited_texts']}.", ""]
        out += [f"- {k}: " + ("equal" if v["equal"] else f"different {({x: y for x, y in v.items() if x != 'equal'})}")
                for k, v in d["keys"].items()]
    if "comparison" in r:
        out += ["", "## Against the delivered results", ""]
        out += [f"- {k}: text {'same' if v['same_text'] else 'DIFFERENT'}, token IDs "
                f"{'same' if v['same_token_ids'] else 'DIFFERENT'}" for k, v in r["comparison"].items()]
    out += ["", f"Self-checks: {r['self_checks'] or 'all passed'}. Skipped: {r['skipped'] or 'nothing'}.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
