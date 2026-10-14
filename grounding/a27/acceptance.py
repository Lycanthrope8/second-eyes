"""A2.7 delivery 2 (D115): the port's acceptance material on real data, built on the PC that holds A2.6's work.

    python -m grounding.a27 port-goldens --bundle BUNDLE --prep PREP --work WORK --out FOLDER

From the Delivery 1 bundle (BUNDLE: the frozen reference outputs and A2.4's recorded outcomes): every parent in both
views becomes one frozen resolution (criterion 1): the reference output's interpretation, the view's scene and command
records, the scene's category map, and A2.4's recorded outcome, target, candidate set and technical flag. Every
distinct reference output and the grammar's 97-string corpus become reader cases (criterion 2): Python's verdict,
interpretation and query. Records are stored once and referenced by index. The headset subset's parents are marked, so
Delivery 3 can push that part to the headset. Judged by tools/csharp-tests/Resolution (mode acceptance) on the PC, and
on the headset in Delivery 3.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..a24 import evaluate as E
from ..a24 import gbnf_check as GC
from ..a24 import language as L
from ..evaluation.iref_vla.protocol import encode_json


def build(*, bundle, prep, work, out) -> dict:
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} already exists")
    body = build_body(bundle=bundle, prep=prep, work=work)
    out.mkdir(parents=True)
    data = encode_json(body)
    (out / "port-acceptance.json").write_bytes(data)
    summary = {"resolutions": len(body["resolutions"]), "headset_resolutions": sum(1 for x in body["resolutions"] if x["headset"]),
               "records": len(body["records"]), "readers": len(body["readers"]), "sha256": hashlib.sha256(data).hexdigest(),
               "bytes": len(data)}
    (out / "summary.json").write_bytes(encode_json(summary))
    return summary


def build_body(*, bundle, prep, work, only_headset=False, parents=None) -> dict:
    """parents: A2.4's parent list, when the caller has loaded it already (it verifies every A2.6 bundle: slow)."""
    bundle, work = Path(bundle), Path(work)
    man = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (bundle / "requests.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if hashlib.sha256((bundle / "requests.jsonl").read_bytes()).hexdigest() != man["files"]["requests.jsonl"]:
        raise SystemExit("the bundle's requests changed since it was built")
    parents = parents or {p["parent_command_id"]: p for p in E._parents(prep, work)}
    spec = L.load_spec()
    records, index = [], {}

    def ref(path: Path) -> int:
        body = path.read_bytes()
        key = hashlib.sha256(body).hexdigest()
        if key not in index:
            index[key] = len(records)
            records.append(json.loads(body))
        return index[key]
    resolutions, texts = [], {}
    for r in rows:
        if only_headset and not r["headset"]:
            continue
        p = parents[r["parent_command_id"]]
        b = work / p["scene"] / "bundle"
        ref_out = r["reference"]
        texts.setdefault(ref_out["text"], None)
        interp = None
        if ref_out["valid"]:
            interp = L.interpretation(L.parse_output(ref_out["text"], spec))
        for v in E.VIEWS:
            vp = p["view_paths"][v]
            e = ref_out["views"][v]
            resolutions.append({"parent_command_id": r["parent_command_id"], "view": v, "headset": r["headset"],
                                "stratum": r["stratum"], "interp": interp,
                                "scene": ref(b / vp["scene_path"]), "command": ref(b / vp["command_path"]),
                                "map": ref(b / "model_records" / "category-map.json"),
                                "expected": {"outcome": e["outcome"], "target_id": e.get("target_id"),
                                             "candidate_ids": e.get("candidate_ids"), "technical": bool(e.get("technical"))}})
    readers = []
    for t in list(texts) + [s for s, _ in GC.corpus(spec)]:
        try:
            parsed = L.parse_output(t, spec)
            interp = L.interpretation(parsed)
            readers.append({"text": t, "valid": True, "render": L.render(parsed),
                            "interpretation": json.dumps(interp, sort_keys=True, separators=(",", ":"))})
        except L.LanguageError:
            readers.append({"text": t, "valid": False})
    from .goldens import resolver_protocol
    return {"format_version": 1, "record_type": "a27_port_acceptance", "bundle_manifest_sha256":
            hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(), "protocol": resolver_protocol(),
            "records": records, "resolutions": resolutions, "readers": readers}
