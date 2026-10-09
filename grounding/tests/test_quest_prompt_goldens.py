"""A2.5 delivery 2 (D104): the goldens for the headset's prompt builder.

Run directly (python grounding/tests/test_quest_prompt_goldens.py) or as a module. They are built on A2.2d's fixture
bundle (A2.3b's scoring fixtures, labelled tokenizer double) with sizes 4, 5 and 6, which that bundle has. A labelled fake
model stands in for float32 references, and test_quest_replay_device's labelled fake device for the headset. Expectations
are recomputed independently or worked out by hand.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def refused(fn, EI):
    try:
        fn()
    except EI as e:
        return " | ".join(i["message"] for i in e.issues)
    return None


class FakeModel:
    """Labelled double: a 100-value row, A = 2, B = 1, the rest 0; for the first golden B = 2 too (an exact tie)."""

    def __init__(self, first_ids):
        self.first = first_ids

    def forward_last(self, ids):
        row = [0.0] * 100
        row[32], row[33] = 2.0, (2.0 if list(ids) == list(self.first) else 1.0)
        return row

    independent_last = forward_last

    def info(self):
        return {"device": "fake", "dtype": "float32"}


def main() -> int:
    SB = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring")
    PG = importlib.import_module("grounding.quest.prompt_goldens")
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError
    ser = importlib.import_module("grounding.serialization.serializer")
    CH = importlib.import_module("grounding.inference.iref_vla.choices")
    proto = importlib.import_module("grounding.inference.iref_vla.protocol").load_protocol()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        A = SB.pilot_helpers()
        bundle = SB.scoring_bundle(tmp, A)
        tok = A.OffsetCharTokenizer()
        kw = dict(tokenizer=tok, sizes=(4, 5, 6), index_sha256_prefix=None)
        print("-- building the goldens (fixture bundle, labelled tokenizer double)")
        r = PG.build_goldens(bundle=bundle, out=tmp / "g", **kw)
        rows = [json.loads(x) for x in (bundle / "preparation-index.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        mat = [x for x in rows if x["preparation_status"] == "materialized"]
        size = {x["derived_scene_id"]: len(set(x["planned_object_ids"])) for x in mat}
        want = [min((hashlib.sha256(f"{PG.SALT}\n{s}".encode()).hexdigest(), s) for s, n in size.items() if n == k)[1] for k in (4, 5, 6)]
        sel = json.loads((tmp / "g" / "selection.json").read_text(encoding="utf-8"))
        check("one snapshot per size, each the lowest SHA-256(salt + LF + selection ID) of its size (recomputed)",
              [c["selection_id"] for c in sel["chosen"]] == want and [c["candidates"] for c in sel["chosen"]] == [1, 2, 1])
        golds = [json.loads(x) for x in (tmp / "g" / "headset" / "goldens.jsonl").read_text(encoding="utf-8").splitlines()]
        n_dataset = sum(1 for x in mat if x["derived_scene_id"] in want)
        check(f"every command bound to a snapshot ({n_dataset}) plus two written fixtures per snapshot, labelled apart",
              r["counts"]["dataset_commands"] == n_dataset and r["counts"]["written_fixtures"] == 6
              and [g["kind"] for g in golds].count("written_fixture") == 6)
        head = tmp / "g" / "headset"
        cmap = json.loads((bundle / "model_records" / "category-map.json").read_text(encoding="utf-8"))
        ok_doc = ok_map = ok_prompt = ok_tok = ok_keep = True
        for g in golds:
            scene = json.loads((head / g["scene_file"]).read_text(encoding="utf-8"))
            cmd = json.loads((head / g["command_file"]).read_text(encoding="utf-8"))
            doc = ser.serialize(scene, cmd, format="coordinates_v2", relation_config_path=PG.REL, direction_config_path=PG.DIR,
                                category_maps=[cmap]).document
            ok_doc &= doc == g["document"]
            ids = sorted(o["object_id"] for o in scene["objects"])
            ok_map &= g["codes"] == proto["object_codes"][:len(ids)] + [proto["ask_code"]] and g["targets"] == ids + [proto["ask_target"]]
            prompt = CH.build_prompt(proto, CH.choice_mapping(ids, proto), doc)
            ok_prompt &= hashlib.sha256(prompt.encode("utf-8")).hexdigest() == g["prompt_sha256"]
            ok_tok &= g["token_ids"] == list(tok.encode(prompt))
            ok_keep &= g["keep_before_last_token"] == g["input_tokens"] - 1 and 0 < g["keep_before_command_line"] < g["input_tokens"] - 1
        check("every document is the serializer's, rebuilt from the shipped records", ok_doc)
        check("every mapping is the serialized order: letters over sorted object IDs, K for ASK last", ok_map)
        check("every prompt is build_prompt's, byte for byte; every token list is the tokenizer's", ok_prompt and ok_tok)
        check("cache boundaries: n - 1 before the last token, and the command line strictly earlier", ok_keep)
        wt = sorted(json.loads((head / g["command_file"]).read_text(encoding="utf-8"))["text"] for g in golds if g["kind"] == "written_fixture")
        check("the written fixtures carry the written texts, with suffixed command IDs",
              wt == sorted([t for _, t in PG.WRITTEN] * 3) and all(g["command_id"].endswith(("-w1", "-w2")) for g in golds if g["kind"] == "written_fixture"))
        check("dataset command files are the bundle's bytes",
              all((head / g["command_file"]).read_bytes() == (bundle / "model_records" / "commands" / f"{g['command_id']}.json").read_bytes()
                  for g in golds if g["kind"] == "dataset_command"))
        asset = json.loads((head / "prompt-asset.json").read_text(encoding="utf-8"))
        check("the asset's static lines open every document; its wrapper, message and codes are the protocol's",
              all(g["document"].startswith(asset["header_line"] + asset["semantics_line"]) for g in golds)
              and asset["system_message"] == proto["system_message"] and asset["object_codes"] == proto["object_codes"]
              and asset["configs"]["grounding/relations/relations.v1.json"] == hashlib.sha256(PG.REL.read_bytes()).hexdigest())
        print("-- readback")
        check("the goldens read back clean, with the tokenizer", PG.verify_goldens(tmp / "g", tokenizer=tok) == [])

        def copy(name):
            shutil.copytree(tmp / "g", tmp / name)
            return tmp / name

        def repin(f):
            gm = json.loads((f / "headset" / "golden-manifest.json").read_text(encoding="utf-8"))
            gm["files"] = {n: hashlib.sha256((f / "headset" / n).read_bytes()).hexdigest() for n in gm["files"]}
            gm["prompt_asset_sha256"] = gm["files"]["prompt-asset.json"]
            (f / "headset" / "golden-manifest.json").write_text(json.dumps(gm), encoding="utf-8")
            root = json.loads((f / "manifest.json").read_text(encoding="utf-8"))
            root["files"] = {n: hashlib.sha256((f / n).read_bytes()).hexdigest() for n in root["files"]}
            (f / "manifest.json").write_text(json.dumps(root), encoding="utf-8")

        t = copy("t1")
        p = head.relative_to(tmp / "g") / golds[0]["scene_file"]
        (t / p).write_bytes((t / p).read_bytes().replace(b'"object_id"', b'"object_id" ', 1))
        check("a changed snapshot byte is reported", any("missing or changed" in x for x in PG.verify_goldens(t)))
        t = copy("t2")
        lines = (t / "headset" / "goldens.jsonl").read_text(encoding="utf-8").splitlines()
        g0 = json.loads(lines[0])
        g0["document"] = g0["document"].replace('"command"', '"Command"')
        lines[0] = json.dumps(g0)
        (t / "headset" / "goldens.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        repin(t)
        check("a changed golden document, even re-pinned, does not rebuild", any("does not rebuild" in x for x in PG.verify_goldens(t)))
        t = copy("t3")
        a3 = json.loads((t / "headset" / "prompt-asset.json").read_text(encoding="utf-8"))
        a3["header_line"] = a3["header_line"].replace("{", "{ ", 1)
        (t / "headset" / "prompt-asset.json").write_text(json.dumps(a3), encoding="utf-8")
        repin(t)
        check("an asset whose static line differs, even re-pinned, is reported", any("static lines" in x for x in PG.verify_goldens(t)))
        print("-- refusals")
        broken = tmp / "broken"
        shutil.copytree(bundle, broken)
        f = sorted((broken / "model_records" / "scenes").iterdir())[0]
        f.write_bytes(f.read_bytes() + b" ")
        check("a bundle that does not read back", (refused(lambda: PG.build_goldens(bundle=broken, out=tmp / "x1", **kw), EI) or "").find("does not read back") >= 0)
        check("an index that is not the accepted one",
              (refused(lambda: PG.build_goldens(bundle=bundle, out=tmp / "x2", **dict(kw, index_sha256_prefix="deadbeef")), EI) or "").find("not the accepted") >= 0)
        check("a size the bundle does not have",
              (refused(lambda: PG.build_goldens(bundle=bundle, out=tmp / "x3", **dict(kw, sizes=(4, 9))), EI) or "").find("9 objects") >= 0)
        check("an existing destination", refused(lambda: PG.build_goldens(bundle=bundle, out=tmp / "g", **kw), EI) is not None)
        print("-- float32 references (labelled fake model)")
        m = PG.golden_references(goldens=tmp / "g", out=tmp / "ref", model_loader=lambda d, dev, pr: FakeModel(golds[0]["token_ids"]))
        refs = [json.loads(x) for x in (tmp / "ref" / "references.jsonl").read_text(encoding="utf-8").splitlines()]
        r1 = refs[1]
        k = len(r1["scores"])
        lse = math.log(math.exp(2) + math.exp(1) + 98)
        share_a = math.exp(2) / (math.exp(2) + math.exp(1) + (k - 2))
        a_score = next(s for s in r1["scores"] if s["code"] == "A")
        check("a reference per golden; A wins by a margin of 1, its share and log-probability worked by hand",
              len(refs) == len(golds) and r1["choice_code"] == "A" and r1["margin"] == 1.0
              and abs(a_score["restricted_share"] - share_a) < 1e-12 and abs(a_score["log_prob"] - (2 - lse)) < 1e-12)
        check("an exact tie at the top goes to K, with both codes listed",
              refs[0]["choice_code"] == "K" and refs[0]["tied_codes"] == ["A", "B"] and refs[0]["selection_reason"] == "exact_score_tie")
        c = m["canary"]
        check("the manifest pins the goldens and records the accepted canary with its offered values",
              m["goldens_manifest_sha256"] == hashlib.sha256((tmp / "g" / "manifest.json").read_bytes()).hexdigest()
              and c["decision"] == "accepted" and c["record"]["within_tolerance"] and c["record"]["max_abs_difference"] == 0.0
              and c["record"]["production_choice"] == c["record"]["independent_choice"] == "K"
              and len(c["record"]["production_offered_logits"]) == len(golds[0]["codes"]))

        class Variant(FakeModel):
            """Labelled double: the independent path (or both) altered as the case says."""
            def __init__(self, first_ids, independent=None, production=None):
                super().__init__(first_ids)
                self._ind, self._prod = independent, production

            def forward_last(self, ids):
                row = FakeModel.forward_last(self, ids)
                return self._prod(row) if self._prod else row

            def independent_last(self, ids):
                row = FakeModel.forward_last(self, ids)
                return self._ind(row) if self._ind else row

        def lifted(row, k=33, by=10.0):
            row = list(row)
            row[k] += by
            return row

        first = golds[0]["token_ids"]
        cases = [("another choice, 10 apart", dict(independent=lambda r: lifted(r)), "outside tolerance or another choice"),
                 ("a non-finite production output", dict(production=lambda r: [float("nan")] + list(r[1:])), "non-finite"),
                 ("an independent row shorter than the production row", dict(independent=lambda r: list(r[:50])), "differ in length"),
                 ("an empty row", dict(production=lambda r: []), "empty")]
        for k, (label, kw2, why) in enumerate(cases):
            dest = tmp / f"ref-bad{k}"
            msg = refused(lambda: PG.golden_references(goldens=tmp / "g", out=dest,
                                                       model_loader=lambda d, dev, pr: Variant(first, **kw2)), EI) or ""
            fail = json.loads((tmp / f"ref-bad{k}.failed" / "failure.json").read_text(encoding="utf-8")) \
                if (tmp / f"ref-bad{k}.failed").is_dir() else {}
            check(f"the canary refuses {label}: nothing published, diagnostics kept",
                  why in msg and not dest.exists() and fail.get("canary", {}).get("decision") == "refused", msg[:120])
        fail = json.loads((tmp / "ref-bad0.failed" / "failure.json").read_text(encoding="utf-8"))["canary"]["record"]
        check("the refused canary keeps both paths' offered values and choices (K against B, 10 apart)",
              fail["production_choice"] == "K" and fail["independent_choice"] == "B" and fail["max_abs_difference"] == 10.0
              and not fail["accepted"] and len(fail["independent_offered_logits"]) == len(golds[0]["codes"]))
        m2 = PG.golden_references(goldens=tmp / "g", out=tmp / "ref-near",
                                  model_loader=lambda d, dev, pr: Variant(first, independent=lambda r: lifted(r, k=34, by=1e-6)))
        check("a difference within atol + rtol on a code below the top is accepted",
              m2["canary"]["decision"] == "accepted" and 0 < m2["canary"]["record"]["max_abs_difference"] <= 2e-5)
        print("-- the device helpers (labelled fake device)")
        TD = importlib.import_module("grounding.tests.test_quest_replay_device")
        GD = importlib.import_module("grounding.quest.golden_device")
        repo = tmp / "repo"
        TD.make_run(repo, "20261009_A2_r014")
        dev = TD.FakeDevice()
        rc = GD.push_goldens(goldens=tmp / "g", run_id="20261009_A2_r014", adb=dev, repo=repo, progress=lambda m: None)
        pushes = [c[2] for c in dev.calls if c[0] == "push"]
        check("every headset file reaches the app's prompting/goldens, the golden manifest last",
              len(pushes) == len(json.loads((head / "golden-manifest.json").read_text())["files"]) + 1
              and pushes[-1].endswith("/golden-manifest.json") and all(dev.files[x] == (head / x.split("/prompting/goldens/")[1]).read_bytes() for x in pushes))
        check("the push receipt sits in the run's raw/goldens", Path(rc["receipt"]).parent == repo / "runs" / "20261009_A2_r014" / "raw" / "goldens")
        base = f"{GD.REMOTE_RESULTS}/20261009-100000"
        log = TD.APP + "/logs/s1.jsonl"
        dev.files.update({f"{base}/identity.json": json.dumps({"event_log": log}).encode(),
                          f"{base}/selfchecks.json": b'{"all_passed":true}\n',
                          f"{base}/results.jsonl": b'{"all_ok":true}\n{"all_ok":true}\n',
                          f"{base}/done.json": json.dumps({"goldens": 2, "checked": 2, "all_ok": 2, "total_ms": 1}).encode(),
                          log: b"{}\n"})
        rp = GD.pull_goldens(run_id="20261009_A2_r014", adb=dev, repo=repo, progress=lambda m: None)
        check("the newest finished results folder and its event log arrive, the receipt complete",
              rp["complete"] and rp["checks"]["lines_match_done"] and (Path(rp["folder"]) / "events.jsonl").is_file())
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
