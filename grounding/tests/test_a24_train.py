"""Tests of A2.4's training harness, fine-tuned identity and evaluation path (D109), with labelled stand-in models.

    python grounding/tests/test_a24_train.py --samples DIR --official DIR

Stand-ins (labelled): random-weight Qwen2 models with the real vocabulary size and the pinned tokenizer files. Fits use
a 2-layer one (memory); the evaluation path uses a 24-layer one shaped like the 0.5B (2 key-value heads, head size 64),
so the accepted runner's architecture and identity checks run for real. They establish integration only: not
real-model training, decoding performance or deployment correctness, which Stage A's smoke on the RTX PC measures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FAILS, PASSES = [], []
TOKDIR = REPO / "quest-app" / "Assets" / "SecondEyes" / "Models" / "qwen2.5-0.5b-instruct"



def _safe_console():
    """Never crash on printing: a Windows console piped to a file may not encode every character (cp1252)."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("  ok    " if ok else "  FAIL  ") + name + ("" if ok or not detail else f"  [{detail}]"))


def standin(folder: Path, layers: int, seed=0):
    """A labelled stand-in: random Qwen2 weights, the real vocabulary size, the pinned tokenizer files."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    cfg = Qwen2Config(vocab_size=151936, hidden_size=128, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32768,
                      tie_word_embeddings=True, eos_token_id=151645, bos_token_id=151643)
    torch.manual_seed(seed)
    m = Qwen2ForCausalLM(cfg).to(torch.float32)
    m.save_pretrained(str(folder), safe_serialization=True)
    for n in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        shutil.copyfile(TOKDIR / n, folder / n)
    return folder


def a24_fixture(tmp: Path, samples: Path, official: Path):
    TP = importlib.import_module("grounding.tests.test_a24_prep")
    R = importlib.import_module("grounding.adapters.iref_vla.release")
    D = importlib.import_module("grounding.preparation.iref_vla_dev.dev")
    P = importlib.import_module("grounding.a24.prep")
    TP.build_release(samples, tmp / "release")
    vocab = samples / "data" / "NYU_Object_Classes.csv"
    inv = R.build_inventory([("Scannet", tmp / "release" / "Scannet")], R.read_vocabulary(vocab.read_bytes()))
    TP.locked(lambda: R.write_inventory(inv, tmp / "inventory"))
    with zipfile.ZipFile(tmp / "Scannet.zip", "w") as z:
        for f in sorted((tmp / "release").rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(tmp / "release").as_posix())
    a26 = TP.locked(lambda: D.freeze(inventory_dir=tmp / "inventory", official_dir=official, out=tmp / "a26spec"))
    a26p = Path(a26["folder"]) / "spec.json"
    frozen = {"a26_spec_sha256": hashlib.sha256(a26p.read_bytes()).hexdigest(),
              "inventory_sha256": a26["inventory"]["inventory_sha256"]}
    st = json.loads(P.SETTINGS_PATH.read_text(encoding="utf-8"))
    st["cohort"] = {"fitting_environments": 3, "validation_environments": 1, "fitting_cap_per_environment": 4,
                    "validation_cap_per_environment": 3, "fitting_minimum": 13, "validation_minimum": 2,
                    "reserve_batch_environments": 1, "reserve_batches_at_most": 2}
    st["batch"]["accumulation"] = 4
    st["checkpoints"]["resume_state_every_optimizer_steps"] = 1
    st["merge"]["check"].update(direct_rows=4, parser_parents=2)
    sp = tmp / "settings.json"
    sp.write_text(json.dumps(st), encoding="utf-8")
    S = P.freeze(a26_spec=a26p, inventory_dir=tmp / "inventory", out=tmp / "spec", frozen=frozen, settings_path=sp)
    P.prepare(spec=Path(S["folder"]) / "spec.json", a26_spec=a26p, zip_path=tmp / "Scannet.zip", inventory_dir=tmp / "inventory",
              vocabulary=vocab, out=tmp / "prep", work=tmp / "work", settings_path=sp, progress=lambda m: None)
    return tmp / "prep", tmp / "work", sp


def main() -> int:
    _safe_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=os.environ.get("SECOND_EYES_IREF_RELEASE_SAMPLES"))
    ap.add_argument("--official", default=os.environ.get("SECOND_EYES_IREF_OFFICIAL"))
    a = ap.parse_args()
    import torch
    TR = importlib.import_module("grounding.a24.train")
    ID = importlib.import_module("grounding.a24.identity")
    E = importlib.import_module("grounding.a24.evaluate")
    mp, desc = E.model_protocol()
    print("-- units")
    d_ids, d_lab = TR.sequence({"token_ids": [5, 6, 7], "answer_token_id": 33}, "direct")
    p_ids, p_lab = TR.sequence({"prompt_ids": [5, 6], "output_ids": [9, 10, 151645]}, "parser")
    check("response-only labels: the direct answer letter alone; the parser's output and end token alone",
          d_ids == [5, 6, 7, 33] and d_lab == [-100, -100, -100, 33] and p_lab == [-100, -100, 9, 10, 151645])
    k = [f"p{i}" for i in range(40)]
    check("the data order depends only on the seed and the epoch, and differs between them",
          TR.order(k, 1, 0) == TR.order(list(reversed(k)), 1, 0) and TR.order(k, 1, 0) != TR.order(k, 1, 1)
          and TR.order(k, 1, 0) != TR.order(k, 2, 0) and sorted(TR.order(k, 3, 2)) == sorted(k))
    f = TR.lr_lambda(100, 5)
    check("the schedule: linear warmup from 0 over 5 steps, then linear decay to 0 at step 100",
          f(0) == 0 and f(5) == 1 and abs(f(50) - 50 / 95) < 1e-12 and f(100) == 0)
    have = bool(a.samples and a.official)
    check("the pinned samples and official lists are available", have, "pass --samples DIR and --official DIR")
    if not have:
        print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
        return 1
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        prep, work, sp = a24_fixture(tmp, Path(a.samples), Path(a.official))
        importlib.import_module("grounding.tests.test_a24_prep").scans_ok(
            importlib.import_module("grounding.a24.prep"), prep, "the training fixture", check_fn=check)
        base = standin(tmp / "base2", layers=2)
        ev = {"revision": "stand-in", "tie": "labelled stand-in base (test only)", "files": {}}
        bevid = lambda d: (ev, mp, desc)   # noqa: E731
        common = dict(prep=prep, work=work, base_dir=base, device="cpu", settings_path=sp, base_evidence=bevid,
                      progress=lambda m: None)
        print("-- the loss is the response's cross-entropy only")
        rows, _ = TR.load_rows(prep, "fitting", {"design": "direct", "format": "coordinates_v2"})
        r0 = rows[0]
        m = TR.load_base(base, "cpu")
        short = {"token_ids": r0["token_ids"][-300:], "answer_token_id": r0["answer_token_id"]}
        ids, lab = TR.sequence(short, "direct")
        prow, _ = TR.load_rows(prep, "fitting", {"design": "parser", "format": None})
        pids, plab = TR.sequence(prow[0], "parser")
        with torch.no_grad():
            full_d = float(m(input_ids=torch.tensor([ids]), labels=torch.tensor([lab])).loss)
            mine_d = float(TR.example_loss(m, short, "direct", "cpu"))
            full_p = float(m(input_ids=torch.tensor([pids]), labels=torch.tensor([plab])).loss)
            mine_p = float(TR.example_loss(m, prow[0], "parser", "cpu"))
        check("the loss computed at the supervised positions only equals the masked full-sequence loss (direct and parser)",
              abs(full_d - mine_d) < 1e-4 and abs(full_p - mine_p) < 1e-4, f"{full_d} {mine_d} {full_p} {mine_p}")
        print("-- a direct fit, resumed exactly, locked")
        full = TR.fit(config="direct.coordinates_v2", seed_index=1, out=tmp / "fit-a", **common)
        fd = tmp / "fit-a"
        log = [json.loads(x) for x in (fd / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        check("the fit completes its planned steps with finite losses, three checkpoints and three validations",
              full["optimizer_steps"] == full["planned_optimizer_steps"] == len(log) and all(math.isfinite(x["loss"]) for x in log)
              and all((fd / "checkpoints" / f"epoch-{e}").is_dir() and (fd / "validation" / f"epoch-{e}.json").is_file()
                      for e in (1, 2, 3)))
        TR.fit(config="direct.coordinates_v2", seed_index=1, out=tmp / "fit-b", stop_after=3, **common)
        stopped = json.loads((tmp / "fit-b" / "resume" / "state.json").read_text(encoding="utf-8"))["position"]
        TR.fit(config="direct.coordinates_v2", seed_index=1, out=tmp / "fit-b", resume=True, **common)
        same = all((fd / "checkpoints" / "epoch-3" / n).read_bytes() == (tmp / "fit-b" / "checkpoints" / "epoch-3" / n).read_bytes()
                   for n in ("adapter_model.safetensors",))
        vals_same = all(json.loads((fd / "validation" / f"epoch-{e}.json").read_text())["metric"] ==
                        json.loads((tmp / "fit-b" / "validation" / f"epoch-{e}.json").read_text())["metric"] for e in (1, 2, 3))
        check("interrupted at optimizer step 3 and resumed (adapter, optimizer, scheduler, RNG, data position): the final "
              "adapter is byte-identical to the uninterrupted fit's (CPU)", stopped["step"] == 3 and same and vals_same)
        st2 = json.loads(sp.read_text(encoding="utf-8"))
        st2["optimizer"]["lr"] = 0.0003
        sp2 = tmp / "settings-other.json"
        sp2.write_text(json.dumps(st2), encoding="utf-8")
        try:
            TR.fit(config="direct.coordinates_v2", seed_index=1, out=tmp / "fit-b", resume=True,
                   **dict(common, settings_path=sp2))
            refused = False
        except TR.TrainError:
            refused = (tmp / "fit-b" / "done.json").is_file()
        check("a resume under different settings is refused and the existing evidence is kept", refused)
        try:
            TR.fit(config="direct.coordinates_v2", seed_index=1, out=tmp / "fit-a", **common)
            refused = False
        except TR.TrainError:
            refused = True
        check("a fit never writes into an existing folder without resume", refused)
        r1 = TR.fit(config="direct.coordinates_relations_v2", seed_index=1, out=tmp / "fit-r", **common)
        la = [json.loads(x) for x in (fd / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        lr = [json.loads(x) for x in (tmp / "fit-r" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        check("the two direct formats at one seed: the same steps and examples per step (matched exposure), more tokens "
              "in the relations format", [x["examples"] for x in la] == [x["examples"] for x in lr]
              and r1["planned_optimizer_steps"] == full["planned_optimizer_steps"]
              and sum(x["tokens"] for x in lr) > sum(x["tokens"] for x in la))
        print("-- selection, merge, identity")
        sel = TR.select(fd)
        vals = [json.loads((fd / "validation" / f"epoch-{e}.json").read_text())["metric"] for e in (1, 2, 3)]
        best = max(range(3), key=lambda i: (vals[i], -i))
        check("the selection applies the frozen rule (best metric, ties to the earlier checkpoint)",
              sel["checkpoint"] == f"epoch-{best + 1}", f"{vals} -> {sel['checkpoint']}")
        mg = TR.merge(fit_dir=fd, prep=prep, base_dir=base, out=tmp / "merged-a", device="cpu", settings_path=sp,
                      base_evidence=bevid)
        ident = ID.load_identity(tmp / "merged-a", mp)
        check("the merge check passes (adapter-loaded against merged, same checkpoint) and the identity names the model",
              mg["merge"]["check"]["passed"] and ident["model_name"] == "a24.direct.coordinates_v2.s1"
              and ident["selection"]["checkpoint"] == sel["checkpoint"] and ident["adapter"]["files"])
        check("the identity binds the base evidence, data, specification, settings, seed, code and software",
              ident["base"]["evidence_tie"] == ev["tie"] and ident["training"]["data_files"] and ident["seed"] == 20261011
              and "grounding/adapters/iref_vla/release.py" in ident["training"]["code"] and ident["training"]["software"]["peft"])
        bad = tmp / "merged-bad"
        shutil.copytree(tmp / "merged-a", bad)
        (bad / "config.json").write_bytes((bad / "config.json").read_bytes() + b" ")
        extra = tmp / "merged-extra"
        shutil.copytree(tmp / "merged-a", extra)
        (extra / "notes.txt").write_text("x")
        refused = []
        for d in (bad, extra):
            try:
                ID.load_identity(d, mp)
                refused.append(False)
            except Exception:   # noqa: BLE001
                refused.append(True)
        check("a changed file or an extra file in a merged folder is refused", refused == [True, True])
        print("-- the parser: fit, constrained validation, merge")
        rp = TR.fit(config="parser", seed_index=1, out=tmp / "fit-p", **common)
        vp = json.loads((tmp / "fit-p" / "validation" / "epoch-1.json").read_text(encoding="utf-8"))
        check("the parser's validation decodes once per parent and resolves each retained view; every decode ends in a "
              "declared outcome, and the reader agrees with the decoder",
              rp["optimizer_steps"] == rp["planned_optimizer_steps"] and vp["rows"]
              and all(r["status"] in ("completed", "length_exhausted", "empty_allowed_set", "premature_eos") for r in vp["rows"])
              and all((r["status"] == "completed") == r["valid"] for r in vp["rows"]))
        TR.select(tmp / "fit-p")
        TR.merge(fit_dir=tmp / "fit-p", prep=prep, base_dir=base, out=tmp / "merged-p", device="cpu", settings_path=sp,
                 base_evidence=bevid)
        TR.select(tmp / "fit-r")
        TR.merge(fit_dir=tmp / "fit-r", prep=prep, base_dir=base, out=tmp / "merged-r", device="cpu", settings_path=sp,
                 base_evidence=bevid)
        smokes = {}
        for c, m_ in (("direct.coordinates_v2", "merged-a"), ("direct.coordinates_relations_v2", "merged-r"), ("parser", "merged-p")):
            p_ = tmp / f"runner-smoke-{c}.json"
            name = ID.load_identity(tmp / m_)["model_name"]
            p_.write_text(json.dumps({"accepted": True, "checkpoint_tie": f"A2.4 fine-tuned local artifact {name}: labelled"}))
            smokes[c] = p_
        cont = TR.continuation(fits={"direct.coordinates_v2": fd, "direct.coordinates_relations_v2": tmp / "fit-r",
                                     "parser": tmp / "fit-p"},
                               merged={"direct.coordinates_v2": tmp / "merged-a", "direct.coordinates_relations_v2": tmp / "merged-r",
                                       "parser": tmp / "merged-p"}, runner_smokes=smokes, settings_path=sp)
        smokes["parser"].write_text(json.dumps({"accepted": True, "checkpoint_tie": "base checkpoint"}))
        cont2 = TR.continuation(fits={"direct.coordinates_v2": fd, "direct.coordinates_relations_v2": tmp / "fit-r",
                                      "parser": tmp / "fit-p"},
                                merged={"direct.coordinates_v2": tmp / "merged-a", "direct.coordinates_relations_v2": tmp / "merged-r",
                                        "parser": tmp / "merged-p"}, runner_smokes=smokes, settings_path=sp)
        check("the continuation check passes on complete fits, and fails when a runner smoke did not use the fine-tuned "
              "identity", cont["passed"] and not cont2["passed"], json.dumps(cont2["configurations"])[:300])
        print("-- the Stage A smoke command (stand-in, CPU)")
        sr = TR.smoke(prep=prep, work=work, base_dir=base, out=tmp / "smoke", device="cpu", settings_path=sp,
                      base_evidence=bevid, progress=lambda m: None)
        cs = sr["configurations"]
        check("the smoke fits every configuration (the longest example first), resumes exactly on the CPU, merges, names "
              "its models a24.smoke.*, and extrapolates Stage B",
              set(cs) == {"direct.coordinates_v2", "direct.coordinates_relations_v2", "parser"}
              and all(x["resume_max_abs_adapter_difference"] == 0.0 and x["merge_check"]["passed"]
                      and x["model_name"].startswith("a24.smoke.") and len(x["steps"]) == 3 for x in cs.values())
              and all(x["steps"][0]["tokens"] >= x["longest_example_tokens"] for x in cs.values())
              and sr["stage_b"]["total_hours"] > 0 and "decode" in cs["parser"]
              and (tmp / "smoke" / "smoke-report.md").is_file(), json.dumps({k: v.get("model_name") for k, v in cs.items()}))
        print("-- the evaluation path through A2.6c's preflight-bound runner (24-layer stand-in)")
        TB = importlib.import_module("grounding.tests.test_iref_vla_baseline")
        PF = importlib.import_module("grounding.inference.iref_vla_dev.preflight")
        RN = importlib.import_module("grounding.inference.iref_vla_dev.runs")
        T = importlib.import_module("grounding.preparation.iref_vla.tokens")
        TP = importlib.import_module("grounding.tests.test_a24_prep")
        p26, w26, frozen = TP.locked(lambda k: TB.prepared_fixture(tmp / f"a26-{k}", Path(a.samples), Path(a.official)),
                                     fresh=True)
        tok = T.load_pinned_tokenizer(TOKDIR)
        toks = {k: tok for k in PF.MODEL_KEYS}
        pre = PF.preflight(prep=p26, work=w26, tokenizers=toks, frozen=frozen, progress=lambda m: None)
        pdir = TP.locked(lambda k: PF.write_preflight(pre, tmp / f"preflight-{k}"), fresh=True)
        noref = tmp / "a26-noref"
        shutil.copytree(p26, noref)
        shutil.rmtree(noref / "reference_only")
        big = standin(tmp / "merged24", layers=24, seed=1)
        TM = importlib.import_module("grounding.inference.iref_vla.model").TorchModel
        cpu_loader = lambda d, dev, proto: TM.load(d, "cpu", proto)   # noqa: E731 - the real loader, on CPU (test only)
        for i, (cfgname, design, fmt) in enumerate((("direct.coordinates_v2", "direct", "coordinates_v2"),
                                                     ("parser", "parser", None))):
            d = tmp / f"m24-{i}"
            shutil.copytree(big, d)
            ID.write_identity(d, {"model_name": f"a24.{cfgname}.s1", "configuration": {"name": cfgname, "design": design,
                                                                                      "format": fmt},
                                  "seed": 20261011, "seed_index": 1,
                                  "base": {"hf_repo": "Qwen/Qwen2.5-0.5B-Instruct", "hf_revision": mp["model"]["hf_revision"],
                                           "evidence_tie": "labelled stand-in (test only)"},
                                  "selection": {"checkpoint": "epoch-1"}, "training": {}, "adapter": {}, "merge": {}})
        dv = E.direct(prep=noref, work=w26, preflight=pdir, model_dir=tmp / "m24-0", out=tmp / "eval-d", device="cpu",
                      frozen=frozen, tokenizers=toks, model_loader=cpu_loader, progress=lambda m: None)
        sm = json.loads((tmp / "eval-d" / "smoke.json").read_text(encoding="utf-8"))
        mj = json.loads((tmp / "eval-d" / "model.json").read_text(encoding="utf-8"))
        nreq = len(TB.__dict__.get("json", json).loads("[]")) + sum(1 for _ in (p26 / "requests-index.jsonl").open())
        res = [json.loads(x) for x in (tmp / "eval-d" / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        check("the accepted runner's smoke and run accept the merged model through the fine-tuned identity: the tie names "
              "the model, every frozen request has a row, and model.json binds the run to the model",
              sm["accepted"] and "a24.direct.coordinates_v2.s1" in sm["checkpoint_tie"] and len(res) == nreq
              and mj["run"]["manifest_sha256"] and mj["primary_format"] == "coordinates_v2", sm.get("checkpoint_tie", "")[:200])
        try:
            E.direct(prep=noref, work=w26, preflight=pdir, model_dir=tmp / "m24-1", out=tmp / "eval-x", device="cpu",
                     frozen=frozen, tokenizers=toks, model_loader=cpu_loader, progress=lambda m: None)
            refused = False
        except Exception:   # noqa: BLE001
            refused = True
        ps = E.direct(prep=noref, work=w26, preflight=pdir, model_dir=tmp / "m24-1", out=tmp / "eval-ps", device="cpu",
                      frozen=frozen, tokenizers=toks, model_loader=cpu_loader, smoke_only=True, progress=lambda m: None)
        check("a parser model is refused a direct run, and takes the runner's smoke only (a mechanical identity-path check)",
              refused and "a24.parser.s1" in ps["smoke"]["checkpoint_tie"] and not (tmp / "eval-ps" / "run").exists())
        pv = E.parser(prep=p26, work=w26, model_dir=tmp / "m24-1", out=tmp / "eval-p", device="cpu", frozen=frozen,
                      progress=lambda m: None)
        prs = [json.loads(x) for x in (tmp / "eval-p" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        parents = len({json.loads(x)["parent_command_id"] for x in (p26 / "requests-index.jsonl").read_text().splitlines()})
        check("the parser evaluation decodes once per parent and resolves both views from that one decode",
              pv["parents"] == parents == len(prs) and all(set(r["views"]) == set(E.VIEWS) for r in prs))
        rules = RN.run_rules(prep=noref, work=w26, out=tmp / "rules", frozen=frozen)
        del rules
        sc = E.score(prep=p26, rules=tmp / "rules", base_runs={}, direct_evals=[tmp / "eval-d"], parser_evals=[tmp / "eval-p"],
                     out=tmp / "scores", frozen=frozen)
        an = E.analyze(prep=p26, scores=tmp / "scores", out=tmp / "analysis", b=50)
        dev = an["partitions"]["development"]
        names = set(dev["systems"])
        check("scoring and analysis: the fine-tuned systems on their own format, the cross-format rows labelled, the parser "
              "per view, rules and always-B; paired differences and the supplementary breakdown",
              any(n.startswith("a24.direct.coordinates_v2.s1 | full_inventory | coordinates_v2") for n in names)
              and any("[cross_format]" in n for n in names) and any(n.startswith("a24.parser.s1") for n in names)
              and dev["differences"] and dev["supplementary"] and "a24.parser.s1" in sc["parsers"],
              str(sorted(names))[:300])
        print("-- the closure's report corrections")
        lines = (tmp / "analysis" / "report.md").read_text(encoding="utf-8").split("\n")
        tables, i = [], 0
        while i < len(lines):
            if lines[i].startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= set("-: "):
                cols, j = lines[i].count("|"), i + 2
                while j < len(lines) and lines[j].startswith("|"):
                    tables.append(lines[j].count("|") == cols)
                    j += 1
                i = j
            else:
                i += 1
        sys_head = next(x for x in lines if x.startswith("| System |"))
        check("every Markdown table in the report has a constant number of columns; the Systems table gives system, view "
              "and format their own cells", tables and all(tables)
              and sys_head == "| System | View | Format | Command-weighted | Environment-weighted | Parents |")
        an2 = E.analyze(prep=p26, scores=tmp / "scores", out=tmp / "analysis-2", b=50, supersedes=tmp / "analysis")
        ch = an2["supersedes"]["changes"]
        check("an analysis regenerated from the same saved scores records what it supersedes, and with the same code nothing "
              "changes: no interval endpoint, no zero crossing, no other number",
              an2["supersedes"]["analysis_sha256"] == hashlib.sha256((tmp / "analysis" / "analysis.json").read_bytes()).hexdigest()
              and ch["intervals_compared"] > 0 and ch["endpoints_changed"] == 0 and not ch["zero_crossing_changes"]
              and not ch["other_numbers_changed"] and ch["only_in_old"] == ch["only_in_new"] == 0, json.dumps(ch)[:300])
        bumped = json.loads((tmp / "analysis" / "analysis.json").read_text(encoding="utf-8"))
        d0 = next(iter(bumped["partitions"]["development"]["differences"].values()))["command_weighted"]
        lo, hi = d0["interval"]["low"], d0["interval"]["high"]
        d0["interval"]["low"], d0["interval"]["high"] = (0.0002, 0.0005) if lo <= 0 <= hi else (-0.0002, 0.0005)
        (tmp / "bumped.json").write_text(json.dumps(bumped), encoding="utf-8")
        dd = E.analysis_diff(tmp / "analysis" / "analysis.json", tmp / "bumped.json")
        check("analysis-diff reports each endpoint's change in percentage points and an interval whose zero crossing changed",
              dd["endpoints_changed"] >= 1 and len(dd["zero_crossing_changes"]) == 1 and dd["max_endpoint_change_pp"] > 0
              and not dd["other_numbers_changed"], json.dumps(dd)[:300])
    importlib.import_module("grounding.tests.test_a24_prep").report_retries()
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
