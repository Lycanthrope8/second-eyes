"""A2.5 delivery 5: inbox-batch and the measurement report (labelled fake device, fixture goldens, a synthetic pulled
session). Run directly (python grounding/tests/test_quest_measure.py) or as a module. Expectations are written by hand.
"""
from __future__ import annotations

import importlib
import json
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


def main() -> int:
    SB = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring")
    PG = importlib.import_module("grounding.quest.prompt_goldens")
    TD = importlib.import_module("grounding.tests.test_quest_replay_device")
    IB = importlib.import_module("grounding.quest.inbox")
    MS = importlib.import_module("grounding.quest.measure")
    EI = importlib.import_module("grounding.evaluation.iref_vla.protocol").EvaluationInputError

    class Device(TD.FakeDevice):
        def run(self, *args):
            if args[:2] == ("shell", "mv"):
                self.calls.append(args)
                self.files[args[3]] = self.files.pop(args[2])
                return 0, "", ""
            return super().run(*args)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        A = SB.pilot_helpers()
        PG.build_goldens(bundle=SB.scoring_bundle(tmp, A), out=tmp / "g", tokenizer=A.OffsetCharTokenizer(), sizes=(4, 5, 6), index_sha256_prefix=None)
        head = tmp / "g" / "headset"
        rows = [json.loads(x) for x in (head / "goldens.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        n_of = lambda r: len(json.loads((head / r["scene_file"]).read_text(encoding="utf-8"))["objects"])  # noqa: E731
        repo = tmp / "repo"
        TD.make_run(repo, "20261010_A2_r018")
        print("-- inbox-batch (labelled fake device)")
        dev = Device(files={f"{IB.INBOX}/processed/old.json": b"{}"})
        sent = IB.inbox_batch(run_id="20261010_A2_r018", goldens=tmp / "g", snapshot="5", adb=dev, repo=repo, progress=lambda m: None)
        five = [r for r in rows if n_of(r) == 5 and r["kind"] == "dataset_command"]
        check("every golden dataset command of the snapshot is sent, in order, with scene-sized IDs",
              [s["request_id"] for s in sent] == [f"m5-{k:02d}" for k in range(1, len(five) + 1)]
              and [s["request"]["command"] for s in sent] == [json.loads((head / r["command_file"]).read_text(encoding="utf-8")) for r in five]
              and all(f"{IB.INBOX}/{s['request_id']}.json" in dev.files for s in sent))
        try:
            IB.inbox_batch(run_id="20261010_A2_r018", goldens=tmp / "g", snapshot="9", adb=dev, repo=repo, progress=lambda m: None)
            ok = False
        except EI:
            ok = True
        check("an object count with no snapshot is refused", ok)
        print("-- measure-report (a synthetic pulled session)")
        sizes = {}
        for r in rows:
            if r["kind"] == "dataset_command":
                sizes.setdefault(n_of(r), []).append(r)
        size = max(sizes, key=lambda k: len(sizes[k]))
        g4 = (sizes[size] * 3)[:3]   # the fixture's snapshot with the most dataset commands (repeated if it has fewer than three)
        pull = repo / "runs" / "20261010_A2_r018" / "raw" / "outbox" / "20261010-120000"
        (pull / "session").mkdir(parents=True)
        (pull / "session" / "session.json").write_text(json.dumps({"session": "20261010-115900", "purpose": "operational", "cache_mode": "Off",
            "execution_path": "U", "context_tokens": 8192, "flags": 0, "llama_cpp": "b11277 (eae11d22)", "load_ms": 1400.0,
            "memory_kb": {"before_load_rss": 300000, "after_load_rss": 1400000}}), encoding="utf-8")
        (pull / "session" / "startup-log.txt").write_text("load_tensors:   CPU_Mapped model buffer size =   500.79 MiB\n"
            "load_tensors:   CPU_REPACK model buffer size =   500.52 MiB\nllama_kv_cache:        CPU KV buffer size =    96.00 MiB\n"
            "sched_reserve:        CPU compute buffer size =   300.25 MiB\n", encoding="utf-8")
        def line(rid, intake, status, golden=None, path="U", app=None, queued=10.0, start=10.0, end=None, tokens=1600, choice="B", rss=1500000, reason="max_offered_logit"):
            o = {"status": status, "reason": reason, "execution_path": path, "choice_code": choice if status in ("completed", "ask") else None,
                 "tokens": tokens, "prompt_sha256": golden["prompt_sha256"] if golden else None, "source": intake,
                 "ms": {"build": 2.0, "tokenize": 5.0, "eval": (end - start) if end is not None and start is not None else 0.0, "score": 1.0, "total": 0}}
            return {"request_id": rid, "outcome": o, "times": {"intake": intake, "detected": 0.0, "queued": queued, "inference_start": start,
                    "inference_end": end, "outcome": app, "app_observed_ms": app}, "memory_kb": {"rss": rss, "peak": rss + 1000}}
        lines = [line("m4-01", "adb_inbox", "completed", g4[0], app=11000.0, queued=5.0, start=5.0, end=11000.0),
                 line("m4-02", "adb_inbox", "completed", g4[1], app=23000.0, queued=6.0, start=11006.0, end=23000.0, rss=1520000),
                 line("s2", "adb_inbox", "refused", None, path="none", app=1.0, start=None, reason="stale_scene_binding"),
                 line("preset-001", "preset", "ask", g4[2], app=12000.0, queued=1.0, start=1.0, end=12000.0, choice="K"),
                 line("preset-002", "preset", "cancelled", g4[0], app=20000.0, queued=2.0, start=8000.0, end=20000.0, reason="cancelled_during_evaluation")]
        (pull / "session" / "outcomes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
        refdir = tmp / "refs"
        refdir.mkdir()
        refs = {}
        for g, c in ((g4[0], "B"), (g4[1], "A"), (g4[2], "K")):
            refs.setdefault(g["request_id"], c)
        (refdir / "references.jsonl").write_text("".join(json.dumps({"request_id": k, "choice_code": c}) + "\n" for k, c in refs.items()),
                                                 encoding="utf-8")
        expect_same = sum(refs[g["request_id"]] == c for g, c in ((g4[0], "B"), (g4[1], "B"), (g4[2], "K")))
        s = MS.measure_report(run_id="20261010_A2_r018", goldens=tmp / "g", references=refdir, repo=repo)
        gi, gp = s["groups"][f"{size} objects, adb_inbox"], s["groups"][f"{size} objects, preset"]
        check(f"inbox requests on the {size}-object scene: two evaluated, median app-observed 17.0 s, max 23.0 s",
              gi["answered_after_evaluation"] == 2 and gi["app_observed_ms"]["median"] == 17000.0 and gi["app_observed_ms"]["max"] == 23000.0, str(gi["app_observed_ms"]))
        check("queue wait and inference are the stage differences on the headset clock (queue 0 and 11000 ms; inference 10995 and 11994 ms)",
              gi["queue_ms"]["max"] == 11000.0 and gi["inference_ms"]["min"] == 10995.0 and gi["inference_ms"]["max"] == 11994.0)
        check("a stale refusal is counted, without evaluation, under its scene from the inbox receipt or as unknown",
              any(g["statuses"].get("refused") == 1 for g in s["groups"].values()))
        check("presets are a separate intake: an ASK evaluated, a cancellation counted but not timed as an answer",
              gp["statuses"] == {"ask": 1, "cancelled": 1} and gp["answered_after_evaluation"] == 1 and gp["app_observed_ms"]["median"] == 12000.0)
        check("llama.cpp's own buffer report is parsed in MiB",
              s["memory"]["runtime_buffers_mib"] == {"CPU_Mapped model": 500.79, "CPU_REPACK model": 500.52, "CPU KV": 96.0, "CPU compute": 300.25})
        check("resident memory after requests: max 1,520,000 kB; around the load as the session recorded",
              s["memory"]["after_requests_rss_kb"]["max"] == 1520000 and s["memory"]["around_load_kb"]["after_load_rss"] == 1400000)
        check("descriptive agreement over the three answered golden commands (the cancellation not compared)",
              s["descriptive_baseline_agreement"]["compared"] == 3 and s["descriptive_baseline_agreement"]["same_choice"] == expect_same,
              str(s["descriptive_baseline_agreement"]))
        rep = (Path(s["folder"]) / "report.md").read_text(encoding="utf-8")
        check("the report states its clock and keeps the PC's times apart", "headset session stopwatch" in rep and "PC clock only" in rep)
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
