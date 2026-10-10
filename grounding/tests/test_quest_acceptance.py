"""A2.5 correction acceptance: the raw-record checks (a synthetic run, fixture goldens). Run directly
(python grounding/tests/test_quest_acceptance.py) or as a module. Expectations are written by hand.
"""
from __future__ import annotations

import hashlib
import importlib
import json
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


def make_run(repo, goldens, variant):
    TD = importlib.import_module("grounding.tests.test_quest_replay_device")
    run = "20261010_A2_r023"
    TD.make_run(repo, run)
    f = repo / "runs" / run / "raw"
    head = goldens / "headset"
    rows = [json.loads(x) for x in (head / "goldens.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    data = [r for r in rows if r["kind"] == "dataset_command"]
    g1, g2 = data[0], next(r for r in data if r["snapshot_id"] != data[0]["snapshot_id"])
    S1, S2 = "20261010-120000", "20261010-121000"
    src = f / "outbox" / "20261010-122000"
    logs = {S1: [], S2: []}
    t = [100.0]

    def res(rid, sess, status, reason, path, g=None, target=None, kept=0):
        t[0] += 1000
        o = {"status": status, "reason": reason, "execution_path": path, "target_object_id": target, "ask_basis": None,
             "kept_tokens": kept, "purpose": "operational", "cache_mode": "Off",
             "prompt_sha256": g["prompt_sha256"] if g and path == "U" else None, "mapping_sha256": g["mapping_sha256"] if g and path == "U" else None}
        line = {"record_type": "a25_interactive_result", "request_id": rid, "session": sess, "scene_epoch": 1, "outcome": o, "times": {"outcome": t[0]}}
        logs[sess].append(line)
        if not rid.startswith("preset-"):
            (src / "outbox").mkdir(parents=True, exist_ok=True)
            (src / "outbox" / f"{rid}.ack.json").write_text(json.dumps({"request_id": rid, "session": sess, "status": "accepted"}), encoding="utf-8")
            (src / "outbox" / f"{rid}.result.json").write_text(json.dumps(line), encoding="utf-8")
            (f / "inbox").mkdir(parents=True, exist_ok=True)
            (f / "inbox" / f"send-{rid}.json").write_text(json.dumps({"request_id": rid, "request": {
                "command": json.loads((head / (g or g1)["command_file"]).read_text(encoding="utf-8"))}}), encoding="utf-8")
        return line

    res("r023-A1", S1, "completed", "max_offered_logit", "U", g1, target="obj_1")
    res("preset-" + S1 + "-001", S1, "completed", "max_offered_logit", "U", g1, target="obj_1")
    if variant in ("late_b", "late_b_then_repeat"):
        res("r023-B1", S1, "completed", "max_offered_logit", "U", g1, target="obj_1")
        res("r023-B2", S1, "refused", "stale_result_scene_changed", "U", g1)
    else:
        res("r023-B1", S1, "refused", "stale_result_scene_changed", "U", g1, target="obj_9" if variant == "refusal_with_target" else None)
        res("r023-B2", S1, "refused", "scene_changed_before_evaluation", "none", g1)
    if variant == "good_b_then_extra":   # r023: the first attempt met the condition; an unneeded retry did not
        res("r023-B1-2", S1, "refused", "stale_result_scene_changed", "U", g1)
        res("r023-B2-2", S1, "refused", "stale_scene_binding", "none", g1)
    if variant == "late_b_then_repeat":
        res("r023-B1-2", S1, "refused", "stale_result_scene_changed", "U", g1)
        res("r023-B2-2", S1, "refused", "scene_changed_before_evaluation", "none", g1)
    res("r023-C2", S1, "cancelled", "session_ended", "none", g2)
    res("preset-" + S1 + "-002", S1, "cancelled", "session_ended", "none", g2)
    res("r023-C1", S1, "cancelled", "cancelled_during_evaluation", "U", g2)
    res("r023-D1", S2, "completed", "max_offered_logit", "U", g1, target="obj_1")
    for k, s in enumerate((S1, S2)):
        d = src / ("sessions/" + s if s == S1 else "session")
        d.mkdir(parents=True, exist_ok=True)
        log = logs[s] if not (variant == "missing_from_log" and s == S1) else [x for x in logs[s] if x["request_id"] != "r023-A1"]
        (d / "outcomes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log), encoding="utf-8")
        (d / "session.json").write_text(json.dumps({"session": s, "startup_check_ok": True, "goldens_equal": 32, "prompt_self_checks_passed": 5,
                                                    "event_log": "/sdcard/x/events.jsonl"}), encoding="utf-8")
        clean = not (variant == "unclean_end" and s == S1)
        (d / "session-end.json").write_text(json.dumps({"session": s, "clean": clean, "answered": len(log), "written_ms": t[0] + 10}), encoding="utf-8")
        (d / "events.jsonl").write_text(json.dumps({"seq": 10 + 5 * k, "ev": "interactive.startup_check", "data": {"session": s}}) + "\n"
                                        + json.dumps({"seq": 11 + 5 * k, "ev": "interactive.session.start", "data": {"session": s}}) + "\n", encoding="utf-8")
    files = {p.relative_to(src).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(src.rglob("*")) if p.is_file()}
    if variant == "tampered":
        files["outbox/r023-A1.result.json"] = "0" * 64
    (src / "pull.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    idd = f / "identity" / "20261010-115000"
    idd.mkdir(parents=True)
    (idd / "identity.json").write_text(json.dumps({"status": {"current_identity": "verified", "a18c_continuity": "linked"}}), encoding="utf-8")
    (idd / "manifest.json").write_text(json.dumps({"files": {"identity.json": hashlib.sha256((idd / "identity.json").read_bytes()).hexdigest()}}), encoding="utf-8")
    (f / "source").mkdir()
    (f / "source" / "commit.txt").write_text("abc1234\n", encoding="utf-8")
    (f / "operator").mkdir()
    (f / "operator" / "checklist.json").write_text(json.dumps({"B": "yes", "D": "yes", "recorded_utc": "2026-10-10T12:30:00Z"}), encoding="utf-8")
    (f / "operator" / "unity.json").write_text(json.dumps({"unity_console_zero_errors": "yes"}), encoding="utf-8")
    return run


def main() -> int:
    SB = importlib.import_module("grounding.tests.test_iref_vla_pilot_scoring")
    PG = importlib.import_module("grounding.quest.prompt_goldens")
    AC = importlib.import_module("grounding.quest.acceptance")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        A = SB.pilot_helpers()
        PG.build_goldens(bundle=SB.scoring_bundle(tmp, A), out=tmp / "g", tokenizer=A.OffsetCharTokenizer(), sizes=(4, 5, 6), index_sha256_prefix=None)
        results = {}
        for v in ("good", "late_b", "late_b_then_repeat", "good_b_then_extra", "missing_from_log", "refusal_with_target", "unclean_end", "tampered"):
            repo = tmp / f"repo-{v}"
            run = make_run(repo, tmp / "g", v)
            results[v] = AC.check_acceptance(run_id=run, goldens=tmp / "g", repo=repo)
        R = lambda v, word: [r for r in results[v]["rows"] if word in r["requirement"]]  # noqa: E731
        g = results["good"]
        check("everything as intended: overall pass, every row passes, worded as pending independent review",
              g["overall"] == "pass" and all(r["result"] == "pass" for r in g["rows"]) and "independent review pending" in g["status_wording"],
              str([(r["requirement"][:40], r["result"], r["limitation"]) for r in g["rows"] if r["result"] != "pass"]))
        check("the table covers both sessions' startup checks and closing records, the four cases, hashes, D102, source and the operator",
              len(R("good", "startup check before serving")) == 2 and len(R("good", "shutdown after final publication")) == 2
              and all(R("good", "case " + c) for c in "ABCD") and R("good", "pull manifest") and R("good", "D102") and R("good", "operator"))
        lb = results["late_b"]
        check("a scene switch that came too late: case B incomplete, not failed; nothing else fails",
              lb["overall"] == "incomplete" and R("late_b", "case B")[0]["result"] == "incomplete" and not any(r["result"] == "fail" for r in lb["rows"]))
        rb = results["late_b_then_repeat"]
        check("case B repeated with fresh IDs: the second attempt meets the condition, the first is listed as additional, overall pass",
              rb["overall"] == "pass" and [r["result"] for r in R("late_b_then_repeat", "case B")] == ["incomplete", "pass"]
              and "additional" in R("late_b_then_repeat", "case B")[0]["requirement"])
        ge = results["good_b_then_extra"]
        check("the first attempt met the condition and an unneeded retry did not: case B passes on attempt 1, the retry is listed as additional",
              ge["overall"] == "pass" and [r["result"] for r in R("good_b_then_extra", "case B")] == ["pass", "incomplete"]
              and "met by attempt 1" in R("good_b_then_extra", "case B")[1]["limitation"], str([(r["result"], r["limitation"]) for r in R("good_b_then_extra", "case B")]))
        check("a result missing from its session log fails the accounting, naming the request",
              results["missing_from_log"]["overall"] == "fail" and "r023-A1" in R("missing_from_log", "every sent inbox request")[0]["limitation"])
        check("a refusal carrying a target fails the outcome rules",
              results["refusal_with_target"]["overall"] == "fail" and R("refusal_with_target", "no target")[0]["result"] == "fail")
        check("an unclean shutdown fails that session's shutdown row", results["unclean_end"]["overall"] == "fail"
              and [r["result"] for r in R("unclean_end", "shutdown after final publication")].count("fail") == 1)
        check("a pull manifest hash that does not read back fails, naming the file",
              results["tampered"]["overall"] == "fail" and "r023-A1.result.json" in R("tampered", "pull manifest")[0]["limitation"])
        check("the acceptance table is written as report.md beside summary.json",
              (Path(g["folder"]) / "report.md").read_text(encoding="utf-8").count("| pass |") == len(g["rows"]))
    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
