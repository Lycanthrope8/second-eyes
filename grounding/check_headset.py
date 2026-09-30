#!/usr/bin/env python3
"""Compare the headset's answers in a run's event log with the PC reference (A1.7c-2, D43).

    python grounding/check_headset.py <run ID> --reference 20260928_A1_r008

Reads every log in runs/<run ID>/raw/ (on the headset, pull them first with tools/logs.py) and the reference
files runs/<reference>/raw/reference_<prompt id>_<variant>.json that grounding/reference.py wrote. Each send is
model.request, then one model.token per answer piece, then model.generate at the end, all with the same request
number. For each send of a prompt file (the fixed prompt or a preset, D49) whose PC reference the reference run
holds, it checks:
  prompt  the token IDs equal the reference's, so the template, the system message and the tokenizer are the
          same as on the PC; this works even if the answer never finished
  answer  the text equals the 16-bit-weights reference (fp16w) as Meta's runner streams it, token by token, and
          otherwise says at which token it first differs; the full-precision reference (fp32) is shown too
Sends without a reference in that run, and typed sends, are listed without a comparison. With more than 8 sends (a
Repeat run, D50), each send gets one line. At the end, the finished sends' timing is summarized by the steps per
frame each ran with (from model.load and model.setting). The result is PASS, and the
exit code 0, only if at least one send finished with its reference's answer and no send contradicts its reference.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from eventlog import read_events  # noqa: E402

TYPED = "typed"


class ToolError(Exception):
    """A problem the user can fix. Printed without a traceback."""


def load_reference(ref_dir: Path, prompt_id: str, variant: str, required: bool):
    path = ref_dir / f"reference_{prompt_id}_{variant}.json"
    if not path.is_file():
        if required:
            raise ToolError(f"{path.relative_to(ROOT).as_posix()} isn't there; the reference run has no {variant} "
                            f"answer for prompt {prompt_id}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def when(event) -> str:
    utc = event.get("utc_us") if event else None
    return dt.datetime.fromtimestamp(utc / 1e6).strftime("%H:%M:%S") if isinstance(utc, (int, float)) else "?"


def seconds(ms) -> str:
    return f"{ms / 1000:.1f} s" if isinstance(ms, (int, float)) else "?"


def sends_of(log: Path) -> list:
    """One entry per request number in one session log: its request, tokens and end, in log order."""
    sends = {}
    for e in read_events(log):
        ev, d = e.get("ev"), e.get("data", {})
        if ev in ("model.request", "model.token", "model.generate") and isinstance(d.get("request"), int):
            s = sends.setdefault(d["request"], {"log": log.name, "request": None, "tokens": [], "end": None})
            if ev == "model.request":
                s["request"] = e
            elif ev == "model.token":
                s["tokens"].append(e)
            else:
                s["end"] = e
    return [sends[k] for k in sorted(sends)]


def compare_answer(pieces: list, text: str, reference: str) -> str:
    """Where the headset's answer first differs from the reference, by answer token."""
    if text == reference:
        return "same"
    at, done = 0, 0
    while at < min(len(text), len(reference)) and text[at] == reference[at]:
        at += 1
    for index, piece in enumerate(pieces):
        if done + len(piece) > at:
            return f"differs from token {index} ({piece!r}); the reference goes on {reference[at:at + 20]!r}"
        done += len(piece)
    if text and reference.startswith(text):
        return "same so far"
    return f"differs at character {at}; the reference goes on {reference[at:at + 20]!r}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare the headset's answers with the PC reference (A1.7c-2).")
    parser.add_argument("run_id", help="the run whose headset logs to check, e.g. 20260929_A1_r010")
    parser.add_argument("--reference", required=True, help="the run holding the PC reference, e.g. 20260928_A1_r008")
    args = parser.parse_args(argv)
    try:
        raw = ROOT / "runs" / args.run_id / "raw"
        logs = sorted(raw.glob("*.jsonl"))
        if not logs:
            raise ToolError(f"No logs in runs/{args.run_id}/raw/. On the headset, pull them first: "
                            f"python tools/logs.py pull {args.run_id}")
        ref_dir = ROOT / "runs" / args.reference / "raw"
        if not ref_dir.is_dir():
            raise ToolError(f"No run {args.reference} with a raw/ folder in runs/")
        events = [e for log in logs for e in read_events(log)]
        sends = [s for log in logs for s in sends_of(log)]
        steps_at = {}   # (log name, request) -> the steps per frame when it was sent
        for log in logs:
            steps = None
            for e in read_events(log):
                d = e.get("data", {})
                if e.get("ev") in ("model.load", "model.setting"):
                    steps = d.get("steps_per_frame", steps)
                elif e.get("ev") == "model.request" and isinstance(d.get("request"), int):
                    steps_at[(log.name, d["request"])] = steps
        for s in sends:
            s["steps"] = steps_at.get((s["log"], s["request"]["data"].get("request") if s["request"] else None))

        print(f"Run {args.run_id}: {len(logs)} log(s). Reference: {args.reference}.")
        loads = [e for e in events if e.get("ev") == "model.load"]
        print("\nModel loads:" + ("" if loads else " none"))
        for e in loads:
            d = e.get("data", {})
            copy = "copied out of the app first" if d.get("copied") else "already copied"
            print(f"  {when(e)}  {d.get('file')}: {seconds(d.get('ms'))}, {copy}; {d.get('backend')}, "
                  f"{d.get('execution_mode')}, {d.get('steps_per_frame')} steps per frame")

        fixed = [s for s in sends if s["request"] and s["request"]["data"].get("prompt_id") not in (None, TYPED)]
        typed = [s for s in sends if s["request"] and s["request"]["data"].get("prompt_id") == TYPED]
        verdicts, unreferenced = [], []
        brief = len(fixed) > 8
        print("\nSends of prompt files:" + ("" if fixed else " none. Press Send once.")
              + (f" {len(fixed)}, one line each." if brief else ""))
        for n, s in enumerate(fixed, start=1):
            d = s["request"]["data"]
            ref16 = load_reference(ref_dir, d["prompt_id"], "fp16w", required=False)
            if ref16 is None:
                unreferenced.append(d["prompt_id"])
                print(f"  #{n} {d['prompt_id']} at {when(s['request'])}: no 16-bit reference for this prompt in "
                      f"{args.reference}, so not compared")
                continue
            ref32 = load_reference(ref_dir, d["prompt_id"], "fp32", required=False)
            ids, ref_ids = d.get("prompt_token_ids") or [], ref16["prompt"]["token_ids"]
            prompt_ok = ids == ref_ids
            pieces = [t["data"].get("text", "") for t in s["tokens"]]
            end = s["end"]["data"] if s["end"] else None
            text = end.get("answer") if end else "".join(pieces)
            state = "unfinished" if end is None else ("stopped" if end.get("stopped") else "finished")
            answer = compare_answer(pieces, text, ref16["answer"]["text_meta_style"])
            if state == "finished" and answer == "same" and len(pieces) != ref16["answer"]["tokens"]:
                answer = f"same text, but {len(pieces)} tokens instead of {ref16['answer']['tokens']}"
            verdicts.append((prompt_ok, state, answer))
            if brief:
                print(f"  #{n} {d['prompt_id']} at {when(s['request'])}: {state}; prompt "
                      f"{'same' if prompt_ok else 'DIFFERENT'}; answer {answer}; {timing(s)}; steps {s['steps']}")
                continue
            print(f"  #{n} {d['prompt_id']} at {when(s['request'])} ({s['log']}, request {d.get('request')}): {state}")
            print(f"     prompt: {len(ids)} tokens; same IDs as the reference ({len(ref_ids)}): "
                  + ("yes" if prompt_ok else "NO, " + first_difference(ids, ref_ids)))
            print(f"     answer: {text!r} ({count(len(pieces))})")
            fp32 = "" if ref32 is None else ("; the fp32 reference is the same" if ref32["answer"]["text_meta_style"]
                                            == ref16["answer"]["text_meta_style"] else "; the fp32 reference differs")
            print(f"             against the 16-bit reference ({ref16['answer']['tokens']} tokens): {answer}{fp32}")
            print(f"     timing: {timing(s)}")

        finished = [s for s in sends if s["end"] and not s["end"]["data"].get("stopped") and len(s["tokens"]) > 1]
        print("\nTiming of finished answers, by steps per frame:" + ("" if finished else " none"))
        for steps in sorted({s["steps"] for s in finished}, key=lambda v: (v is None, v or 0), reverse=True):
            group = [s for s in finished if s["steps"] == steps]
            first = [s["tokens"][0]["data"]["ms"] / 1000 for s in group]
            per = [(s["tokens"][-1]["data"]["ms"] - s["tokens"][0]["data"]["ms"]) / (len(s["tokens"]) - 1) / 1000
                   for s in group]
            total = [s["end"]["data"]["total_ms"] / 1000 for s in group if s["end"]["data"].get("total_ms") is not None]
            print(f"  {steps if steps is not None else '?'} steps: {len(group)} answer(s); median first token after "
                  f"{statistics.median(first):.1f} s (range {min(first):.1f}-{max(first):.1f}), then one every "
                  f"{statistics.median(per):.2f} s, all after {statistics.median(total):.1f} s")

        print("\nTyped sends:" + ("" if typed else " none"))
        for s in typed:
            d = s["request"]["data"]
            pieces = [t["data"].get("text", "") for t in s["tokens"]]
            text = s["end"]["data"].get("answer") if s["end"] else "".join(pieces)
            state = "unfinished" if s["end"] is None else ("stopped" if s["end"]["data"].get("stopped") else "finished")
            print(f"  {when(s['request'])}  {d.get('prompt_tokens')} prompt tokens -> {text!r} ({count(len(pieces))}, "
                  f"{state}); {timing(s)}")

        notes = [e for e in events if e.get("ev") in ("model.message", "error", "ui.keyboard", "model.setting")]
        print("\nSettings, keyboard, Meta's messages and errors:" + ("" if notes else " none"))
        for e in notes:
            d = e.get("data", {})
            if e["ev"] == "model.setting":
                print(f"  {when(e)}  setting: Repeat {'on' if d.get('repeat') else 'off'}, "
                      f"{d.get('steps_per_frame')} steps per frame")
            elif e["ev"] == "ui.keyboard":
                print(f"  {when(e)}  keyboard: {d.get('event')} ({d.get('chars')} characters in the text box)")
            elif e["ev"] == "model.message":
                print(f"  {when(e)}  {d.get('level')}: {d.get('text')}")
            else:
                print(f"  {when(e)}  error in {d.get('where')}: {d.get('message')}")

        # A stopped or unfinished send only counts against the run if what it produced already differs.
        prompt_wrong = any(not p for p, _, _ in verdicts)
        answer_wrong = any(a not in ("same", "same so far") for _, _, a in verdicts)
        matched = any(p and st == "finished" and a == "same" for p, st, a in verdicts)
        passed = matched and not prompt_wrong and not answer_wrong
        if unreferenced:
            print(f"\nNo reference in {args.reference} for: {', '.join(sorted(set(unreferenced)))}. Make them with "
                  "grounding/reference.py (with and without --variant fp16w).")
        if passed:
            print("\nResult: PASS. The headset sees exactly the reference's prompt and gives the reference's answer.")
        elif not verdicts:
            print("\nResult: no result yet: no send had a reference to compare with.")
        elif prompt_wrong:
            print("\nResult: FAIL. The prompt differs: check the provider's template, system message and tokenizer files "
                  "(Second Eyes > Fill chat provider).")
        elif answer_wrong:
            print("\nResult: FAIL. The prompt is right, but the answer differs: the model computes different numbers here.")
        else:
            print("\nResult: no result yet. The prompt is right and the answer matches so far, but no answer finished.")
        return 0 if passed else 1
    except (ToolError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


def count(n: int) -> str:
    return "1 token" if n == 1 else f"{n} tokens"


def first_difference(a: list, b: list) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first difference at token {i}: here {x}, reference {y}"
    return f"lengths differ: here {len(a)}, reference {len(b)}"


def timing(send: dict) -> str:
    ms = [t["data"].get("ms") for t in send["tokens"] if isinstance(t["data"].get("ms"), (int, float))]
    if not ms:
        return "no tokens"
    text = f"first token after {seconds(ms[0])}"
    if len(ms) > 1:
        text += f", then one every {(ms[-1] - ms[0]) / (len(ms) - 1) / 1000:.2f} s"
    end = send["end"]["data"] if send["end"] else None
    return text + (f", all after {seconds(end.get('total_ms'))}" if end else "")


if __name__ == "__main__":
    sys.exit(main())
