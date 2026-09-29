#!/usr/bin/env python3
"""Compare the headset's answers in a run's event log with the PC reference (A1.7c-2, D43).

    python grounding/check_headset.py <run ID> --reference 20260928_A1_r008

Reads every log in runs/<run ID>/raw/ (on the headset, pull them first with tools/logs.py) and the reference
files runs/<reference>/raw/reference_<prompt id>_<variant>.json that grounding/reference.py wrote. Each send is
model.request, then one model.token per answer piece, then model.generate at the end, all with the same request
number. For each send of an unedited fixed prompt it checks:
  prompt  the token IDs equal the reference's, so the template, the system message and the tokenizer are the
          same as on the PC; this works even if the answer never finished
  answer  the text equals the 16-bit-weights reference (fp16w) as Meta's runner streams it, token by token, and
          otherwise says at which token it first differs; the full-precision reference (fp32) is shown too
Typed sends are listed without a comparison. The result is PASS, and the exit code 0, only if there is at least
one fixed-prompt send, and every one has the reference's prompt and finished with the reference's answer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
        verdicts = []
        print("\nSends of fixed prompts:" + ("" if fixed else " none. Press Send once without editing the text."))
        for n, s in enumerate(fixed, start=1):
            d = s["request"]["data"]
            ref16 = load_reference(ref_dir, d["prompt_id"], "fp16w", required=True)
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
            print(f"  #{n} {d['prompt_id']} at {when(s['request'])} ({s['log']}, request {d.get('request')}): {state}")
            print(f"     prompt: {len(ids)} tokens; same IDs as the reference ({len(ref_ids)}): "
                  + ("yes" if prompt_ok else "NO, " + first_difference(ids, ref_ids)))
            print(f"     answer: {text!r} ({count(len(pieces))})")
            fp32 = "" if ref32 is None else ("; the fp32 reference is the same" if ref32["answer"]["text_meta_style"]
                                            == ref16["answer"]["text_meta_style"] else "; the fp32 reference differs")
            print(f"             against the 16-bit reference ({ref16['answer']['tokens']} tokens): {answer}{fp32}")
            print(f"     timing: {timing(s)}")

        print("\nTyped sends:" + ("" if typed else " none"))
        for s in typed:
            d = s["request"]["data"]
            pieces = [t["data"].get("text", "") for t in s["tokens"]]
            text = s["end"]["data"].get("answer") if s["end"] else "".join(pieces)
            state = "unfinished" if s["end"] is None else ("stopped" if s["end"]["data"].get("stopped") else "finished")
            print(f"  {when(s['request'])}  {d.get('prompt_tokens')} prompt tokens -> {text!r} ({count(len(pieces))}, "
                  f"{state}); {timing(s)}")

        notes = [e for e in events if e.get("ev") in ("model.message", "error", "ui.keyboard")]
        print("\nKeyboard, Meta's messages and errors:" + ("" if notes else " none"))
        for e in notes:
            d = e.get("data", {})
            if e["ev"] == "ui.keyboard":
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
        if passed:
            print("\nResult: PASS. The headset sees exactly the reference's prompt and gives the reference's answer.")
        elif not verdicts:
            print("\nResult: no result yet, since no fixed prompt was sent.")
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
