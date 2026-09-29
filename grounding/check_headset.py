#!/usr/bin/env python3
"""Compare the headset's answers in a run's event log with the PC reference (A1.7c-2).

    python grounding/check_headset.py <run ID> --reference 20260928_A1_r008

Reads every log in runs/<run ID>/raw/ (pull them first with tools/logs.py) and the reference files
runs/<reference>/raw/reference_<prompt id>_<variant>.json that grounding/reference.py wrote. For each send of
an unedited fixed prompt (a model.generate event carrying that prompt's ID), it checks:
  prompt  the token IDs equal the reference's, so the template, the system message and the tokenizer are the
          same as on the PC
  answer  the text equals the 16-bit-weights reference (fp16w) as Meta's runner streams it, token by token;
          the full-precision reference (fp32) is shown too
Typed sends are listed without a comparison. The result is PASS, and the exit code 0, only if there is at least
one fixed-prompt send and every one matches.
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


def first_difference(a: list, b: list) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"first difference at token {i}: headset {x}, reference {y}"
    return f"lengths differ: headset {len(a)}, reference {len(b)}"


def when(event: dict) -> str:
    utc = event.get("utc_us")
    return dt.datetime.fromtimestamp(utc / 1e6).strftime("%H:%M:%S") if isinstance(utc, (int, float)) else "?"


def seconds(ms) -> str:
    return f"{ms / 1000:.1f} s" if isinstance(ms, (int, float)) else "?"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare the headset's answers with the PC reference (A1.7c-2).")
    parser.add_argument("run_id", help="the run whose headset logs to check, e.g. 20260929_A1_r010")
    parser.add_argument("--reference", required=True, help="the run holding the PC reference, e.g. 20260928_A1_r008")
    args = parser.parse_args(argv)
    try:
        raw = ROOT / "runs" / args.run_id / "raw"
        logs = sorted(raw.glob("*.jsonl"))
        if not logs:
            raise ToolError(f"No logs in runs/{args.run_id}/raw/. Pull them first: python tools/logs.py pull {args.run_id}")
        ref_dir = ROOT / "runs" / args.reference / "raw"
        if not ref_dir.is_dir():
            raise ToolError(f"No run {args.reference} with a raw/ folder in runs/")
        events = [e for log in logs for e in read_events(log)]

        print(f"Run {args.run_id}: {len(logs)} log(s). Reference: {args.reference}.")
        loads = [e for e in events if e.get("ev") == "model.load"]
        print("\nModel loads:" + ("" if loads else " none"))
        for e in loads:
            d = e.get("data", {})
            copy = "copied out of the app first" if d.get("copied") else "already copied"
            print(f"  {when(e)}  {d.get('file')}: {seconds(d.get('ms'))}, {copy}; {d.get('backend')}, "
                  f"{d.get('execution_mode')}, {d.get('steps_per_frame')} steps per frame")

        sends = [e for e in events if e.get("ev") == "model.generate"]
        fixed = [e for e in sends if e.get("data", {}).get("prompt_id") not in (None, TYPED)]
        typed = [e for e in sends if e.get("data", {}).get("prompt_id") == TYPED]
        all_match = bool(fixed)
        print("\nSends of fixed prompts:" + ("" if fixed else " none. Press Send once without editing the text."))
        for n, e in enumerate(fixed, start=1):
            d = e["data"]
            ref16 = load_reference(ref_dir, d["prompt_id"], "fp16w", required=True)
            ref32 = load_reference(ref_dir, d["prompt_id"], "fp32", required=False)
            ids, ref_ids = d.get("prompt_token_ids"), ref16["prompt"]["token_ids"]
            prompt_ok = ids == ref_ids
            answer = d.get("answer")
            answer_ok = answer == ref16["answer"]["text_meta_style"] and d.get("answer_tokens") == ref16["answer"]["tokens"]
            fp32_note = ""
            if ref32 is not None:
                fp32_note = "; as fp32: " + ("yes" if answer == ref32["answer"]["text_meta_style"] else "no")
            all_match = all_match and prompt_ok and answer_ok
            print(f"  #{n} {d['prompt_id']} at {when(e)}")
            print(f"     prompt: {len(ids or [])} tokens; same IDs as the reference ({len(ref_ids)}): "
                  + ("yes" if prompt_ok else "NO, " + first_difference(ids or [], ref_ids)))
            print(f"     answer: {answer!r} ({d.get('answer_tokens')} tokens)")
            print(f"             same as the 16-bit reference ({ref16['answer']['tokens']} tokens): "
                  + ("yes" if answer_ok else f"NO, the reference is {ref16['answer']['text_meta_style']!r}") + fp32_note)
            print(f"     first token after {seconds(d.get('first_token_ms'))}, all after {seconds(d.get('total_ms'))}")

        print("\nTyped sends:" + ("" if typed else " none"))
        for e in typed:
            d = e["data"]
            print(f"  {when(e)}  {d.get('prompt_tokens')} prompt tokens -> {d.get('answer')!r} "
                  f"({d.get('answer_tokens')} tokens); first after {seconds(d.get('first_token_ms'))}, "
                  f"all after {seconds(d.get('total_ms'))}")

        notes = [e for e in events if e.get("ev") in ("model.message", "error")]
        print("\nMeta's messages and errors:" + ("" if notes else " none"))
        for e in notes:
            d = e.get("data", {})
            text = d.get("text") if e["ev"] == "model.message" else f"{d.get('where')}: {d.get('message')}"
            print(f"  {when(e)}  {d.get('level', 'error')}: {text}")

        if not fixed:
            print("\nResult: no result yet, since no fixed prompt was sent.")
        elif all_match:
            print("\nResult: PASS. The headset sees exactly the reference's prompt and gives the reference's answer.")
        elif all(e["data"].get("prompt_token_ids") == load_reference(ref_dir, e["data"]["prompt_id"], "fp16w", True)["prompt"]["token_ids"]
                 for e in fixed):
            print("\nResult: FAIL. The prompt is right, but the answer differs: the model's numbers differ on the headset.")
        else:
            print("\nResult: FAIL. The prompt differs: check the provider's template, system message and tokenizer files "
                  "(Second Eyes > Fill chat provider).")
        return 0 if all_match else 1
    except (ToolError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
