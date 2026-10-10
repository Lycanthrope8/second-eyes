# A2.5 delivery 3: the interactive pipeline (headset)

The pipeline turns a request into a structured outcome: a target object, or ASK. A request is a versioned annotated
snapshot plus a typed command. Every intake hands the pipeline the same request type: the panel's check now, and the
presets and the ADB inbox in delivery 4.

Code: `quest-app/Assets/SecondEyes/Grounding/Interactive/`

- `InteractiveCore.cs`: Unity-free, tested on a PC. It holds the request and outcome types, the cache (`PrefixCache`)
  and the pipeline (`Pipeline.Run`).
- `InteractiveRunner.cs`: the headset side.

Phase note: `notes/phases/A2.5_d3_interactive.md`.

## A request, start to finish

1. **Checks:** the scene text must hash to the registered snapshot, and the command must name the same scene and
   revision. Malformed records are refused with a reason.
2. **The prompt:** delivery 2's core builds it, with the D104 mapping; its hash and the mapping's are recorded.
3. **Tokens:** the prompt is tokenized natively, and refused if it would not fit 8,192 tokens with one continuation
   position.
4. **The cache plan,** by mode:
   - `Off`, the default and the only operational mode: prior KV cleared, complete prompt (path `U`);
   - `ExactPrefix`, diagnostic execution only: the longest common prefix with the previous request on the same snapshot,
     at most n − 1 (path `P`). Another snapshot invalidates it, and a failure resets it. An operational request with it
     is refused (`prefix_reuse_not_accepted`).
5. **Evaluation and scoring:** offered letters only; exact ties go to K; a non-finite output fails.
6. **The outcome:**
   - a status: `completed`, `ask`, `refused`, `failed` or `cancelled`, with a reason. An ASK records its basis:
     `model_selected` or `exact_tie`. Technical failures are never an ASK;
   - the purpose and the execution path that actually ran;
   - the target object;
   - the offered logits, shares and log-probabilities, and the margin;
   - the tokens kept and evaluated, and the time of each stage.

Three statuses are kept separate: the functional implementation, uncached numerical acceptance against float32, and
cached numerical acceptance under D56/D101. Enabling prefix reuse operationally needs cached numerical acceptance, or an
explicit amendment by the project lead.

## The Interactive check

**Interactive check (A2.5 d3)** on the panel needs the goldens on the headset (`golden-push`). It sends the 32 golden
requests through the pipeline twice, cache off and then exact prefix, in about 15 minutes. It writes
`files/interactive/results/<UTC time>/`: `identity.json`, `outcomes-off.jsonl`, `outcomes-prefix.jsonl` and
`done.json`, written last.

On the laptop:

```text
python -m grounding.quest interactive-pull --run RUN [--results NAME]
```

It prints each pass's statuses, whether the result lines match `done.json`, the prompts equal to the goldens, and the
choices and offered logits that agree with and without the cache.

## The session (delivery 4)

The panel's row **Session (A2.5 d4)** has four buttons: start or end, Scene (3, 6 or 10 objects), Preset (the scene's
next golden dataset command) and Cancel.

A running session answers the ADB inbox. Requests are pushed under a temporary name and renamed when complete, and each
request ID is processed once. Every request is answered under its ID: `outbox/(ID).ack.json`, then
`outbox/(ID).result.json`.

A request bound to another scene is refused (`stale_scene_binding`), and so is a result that finishes after a scene
switch (`stale_result_scene_changed`).

Cancel works in two ways:

- queued requests are answered as cancelled at once;
- a running evaluation finishes, and its result is answered as cancelled.

The session is operational, so it runs path U only.

```text
python -m grounding.quest inbox-send --run RUN --goldens DIR --snapshot 3 --dataset 1 [--request-id ID] [--resend]
python -m grounding.quest inbox-send --run RUN --goldens DIR --snapshot 3 --text "the chair by the window"
python -m grounding.quest outbox-pull --run RUN
```

Design and protocol: `notes/phases/A2.5_d4_intake.md`. Tests: `python grounding/tests/test_quest_inbox.py`.

## Measurement (delivery 5)

```text
python -m grounding.quest inbox-batch --run RUN --goldens DIR --snapshot 3        # every dataset command of the 3-object scene, one at a time
python -m grounding.quest outbox-pull --run RUN                                   # also brings the session's load log
python -m grounding.quest measure-report --run RUN --goldens DIR --references REFS
```

The report covers app-observed latency per scene size and intake, stage times, tokens, memory as the runtime reports it,
and the PC's push times kept apart. Agreement with the CPU baseline is descriptive only. Design:
`notes/phases/A2.5_d5_measurement.md`. Tests: `python grounding/tests/test_quest_measure.py`.

`inbox-batch` is paced. It sends one request, waits for its answer, prints it, then sends the next, so no request waits
in the headset's queue. It returns when the batch is answered. `--all-at-once` sends without waiting.

On the headset:

- a waiting request whose scene changed is refused before evaluation (`scene_changed_before_evaluation`);
- End session answers every waiting request (`session_ended`) and lets the running one finish first.
