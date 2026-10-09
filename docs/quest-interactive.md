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
   - `Off` (the default) evaluates in full;
   - `ExactPrefix` keeps the longest common prefix with the previous request on the same snapshot, at most n − 1. Another
     snapshot invalidates it, and a failure resets it.
5. **Evaluation and scoring:** offered letters only; exact ties go to K; a non-finite output fails.
6. **The outcome:**
   - a status: `completed`, `ask`, `refused`, `failed` or `cancelled`, with a reason;
   - the target object;
   - the offered logits, shares and log-probabilities, and the margin;
   - the tokens kept and evaluated, and the time of each stage.

The cache's numerical acceptance is delivery 1's open question (D101). Turning `ExactPrefix` on for interactive use is
the project lead's decision.

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
