# The headset's prompt: goldens and the on-device check (A2.5 delivery 2, D104)

The headset builds the `coordinates_v2` document, the choices line and the prompt itself, from a snapshot record and a
command record. They must be byte for byte the PC's. This page covers:

- the goldens that pin the PC's version;
- the check that compares the headset's with them;
- the commands.

The design is D104 (confirmed by the project lead; ChatGPT's review pending). The record is
`notes/phases/A2.5_d2_prompting.md`.

## The headset's code

`quest-app/Assets/SecondEyes/Grounding/Prompting/`:

- **`PromptCore.cs`**, plain C#:
  - a strict JSON reader that keeps field order and number text;
  - the canonical writer (`codec.enc` and `codec.number`, with floats as Python's `repr`, by exact integer arithmetic);
  - the objects, pose and command lines;
  - the serialized-order mapping (letters over object IDs in ordinal order, K for ASK last);
  - the choices line and the prompt.
- **`GoldenCore.cs`**, plain C#: rebuilds one golden and compares its document, mapping, mapping hash, prompt bytes and,
  if given a tokenizer, its token IDs.
- **`GoldenRunner.cs`**, the Unity glue behind the panel's **Golden check (A2.5 d2)** button, next to Replay. It:
  1. checks every pushed file against `golden-manifest.json`;
  2. loads the model only for its tokenizer, with 1,024 tokens of context and nothing evaluated;
  3. checks every golden;
  4. writes `files/prompting/results/<UTC time>/`: `results.jsonl`, `identity.json`, and `done.json` last.

  The button is disabled while a check or replay runs, and once a panel model has been loaded in the session.

## The goldens

```text
python -m grounding.quest golden-build --bundle DIR --out-root DIR [--tokenizer-dir DIR]
python -m grounding.quest golden-verify --goldens DIR [--tokenizer-dir DIR]
python -m grounding.quest golden-references --goldens DIR [--device cpu|cuda] [--model-dir DIR]
```

**`golden-build`:**

1. Reads the accepted A2.2d bundle (`a22d-20261005-153413`) after its own readback, with its index pinned (`44a994b4…`).
2. Chooses one snapshot each of 3, 6 and 10 objects. Among the materialized selections of that size, it takes the lowest
   SHA-256 of `a25.d2.snapshots.v1` + LF + the selection ID.
3. Takes every derived command of each snapshot (dataset commands), plus two written fixtures per snapshot. The fixtures
   carry quotes, non-ASCII text and a tab, are labelled apart, and serve as implementation checks only.
4. Writes `goldens-<UTC time>/` with:
   - `headset/`: `prompt-asset.json`, the snapshot and command records (the bundle's bytes), `goldens.jsonl` and
     `golden-manifest.json`;
   - `selection.json`;
   - the root `manifest.json`, with the provenance.

Each golden holds:

- the document and its hash;
- the prompt's hash and size;
- the token IDs from the pinned tokenizer, with their hash;
- the mapping and its hash;
- both cache boundaries.

**`golden-verify`** rebuilds every document, mapping and prompt from the headset files, and checks every hash. With the
tokenizer it checks every tokenization too.

**`golden-references`** writes fresh float32 references into `goldens-<time>-references-<UTC time>/`:

- offered logits, log-probabilities, shares, the choice (exact ties to K) and the margin;
- the model is A2.3a's `TorchModel`: float32, eager attention, the checkpoint checked against its pins;
- the device is recorded, and a canary compares the two forward paths.

## On the headset

```text
python -m grounding.quest golden-push --goldens DIR --run RUN_ID
python -m grounding.quest golden-pull --run RUN_ID [--results NAME]
```

- **`golden-push`** reads the goldens back first. It pushes the headset part into `files/prompting/goldens`, with the
  golden manifest last, because the app sees goldens only once that file arrives. It writes a receipt to
  `runs/<id>/raw/goldens/`.
- **`golden-pull`** copies the newest finished results folder and the session's event log into `runs/<id>/raw/goldens/`.
  It never overwrites a folder.

Neither installs, launches or reruns the app.

Exit codes: 0 done; 1 a readback with problems (`golden-verify`); 2 refused, with nothing changed; 3 an output error.

Tests:

- `python grounding/tests/test_quest_prompt_goldens.py`;
- the C# logic was checked at C# 9 with .NET 8 in Claude's sandbox, against the Python serializer, the prompt builder and
  fixture goldens. That check is not part of the repository's suites.
