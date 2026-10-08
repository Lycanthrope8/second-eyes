# Frozen-request replay on the headset (A2.5, delivery 1; D98, D99)

The headset replays requests whose PC float32 results are already known, so its llama.cpp path can be compared with them
under D56 before anything interactive is built. This page covers the first step: the fixture and reference bundle, built
on the PC from existing artifacts only. Package `grounding/quest/`; policy `replay-policy.v1.json`
(`a25.replay.frozen_requests.v1`). No model runs here.

Scope: the replayed prompts stay byte-identical to A2.3d's, in both formats. The augmented format
(`coordinates_relations_v2`) is replay-only for now, and any replay timing excludes relation construction, since the
prompts are frozen. `coordinates_v2` is the first interactive implementation, not a format winner.

## Inputs and their identity chain

- **Requests:** A2.3d's request bundle (`requests-20261007-115636`). It is read back in full and must have the pinned
  manifest `0e49a9f6…`.
- **References:** the 0.5B run `20261007_A2_r006` (`raw/results`), read back in full against the requests.
  - Its manifest's full SHA-256 is the one the pinned A2.3d scores manifest (`3c92ac4e…`) records, and it must begin
    with the recorded `f06cb6c6`.
  - Only `manifest.json` is opened in the scores folder; no annotation, score or rules file is read.
- **Tokenizer:** the pinned 0.5B tokenizer. It must be the one the requests were prepared with, and must reproduce every
  frozen token list.

## The frozen list

Three sources, each through the existing code, so the rules match the ones already accepted:

- `a23e_case`: the 32 requests of A2.3e's 16 cases, from `costs.select_cases`, expanded as `cache.py` does.
- `a23e_audit`: the 8 audit requests, from `audit.audit_parents`, expanded as `audit.py` does. The build stops unless
  they are r0001–r0004 and r0329–r0332.
- `longest_in_format`: per format, across both views, the request with the largest (0.5B tokens, request index). This is
  the runs' canary rule applied within each format, so ties go to the later request. For the augmented format this
  should be r0330 (full inventory, augmented), the longest request overall under the same rule, which is already an
  audit request; the build reports which requests it found.

The list holds each request once, in request order, with every source that chose it. Its hash is frozen in the manifest.
With r0330 counted once, it holds at most 41 requests.

**References.** A reference is reused only where the prompt hash, token hash and mapping match exactly (D99(9)). Its
status must be completed, and its choice, restricted shares, tie handling and margin are recomputed from its own offered
logits. Recorded shares may differ from the recomputation by at most 1e-12, since `exp` can differ in the last bit
between platforms.

## Cache boundaries

Both are given per request as the number of leading tokens to keep, for the cached replay paths. Which paths the headset
replays is decided in delivery 1's next step.

- `keep_before_last_token` is n − 1, a repeated identical request. It is verified when the text before the last token
  re-tokenizes to exactly those n − 1 tokens.
- `keep_before_command_line` keeps everything before the document's last line, which must be the command line
  `{"command": …}`. It is verified when the text before that line re-tokenizes to an exact prefix of the frozen tokens.

## The headset's input checks and the written fixtures

The headset applies five input checks in this order, and the first failure decides the outcome:

1. `prompt_bytes`: base64 of UTF-8, then byte count and SHA-256.
2. `mapping`: K and ASK last, letters A..J distinct, targets distinct and not ASK, the mapping hash, then each code's token
   ID.
3. `token_ids`: plain integers, then count and hash, then every ID inside the vocabulary (151,936).
4. `tokenization`: the runtime's own tokenization equals the frozen IDs.
5. `context`: tokens plus the one scored continuation within 8,192, never truncated.

The first three fail as `invalid_input`, the fourth as `tokenization_mismatch` and the fifth as
`context_budget_exceeded`. `grounding/quest/replay_inputs.py` is the reference implementation, and the headset must
reproduce it.

Five **written integration fixtures** are frozen with the bundle. They are labelled `written_integration_fixture`: implementation checks, not dataset
data or a benchmark (D99(5)). Each stops at one check, before any evaluation:

| Fixture | Change from the written base prompt | Expected |
|---|---|---|
| `written.context_overflow` | the fewest filler lines that exceed the context limit | `context_budget_exceeded` |
| `written.prompt_hash_mismatch` | the command's first letter upper-cased in the stored bytes; the hash stays the original's | `invalid_input` / `prompt_hash_mismatch` |
| `written.mapping_ask_not_last` | K/ASK listed first | `invalid_input` / `mapping_invalid` |
| `written.code_token_mismatch` | A carries token ID 33 | `invalid_input` / `code_token_mismatch` |
| `written.token_out_of_vocabulary` | the last token ID is the vocabulary size | `invalid_input` / `token_out_of_vocabulary` |

Non-finite scores and exact ties cannot be produced through the model; the headset's own self-checks cover them.

## Layout

```text
bundle-<time>/
  manifest.json            inputs and their hashes, list hash, counts, token statistics, files, code, runtime
  policy.json              the policy used, byte for byte
  selection.json           the cases, audit parents, longest per format, the list and each request's sources
  headset/                 pushed to the headset; nothing here is a reference
    replay-manifest.json   file hashes and counts, the context limit, continuation, vocabulary, codes and their token IDs, the check order
    requests.jsonl         one record per dataset request
    fixtures-written.jsonl the five written fixtures
  reference/
    references.jsonl       per request: hashes, sources, both boundaries with their character offsets, and r006's row unchanged
```

Each headset record has only flat fields (strings, integers, and lists of them):

- `prompt_b64`, `prompt_bytes`, `prompt_sha256`: the prompt as exact bytes, which reach the tokenizer without a string
  conversion;
- `token_ids`, `input_tokens`, `token_ids_sha256`: A2.3d's hash rule, the decimal IDs joined by commas in ASCII;
- `codes`, `targets`, `code_token_ids`, `mapping_sha256`: SHA-256 of the UTF-8 lines "code TAB target TAB token ID LF",
  one per choice, in order;
- `keep_before_last_token`, `keep_before_command_line`: both −1 in fixtures;
- `kind`, `expected_outcome`, `expected_reason`.

## Commands (laptop)

```text
python -m grounding.quest replay-bundle --requests DIR --small-run DIR --scores DIR --tokenizer-dir DIR --out NEW_DIR
python -m grounding.quest verify-replay-bundle --bundle DIR [--requests DIR [--small-run DIR --scores DIR] [--tokenizer-dir DIR]]
```

**`replay-bundle`** stops at the first kind of mismatch and lists every instance of it, then publishes nothing. It
writes beside the destination, reads everything back against the sources and the tokenizer, and only then renames the
folder into place. It never overwrites.

**`verify-replay-bundle`** reads a bundle back on its own: hashes, every record's input checks, boundaries, references,
selection and fixtures. Given the sources, it also re-derives the selection and compares every record and reference row.
Given the tokenizer, it recomputes both boundaries and rebuilds the fixtures.

Exit codes:
- 0: complete, or a clean readback;
- 1: a readback that found problems;
- 2: invalid input or a mismatch; nothing is published;
- 3: an output error or an unexpected failure; nothing is published.

Tests: `python grounding/tests/test_quest_replay_bundle.py [--tokenizer-dir DIR]`.

## Step 2: the headset replay (D103)

### On the headset

The panel has a **Replay frozen requests (A2.5)** button: one explicit action, disabled while a replay runs. It is
available only when two things hold:

- a bundle has been pushed;
- no panel model has been loaded in the session.

The replay loads its own model and frees it afterwards, so two models are never in memory together.

The code is in `quest-app/Assets/SecondEyes/Grounding/Replay/`:

- `ReplayCore.cs`: plain C#. It holds the records, the hash rules, the ordered input checks, the scoring, the self-checks
  and a JSON writer. It is a port of `grounding/quest/replay_inputs.py` and must match it.
- `StderrCapture.cs`: the capture of `stderr` around the load.
- `ReplayRunner.cs`: the run itself.
- `LlamaNative.cs` and `LlamaRuntime.cs` add the bindings and two worker-thread entries:
  - `LoadCapturedAsync` loads with the capture;
  - `WithModel` runs the replay's native calls on the worker thread.

In order:

1. **Bundle.** `files/replay/bundle/` is read and checked against its own manifest (file hashes, counts, the order of
   the checks).
2. **Self-checks.** These run with synthetic rows and records: the scoring, exact ties going to K, non-finite values
   being errors, and the order of the input checks. A failure stops the replay before the model loads.
3. **Load.**
   - What's requested: 8,192 tokens of context, 2 threads (D59), 16 sequences, flags 0 (the accepted settings).
   - Verbose logging is on and `stderr` (file descriptor 2, process-wide) is redirected to `startup-log.txt` for
     `se_load_ex` only. The original descriptor is flushed and restored however the load ends.
   - Recorded: the requested context and the `se_n_ctx` allocation, apart; `se_llama_version`, `se_system_info`,
     `se_n_vocab`, `se_n_seq` and `se_flags`; the process's memory (VmRSS, VmHWM) before and after the load.
   - A vocabulary that differs from the bundle's stops the replay.
4. **Fixtures.** Each written fixture goes through the ordered input checks, the native tokenizer included, and must
   stop at its expected check. None is evaluated.
5. **Requests.** Each request, in the bundle's order, passes the input checks. These include `se_tokenize` on the
   complete prompt bytes, compared token by token with the frozen IDs; a difference stops the request, with its
   position. Then three paths, all from slices of the frozen list:
   - **U:** all n tokens with keep 0;
   - **R:** right after U, keep n − 1 and evaluate the last token;
   - **P:** clear; the tokens before the command line with keep 0; then the rest with keep k.

   `se_n_cached` is checked after every evaluation, and the intermediate prefix of P is never scored. A path whose
   evaluation fails, or whose cache count differs, keeps its record and is not scored. R is skipped when U did not
   complete.

For each scored path, the full final row is copied (`se_logits`), and a non-finite value anywhere is an error. The
offered logits are written as float32, readable back bit for bit. Log-probabilities come from `se_logprob` and are
reported only. Shares are taken over every offered code, K included, and exact ties go to K.

The output goes to `files/replay/results/<UTC time>/`:

| File | Content |
|---|---|
| `identity.json` | The bundle's hashes, the model file, the requested and allocated settings, the runtime's report, memory, the capture's scope, size and any error, and the app and event-log identity |
| `startup-log.txt` | llama.cpp's own startup lines, raw |
| `selfchecks.json` | Every self-check |
| `fixtures.jsonl` | One line per written fixture |
| `results.jsonl` | One line per request, written as it completes |
| `done.json` | Counts and times, written last; a folder without it is unfinished |

The session's event log gets `replay.start`, `replay.load`, `replay.request` and `replay.end` (`docs/logging.md`). The
D101 comparisons are made on the laptop, never on the headset.

### On the laptop

```text
python -m grounding.quest replay-push --bundle DIR --run RUN_ID [--adb PATH]
python -m grounding.quest replay-pull --run RUN_ID [--results NAME] [--adb PATH]
```

**`replay-push`:**
- reads the bundle back on its own;
- pushes only `headset/` into `files/replay/bundle/` and checks each file's size on the headset;
- writes a receipt with every file's SHA-256 to `runs/<id>/raw/replay/push-<time>.json`.

The float32 references never leave the laptop.

**`replay-pull`:**
- copies the newest results folder that has `done.json`, or the one named, into `runs/<id>/raw/replay/<results>/`;
- also copies the session's event log that `identity.json` names, as `events.jsonl`;
- records what arrived in `pull.json`: hashes, missing files, and whether the result lines match `done.json`.

It never writes over an existing folder.

Neither helper installs, launches or reruns the app.

Exit codes: 0 done; 2 refused, with nothing changed; 3 an output error.

Tests:
- `python grounding/tests/test_quest_replay_device.py`.
- The C# logic was compiled at C# 9 with .NET 8 in Claude's sandbox, and checked against vectors this Python code made
  from the real bundle. It is not part of the repository's suites.
