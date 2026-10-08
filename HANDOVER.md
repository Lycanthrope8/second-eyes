# Second Eyes · Handover for the A2.5 implementation chat

Date: 7 October 2026, written by Claude at the end of a long, many-times-compacted chat. It replaces the earlier
handover (`Second_Eyes_Handover_2026-10-07.md`), which predates A2.3c's closure, A2.3d and A2.3e. Read it fully before
acting. The code and committed records are in the public repository **https://github.com/Lycanthrope8/second-eyes**
(clone it; `github.com` is reachable from the sandbox, the GitHub REST API is often rate-limited, so use `git clone`).

**Corrected on 8 October 2026** with ChatGPT's review of this handover (`Second_Eyes_Claude_Handover_Review_2026-10-08.md`,
against `168344e`); its replacement wording is applied below. Since then, in the commit that added this file to the
repository: the A2.3e records follow-up of section 0 is done (`notes/phases/A2.3e_checks.md` section 5), and A2.5
delivery 1's first step, the fixture and reference bundle, is built and verified (D100, D101;
`notes/phases/A2.5_d1_replay.md`).

---

## 0. Start here

1. **Baseline and one records follow-up.** The current repository baseline is `168344e`, which contains the
   legacy-header regeneration repair and D99. The repository and updated A2 report record the corrected costs
   regeneration as complete. Do not reapply the repair by default. Obtain the corrected folder's full manifest hash and
   successful readback if they are not already in the records, and append them to the A2.3e phase note. ChatGPT has
   checked the correction rule and expected values, but has not independently reviewed the regenerated folder/readback.
   This narrow records follow-up does not reopen A2.3 or block A2.5 delivery 1.
2. **Then A2.5, delivery 1: frozen-request replay on the headset** (section 7). D98 defines A2.5; D99 records the
   project lead's input and protocol decisions; the implementation order is approved.
3. **Working rules** (section 2) matter as much as the facts: ChatGPT designs, Claude implements, the project lead runs
   every acceptance and commits.

## 1. The project in one paragraph

Second Eyes (proposal Revision 3, September 2026; A2 replaced by the Revision 3.1 supplement, 1 October 2026): a
micro-UAV pre-scans a region the user cannot see; a Meta Quest 3 turns the observations into a semantic memory; the user
then tasks the drone in natural language about things only the drone has seen, with all AI inference on the headset at
deployment. A sub-billion-parameter model decides what the user means, deterministic geometry decides where to go, and
the drone's flight controller flies. Phases: A0 scope/IRB, A1 headset feasibility, A2 grounding, A3 Wizard-of-Oz
elicitation, A4 pre-scan without a drone, A5 drone purchase, A6 drone integration, A7 goal execution, A8 experiment, A9
pilot study, A10 formal study; Track B (SpatialLM, Gate D) runs in parallel. Gates A–F are in the proposal's section 11.

## 2. People, roles and working rules

- **Project lead: Jubayer Hossain** ("JH" in run configs). Runs every acceptance on his Windows laptop, the lab RTX PC
  (over Remote Desktop) and the Quest 3. Pastes console output back, commits and pushes himself, and decides.
- **ChatGPT** owns the research design. It writes the briefs and correction instructions and reviews uploaded archives.
  A brief is authoritative for its increment.
- **Claude** implements:
  - read the whole brief and the actual repository first;
  - report a material contradiction or an open design question before coding, and disclose routine choices;
  - never redesign, tune after seeing results, commit, push, reset or unstage;
  - stop at the brief's stopping point.
- **Delivery pattern:**
  - a ZIP of new or changed files (`Expand-Archive -Force` when it updates existing files);
  - shared notes (`notes/decisions.md`, `notes/limitations.md`, `notes/failures.md`, `README.md`, `docs/README.md`)
    changed only by one-off Python scripts that insert or replace one line at a stable anchor, keep the file's line
    endings, skip what is present, and are run once and deleted;
  - a PowerShell sequence with the expected output and hashes;
  - narrow `git add` lists. Do not stage the unrelated pre-existing AndroidManifest/O25 and solution-file changes in
    these records commits. Intended future brief-scoped changes are reviewed on their own merits.
- **Test discipline:**
  - expectations derived by hand before the code;
  - suites runnable directly and module-style;
  - relevant earlier suites rerun;
  - labelled sandbox rehearsals, never called acceptance.
- **Verification habit.** When outputs come back, Claude re-verifies them in the sandbox with independently written code
  (readbacks, recomputation from raw rows) before writing acceptance records, and records discrepancies honestly as
  "Claude's error" in `notes/failures.md`.
- **Records:**
  - `notes/decisions.md`: an Open table (`| ID | Question | Needed by | Options so far |`) and a Decided table, newest first.
  - `notes/phases/<ID>_<name>.md`: what was built, configuration, key numbers with run IDs, decisions, failures,
    deviations, open issues, paper hooks; acceptance sections are appended.
  - Runs: `python tools/runs.py new A2 --purpose "..." --operator "JH"`, then `runs.py note <id> "..."` and `runs.py check`.
    `raw/` is Git-ignored and only `config.yaml` is committed.
- **Style:** plain, precise English; every number traceable to a run or artifact; no claim beyond the evidence.
- **Sandbox limits for Claude:**
  - commands stop after 300 s, and background processes die, so split long jobs;
  - Hugging Face is unreachable, but the pinned tokenizer files are in the repository;
  - there is no PyTorch, GPU, Unity or Android NDK: Quest code is written blind, built and run by the project lead.

## 3. Machines, paths and environments

**Laptop (Windows):**
- repository `C:\Users\jubay\second-eyes`; data `C:\Users\jubay\second-eyes-data`;
- `.venv`: Python 3.11.9, transformers 4.57.6, tokenizers 0.22.2, jsonschema 4.26.0;
- Git `core.autocrlf=true`;
- CPU AMD64 Family 25 Model 80, 16 logical CPUs, Windows 10.0.26200.

**Lab RTX PC (Remote Desktop):**
- repository `C:\Users\jhossai3\second-eyes`;
- Python `C:\Users\jhossai3\second-eyes-venvs\a2-gpu\Scripts\python.exe` (3.12.10, torch 2.11.0+cu128, CUDA 12.8,
  transformers 4.57.6, tokenizers 0.22.2);
- NVIDIA RTX PRO 6000 Blackwell (96 GB), driver 596.71, 127 GB RAM, about 1.5 TB free;
- data `C:\Users\jhossai3\second-eyes-data`; transfers `C:\Users\jhossai3\second-eyes-transfer`;
- laptop files reached as `\\tsclient\C\Users\jubay\...` (Remote Desktop drive redirection).

**Quest 3:**
- Unity 6000.3, Meta XR SDK 207, URP, 72 Hz;
- llama.cpp b11277 (eae11d22) through the project's `libse_llama`;
- Qwen2.5-0.5B-Instruct as GGUF Q8_0 (531 MB) in the app's data folder;
- runs plugged in (D22); logs pulled with adb; OVR Metrics for frames.

**Model folders** (Git-ignored):
- 0.5B: `grounding\models\qwen2.5-0.5b-instruct\hf` on the RTX PC, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- 7B, RTX PC: `grounding\models\qwen2.5-7b-instruct\hf`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, Apache-2.0.
- 7B, laptop: tokenizer files only, in `grounding\models\qwen2.5-7b-instruct\tokenizer`.
- Tokenizers: the 7B's `vocab.json`, `merges.txt` and `tokenizer_config.json` are byte-identical to the 0.5B's pinned
  files, plus `tokenizer.json` `c0382117…`.

**The RTX PC's untracked run configs.** A run created there leaves an untracked `runs/<id>/config.yaml`. Once the laptop
commits that config, the RTX PC's next `git pull` is blocked. Before pulling, move the config aside to
`C:\Users\jhossai3\second-eyes-data\run-configs\`. This was done for r006, r007 and r008; the r004 and r005 folders are
in `C:\Users\jhossai3\second-eyes-data\`.

## 4. Status

**A1: closed for now (D93).**
- llama.cpp is the headset runtime (D60).
- The detector is YOLOX-Nano on the CPU, scanning at 1 Hz and released before commands (D91; r045: p95 98 ms, 0.17%
  stale frames).
- Speech (A1.9) is optional. A1.10d's formal conditions (D92), the 30-minute soak (A1.11) and formal Gate A are deferred
  to the final evaluation.

**A2.1–A2.2: accepted.** D66–D78: contract, relations, directions, serializer, resolver; IRef-VLA scene0010_01 adapter
(61 objects, 1,951 annotations); rules baseline; category-complete subscenes; model inputs.

**A2.3a pilot** (r003) and **A2.3b scoring** (D81, D82): the 0.5B never beat always answering B.

**A2.3c (D94, run `20261007_A2_r005`): closed.**
- A crossed grid of letter assignment and list order: 4,536 cells; canaries 3 of 3 and repeat controls 128 of 128, all
  with difference 0.0.
- Strong answer-code sensitivity plus some list-order sensitivity; full-grid agreement close to the always-B reference.

**A2.3d (D95): closed.** Runs `20261007_A2_r006` (0.5B) and `20261007_A2_r007` (7B): 1,024 of 1,024 each; 256 more
parents; one frozen hash-based letter assignment and list order per parent and view. Correct out of 256:

| View / format | 0.5B | 7B | Rules | Always B | Second position |
|---|---|---|---|---|---|
| full inventory / coordinates | 32 | 5 | 2 | 39 | 33 |
| full inventory / augmented | 32 | 6 | 2 | 39 | 33 |
| source-known / coordinates | 54 | 7 | 253 | 57 | 54 |
| source-known / augmented | 55 | 1 | 253 | 57 | 54 |

- The 7B answered ASK in 917 of 1,024 requests.
- The 0.5B chose B or C in 973 of 1,024.
- The rules' full-inventory failures are all `uncertain_anchor` and `unknown_category`.

**A2.3e (D96): closed with documented perspective coverage pending under O27.**
- Input audit: 128 of 128 checks.
- Costs (PC):
  - augmented relation construction median 30 ms (p95 179) in the full inventory, plus about 830 tokens;
  - serialization about 30 ms; tokenization 3–6 ms.
- 0.5B cache check (r008): 192 numerical D56 comparisons and 32 identity checks, all passed.
  - A repeated identical request scores its last token in 19.2 ms against 90.6 ms uncached.
  - A new command reuses about 94 tokens, because the choices line precedes the scene.
- Perspective check blocked (O27).

**A2.3: closed (D97)** "for baseline development and integration handoff. Input integrity and PC cache checks accepted.
Published-data perspective coverage remains pending under O27." The blockers justify the deferral; they do not show that
no suitable resource exists; no viewpoint result is claimed.

**Percentile rule (D97).** Nearest rank: the sorted value at zero-based index ceil(p × n) − 1.
- The earlier index round(p × (n − 1)) changed only the n = 256 cells.
- A2.3e's costs are regenerated from saved measurements (`costs-20261007-174912-p95`, manifest `dd1acc12…`, at
  `168344e`), and A2.3d's per-format table is corrected.
- Folders made under the old rule record no rule and still verify.

**Sequence (D97):** A2.5 base-model Quest integration → A2.6 development and baseline assessment → A2.4 fine-tuning →
the export and evaluation checks again. The final held-out commands stay closed.

## 5. Commits since A1's closure

`22ad1df` A1 closed (r047 config only) · `5326f8e` A2.3c code · `d0ed082` A2.3c closed, A1 records (D93) · `7ab2917`
7B pins and acquisition · `b4fb0bd` A2.3d preparation and rules · `bc330e9` smoke check and runs · `9e8f8b0` runs
r006/r007 and scoring · `4ae9c07` input audit · `75d385d` A2.3d closed, D96 · `eabc0d3` costs and cache check ·
`695aa27` A2.3e results, O27 · `c21bf2b` A2.3 closed (D97), A2.5 defined (D98); its message says the costs were
regenerated, but that happened only after the repair (`notes/failures.md`) · `168344e` the regeneration repair, costs
regenerated, D99: the current baseline.

## 6. Key artifacts and hashes

| Artifact | Where | Identity |
|---|---|---|
| A2.2d bundle | `second-eyes-data\iref-model-inputs\a22d-20261005-153413` | preparation index `44a994b4…` |
| Annotations (pinned) | `second-eyes-data\iref-out\a22a-20261004-211140\again\reference_only\iref-annotations.json` | `06cfdfb4c8b55fdbd8f86ae35030471d5f5a793fe9ebf144f067649e7d4ce72b` |
| A2.3a requests; r003 | `iref-pilot\requests-20261005-212446`; `runs\20261005_A2_r003\raw\pilot` | manifest `a09d9d51…`; results `7e558382…` |
| A2.3b scores | `iref-pilot-scoring\scores-20261006-021046` | summary `9a7231ae…`, report `b3e56936…` |
| A2.3c requests; run r005; scores | `iref-order\requests-20261007-020734`; `runs\20261007_A2_r005`; `iref-order\scores-transfer-20261007-020734` | manifest `f0c883d3…`; run zip `75023049…`; scores summary `f48afcc8…` |
| A2.3d requests | `iref-compare\requests-20261007-115636` (also on the RTX PC) | manifest `0e49a9f6ed1bd07569ed0a24ec5753cbc4c26a55d1389f0e2240695f8bfe3db0`; selection `b42e752f…` |
| A2.3d rules | `iref-compare\rules-20261007-115636` | manifest `151b0e82f0a94ddf…` |
| Runs r006, r007 | `runs\20261007_A2_r006`, `runs\20261007_A2_r007` (`raw\results`, smoke records) | zips `d7f2f8ba…`, `4f7bf7cc…`; manifests `f06cb6c6…`, `de52909b…` |
| A2.3d scores | `iref-compare\scores-20261007-115636` | manifest `3c92ac4e843704a821f16718635f7815a2bfc78a1ac38c0e2ff517df43e33f57` |
| A2.3d review archive | `A2_3d-review-20261007-165741.zip` | given to Claude; ChatGPT reviewed A2.3d |
| Input audit | `iref-compare\input-audit-20261007-171707` | manifest `150bd6f17702564fd3839896d18120178321f9311302805df19ad9c03174842b` |
| Costs (original); regenerated | `iref-compare\costs-20261007-174912`; `costs-20261007-174912-p95` | regenerated manifest `dd1acc12058faf4ad8038dd2342ca3cababf8d04db40905417572d2dd5421060` |
| Run r008 (cache check) | `runs\20261007_A2_r008\raw\cache` | zip `db0f776f…`; manifest `7ee088478b96c8f4f10116a13ea7590453ee2ad922a2056d67a7512c08b76505` |

## 7. A2.5: base-model Quest integration (D98, D99)

**Objective.** Typed command plus a versioned scene snapshot → prompt → local offered-letter scoring → object-ID
mapping → structurally validated target proposal shown in the app.
- Qwen2.5-0.5B-Instruct through the accepted llama.cpp integration, with D59's two threads and the app's scheduling
  constraints.
- All runtime inference and grounding preparation run on the headset; the PC only prepares, references and analyses.
- No drone motion, speech, detector concurrency, soak, Gate A or accuracy claim.

**D99, the project lead's decisions:**
1. **O21 resolved for A2.5:** PC-entered commands through an **ADB inbox**, plus **on-panel presets**, both feeding **one
   request-processing path**. A headset keyboard is deferred, and ADB is development transport only.
2. **Inbox protocol:**
   - write under a temporary name, then rename when complete;
   - each request carries a unique request ID and the expected scene identity and revision;
   - each is processed once, with an acknowledgement and result under the same ID;
   - stale scene bindings are rejected explicitly.
3. **Timing:** "app-observed command to validated target proposal", with the intake method stated.
   - Log detection, parsing and validation, queueing, inference start and end, and the outcome; for presets, from the
     button event.
   - The reported app-side total includes parsing/validation and queueing from app observation of a ready inbox
     request, or from the preset button event, through the validated target proposal or other terminal outcome.
     Report delivery/polling outside that boundary separately where measurable. Never subtract unsynchronized PC and
     headset clocks. Inference-only time is a component, not the end-to-end result.
   - This is partial runtime evidence, not the speech gate.
4. **Number formatting:** test the existing serializer against goldens first, and avoid a general Python-compatible
   formatter unless needed. Any new canonical serialization is versioned with matching PC references, and frozen replay
   prompts stay byte for byte. Preserve numeric values, units, schema semantics and unknowns. Shortest-round-trip output
   is not itself a proof of byte equality across languages. Compare against goldens; version any required canonical
   rendering policy and create matching PC references. Original replay prompts remain byte-identical.
5. **Snapshots:** annotated 3-, 6- and 10-object snapshots with their provenance verified. Dataset commands and written
   integration fixtures are labelled apart; written ones are implementation checks, not benchmark or held-out data.
6. **Stable choices:**
   - letters and list order come from the scene's inventory identity, the stable object IDs and a fixed salt;
   - command text and targets are excluded, and choices stay before the scene;
   - the mapping identity is recorded;
   - a scene or inventory change invalidates incompatible cache state, and a pose change invalidates the affected
     prefix even when the mapping is unchanged.
7. **Runtime identity:** before replay, record the actual GGUF hash, quantization, llama.cpp build, context allocation
   and KV-cache types, and verify that this is the accepted A1 artifact. Memory is as the runtime reports it; "about
   100 MB" of KV cache is only an estimate until measured. The proposed context allocation is 8,192 tokens under the
   accepted brief; keep the actual runtime identity checks.
8. **D56, unchanged:**
   - every mismatch is reported individually, and an 8-bit-versus-float32 difference is no exemption;
   - if replay fails, run the same GGUF on desktop llama.cpp to separate export from integration errors;
   - implementation progress and replay acceptance are separate statuses.
9. **References:** replay may reuse the float32 results (r006) only where prompt bytes, token IDs and mappings match
   exactly. Interactive prompts need fresh PC references (RTX PC, float32, A2.3a's TorchModel path).

**The brief's other requirements (D98):**
- **Replay first.**
  - Use the 32 A2.3e requests (16 cases by `costs.select_cases`, both formats), the 8 audit requests (r0001–r0004 and
    r0329–r0332) and the longest A2.3d request per format, deduplicated, with the list frozen before testing.
  - Check token IDs and special tokens, final-position logits for the offered letters, letter-to-object mapping, K as
    ASK, exact ties to K, and explicit errors for overflow, invalid input and non-finite scores.
  - Keep the 8,192-token ceiling, never truncate, and record the context allocation.
  - Preserve the best candidate and the ranking over the union of candidates with at least 1% restricted share in
    either compared path. Apply D56's TVD ≤0.05 requirement to the cached Quest path against the pinned float
    reference, and the A2 cache-check requirement to cache-on against cache-off. The original D56 requires
    best/plausible-order agreement for the uncached path against float and reports its distribution difference; it
    does not independently impose the same ≤0.05 uncached-to-float TVD ceiling. Save every offered score and
    restricted share and report each mismatch. Log-probability equality is not the acceptance criterion. Quantization
    is not an exemption (recorded as D101).
  - Save every offered logit, restricted share, mapping, decision, input and token hash, identity and cache state per
    path; pass flags alone are not enough.
- **Interactive:**
  - `coordinates_v2` is the first interactive implementation, not a format winner. The augmented format is replay-only
    initially, with relation-construction cost excluded from those replay timings; it remains relevant to later
    training work;
  - the prompt is built on the headset, with golden comparisons against the PC serializer;
  - the whole 3–10-object snapshot is offered, with no command-dependent cropping;
  - prompt order is not changed; the reusable prefix with stable choices is measured first.
- **Caching:** only exactly equal tokens under the same model, tokenizer, runtime configuration and positions.
  - The boundary comes from actual token sequences.
  - Reuse is based on actual token identity and compatible positions/state; reset when compatibility cannot be
    established. Passing pose-related invalidation checks is not evidence of viewpoint grounding and does not close O27.
  - It must handle a new command in the same scene, a repeated command, changed attributes or geometry, changed
    inventory, mapping, order, pose, format, model or tokenizer, and cancellation, scene replacement and pause/resume.
  - Cache ownership is bounded and released explicitly.
- **Outcomes:** object selected, ASK, invalid or stale request, context overflow, execution failure, cancelled.
  - An object selection is checked: the letter was offered, the object belongs to the active snapshot, and the scene and
    mapping revision still match.
  - Structural validity does not establish semantic correctness. A selected object may be correct or incorrect; the
    structural checks establish offered membership, active object membership and matching scene/mapping revision only.
  - Inference stays off the main thread; overlapping submissions are queued or rejected explicitly; a cancelled or
    superseded request never publishes a late proposal.
  - The proposal goes through a replaceable interface for later goal execution.
- **Measurements:** model loading; the first command with an empty cache; a different command with the same scene; a
  repeated identical request; the first command after invalidation.
  - Report median and p95 (corrected rule), the input, reused and suffix token counts, resident and peak memory with the
    cache allocation, and frame behavior (OVR convention).
  - Include cancellation and lifecycle results, on 3-, 6- and 10-object snapshots with a fixed modest command sequence.
  - No soak.
  - Record memory against the existing app limit, approximately 5.75 GiB, and check UI responsiveness. Report median and
    nearest-rank p95 with sample counts and timing boundaries. A2.5 adds no new latency threshold or soak requirement
    and cannot close Gate A.
- **Acceptance:** replay passes token and mapping checks and D56; interactive prompts match their versioned PC
  references; caching and invalidation behave; typed requests give valid structured outcomes; cancellation, stale-result
  rejection and lifecycle checks pass; resources and limitations are recorded. Use focused offline checks and on-headset
  self-checks through the existing harness, with no new Unity test framework.
- **Deliverables:** code, the fixture and reference bundle, raw headset scores and events, the resource report, the
  phase note, and reproducible laptop and headset instructions.

**Approved order of work:**
1. **Replay:**
   - the fixture and reference bundle builder on the PC, from existing artifacts with an exact-match check against r006;
   - a headset replay mode, uncached and cached through `se_eval(tokens, n, keep)`;
   - a laptop D56 comparison;
   - a desktop llama.cpp fallback if needed.
2. The C# `coordinates_v2` serializer and the stable letter policy, with golden PC references and fresh PC float32
   references for interactive prompts.
3. The cache manager and the interactive pipeline, with the **common request-intake interface** wired here, so that ADB
   and presets cannot behave differently.
4. ADB inbox intake, presets and the UI: target highlight, cancellable workflow, structured outcomes.
5. The measurement run, the phase note and acceptance.

**What exists on the headset side:**
- **`native/se_llama.h`:**
  - `se_load_ex(path, n_ctx, n_threads, n_seq, flags)`, `se_tokenize` (parses special tokens, adds none), `se_piece`;
  - `se_n_cached`, `se_eval(tokens, n, keep)` (keep a cached prefix and evaluate after it), `se_argmax`,
    `se_logprob(token)`, `se_logits(out, max)` (the raw row);
  - `se_score`, `se_score_many`, `se_memory_kb(peak)`.
- **`Grounding/LlamaNative.cs`:** the C# bindings. It lacks `se_logits`, `se_logprob` and `se_score`. The repository's
  native API already declares the required raw-logit and prefix operations. First verify the deployed library's
  identity/API against the accepted build under D99. A C# binding addition should suffice if the installed library
  matches that API; do not assume a rebuild or its absence before this check.
- **`Grounding/LlamaRuntime.cs`:** one worker thread off Unity's main thread, 2 threads (D59).
- **`Grounding/ChatPanel.cs`:** the preset panel (D49), with prompt files `grounding/prompts/a17-*.json` and a copy in
  the app.
- **`Grounding/CommandSchedule.cs`:** the A1.10 fixed command schedule.
- **`Logging/EventLog.cs`:** the JSONL event log, schema `schemas/log-event.v1.json`.
- **The serializer to port** is `grounding/serialization/serializer.py`. Its output is a header line, an objects line, a
  pose line (`pose_kind: none` in the room data), the command line last, and for `coordinates_relations_v2` the relation
  tables. The A2.3a prompt wrapper and system message are in `grounding/inference/iref_vla/pilot-protocol.v1.json`.

## 8. Decisions D93–D99 and open items

- **D93** A1 closed for now.
- **D94** A2.3c.
- **D95** A2.3d.
- **D96** A2.3e.
- **D97** A2.3 closed, with the percentile rule and the sequence.
- **D98** A2.5 defined.
- **D99** A2.5's input and protocol decisions (committed in `168344e`; resolves O21 for A2.5).

Open items:
- **O3, O4, O7–O10:** infrastructure.
- **O13:** hand models.
- **O14:** graphics memory after headset off and on.
- **O16:** permissions for speech.
- **O19:** Gate A while speech waits.
- **O20:** Gate B thresholds.
- **O21:** typed input. Resolved for A2.5 by D99; a headset keyboard is deferred; input is not to be decided again.
- **O22:** the resolver's open choices.
- **O24:** ECMAScript labels.
- **O25:** MRUK permissions in the manifest.
- **O26:** inference-interval frame acceptance.
- **O27:** perspective coverage, pending.

## 9. Pending actions, in order

1. **A2.3e records follow-up** (section 0): append the corrected folder's full manifest hash and successful readback to
   `notes/phases/A2.3e_checks.md` section 5.
2. **A2.5 delivery 1 (replay).**

## 10. Lessons and gotchas

- **Keep old wording renderable.** When a report's rendering changes, keep the original wording for folders made
  earlier, or their readback fails (this repair). Test legacy folders with the old code's output, not a re-render.
- **Dictionary order and float sums.** Cross-version floating-point accumulation can change last bits. Preserve frozen
  artifacts and use an explicit accumulation/rounding policy or declared numerical comparison where sums are needed. Do
  not silently regenerate byte-pinned summaries on another Python version and expect identical hashes. Use explicit
  output ordering rather than incidental dictionary order.
- **Hashes and line endings.** Hashes are of bytes, and CRLF versus LF changes them; transfer the exact file a manifest
  pinned.
- **Selection rules must match.** A canary or case-selection rule copied into a second tool must match the first
  exactly (the audit nearly broke ties the other way).
- **The RTX PC's run configs** block pulls (section 3).
- **Missing output in a paste** usually means a long command was still running; check before committing.
- **Speed.** jsonschema over large arrays is slow; check those structures directly.

## 11. A good first message in the new chat

"Read the handover. Then start A2.5 delivery 1, frozen-request replay, under D98 and D99: the PC fixture and reference
bundle builder, the headset replay mode (uncached and cached), and the laptop D56 comparison."

Reports: the available A1 report is historical through A1.8; D91–D93 and the current handover carry the later detector
results and closure until the report is updated. `Second_Eyes_A2_Report.pdf` is current through 7 October evening,
including A2.3's closure, D98, D99 and `168344e`.
