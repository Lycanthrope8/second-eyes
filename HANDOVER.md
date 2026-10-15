# Second Eyes: handover (A2.7 closed, A2.8 pending)

**Second Eyes** is a research system. A drone pre-scans an indoor space; a Quest 3 headset then grounds typed commands
("the chair closest to the door") to objects in that scan.

This file replaces earlier handovers; Git keeps their history. The repository's records are the source of truth, not
memory or chat.

**Roles:**
- **ChatGPT** designs and reviews. The project lead delegated A2.4's design to it (D109).
- **Claude** implements, tests, verifies, writes the records and gives the runbooks.
- **The project lead** runs everything on two Windows machines, commits, and decides.

## 1. Where the truth lives

1. `notes/phases/A2.7_closing.md` and `A2.7_plan.md`: A2.7's closing, its plan and frozen criteria, and its deliveries.
   How it works: `docs/a27-export.md` and `docs/a27-headset.md`; the C# checks: `tools/csharp-tests/README.md`.
2. `notes/decisions.md`, D1–D119, newest first, with the open items (O28 and O29 the newest) at the top:
   - D119: A2.7 accepted and closed;
   - D115: A2.7's plan and frozen criteria; D116 to D118: its deliveries and corrections;
   - D111: phase numbering A2.7 and A2.8, and the lab commands' status;
   - D109 and D110: A2.4's design and resolver choices;
   - D108: A2.6 closed;
   - D97: the sequence;
   - D104: the request format.
3. `notes/failures.md` and `notes/limitations.md` (read the recent rows), `docs/conventions.md`, `grounding/README.md`.
4. A2.6's notes (`A2.6_closing.md`, `A2.6a`, `A2.6b`, `A2.6c`) and `docs/iref-vla-development.md`.
5. The project files: the Research Proposal Revision 3, the Revision 3.1 supplement, and the A1 and A2 reports. The A2
   report is a snapshot from 7 October, not the current status. It is updated when A2.4 closes.

## 2. State

- **A1** is closed for now (D93). On-device speech (A1.9) is optional.
- **A2.1–A2.3, A2.5 and A2.6** are closed (D97, D105, D108).
  - A2.5: the headset runs the base 0.5B Q8_0 through llama.cpp, on path U with the cache off. Its uncached and cached
    numerical acceptance remain **unresolved**.
- **A2.4 (LoRA fine-tuning) is closed** (D114): nine fits and nine development evaluations, reviewed and accepted. The
  fine-tuned direct models reach about 43% source-known and 30% to 32% full inventory, 5.7 to 12.6 points above the
  base 0.5B; the parser with the resolver reproduces the rules (93.2% and 16.8%). The run folder is
  `second-eyes-data\a24-20261012-032009`, its corrected analysis `analysis-v2`.
  - For long GPU runs on the RTX PC, set `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512`
    first (`notes/failures.md`, 2026-10-12).
- **A2.7 is closed** (D119), parser-only: the candidate `a24.parser.s1` (epoch 2), exported to GGUF Q8_0, runs on the Quest
  through libse_llama's constrained decoding and the C# resolution path. Every criterion of D115 is met on the headset:
  the deployed identity, 300 decodes identical to the desktop, the masks, 600 resolutions and 1,719 reader strings with
  their queries, technical failures apart; D105's discrepancy stays unresolved. Measured (`notes/phases/A2.7_closing.md`):
  about 1.23 s from a Preset to a validated decision, cold and warm; about 72 fps during request processing. Known
  issues: O28 (a session's start blocks the main thread), O29 (a first-result hitch), O14. Deployment fidelity only:
  not general language accuracy, nor automatic perception (`notes/limitations.md`).
  - **Next:** A2.8 stays pending (the sealed lab commands, O20's thresholds); the next app change fixes O28.
- **A2.8** (frozen evaluation and preliminary Gate B) is **pending** (D115). The lab commands are neither made nor
  sealed: the project lead makes and seals them once the whole system runs, and only then are they evaluated. Until
  then, development uses basic commands. A2.8 also needs O20's thresholds, agreed with the advisor. Never request or
  inspect the lab commands' contents.
- **The A2.6 review.** ChatGPT independently reviewed the original baseline evidence and accepted A2.6 with small
  corrections. The corrected `scores-v2` and `analysis-v2` are recorded at `0395a43`. A later repository check is not a
  fresh independent readback of those corrected external artifacts.

## 3. The A2.6 development benchmark: frozen, A2.4's before-and-after comparison

- **ScanNet only**, with an official-first partition by physical environment (`a26.partition.v1`):
  - training: 564 environments (1,199 scans);
  - development: 47;
  - calibration (45) and test (50): **closed**;
  - legacy `scene0010`: separate.
- **Hashes:**
  - specification: `9fa980010eb46f25bb471ede64a37b366d812adb654e23e8370edae450c538b0`;
  - request file: `dc725cfe7bfab1b80a468b748d12a8b21b5583f14176d43337a0835c1b6f03ea`;
  - inventory: `87bacea9455064dc20fc615a0b70a0c72a81caf95ad2c5ba1ec263a8d708f831`, in
    `inventory-20261010-153906`. The hash, not the folder name, is the identity.
- **Sample:** 2,651 development parents from 46 environments, plus 64 legacy parents. 10,860 requests: two views by
  two formats.
- **Baselines** (command-weighted target accuracy):
  - the 0.5B: 24–31%, at the always-B floor;
  - the 7B: 1–4% (ASK 90–96%);
  - the rules: 93.5% in the source-known view, 16.9% in the full inventory (39.1% environment-weighted).
- **Eligibility is shaped by the accepted parser:** no "above" or "in", and "near" only about 24% eligible.

## 4. Machines, environments, paths

**Laptop** (`C:\Users\jubay\second-eyes`, data `C:\Users\jubay\second-eyes-data`):
- Python 3.11.9 in `.venv`;
- the Quest 3 over ADB;
- the llama.cpp host build;
- the pinned llama.cpp source in `third_party\llama.cpp` (b11277), where the GBNF validator is built in Stage A.

**RTX PC** (`C:\Users\jhossai3\second-eyes`, data `C:\Users\jhossai3\second-eyes-data`):
- RTX PRO 6000 Blackwell (96 GB), 24 CPU threads;
- **`.venv`** (accepted evaluation, unchanged): Python 3.12.10, torch 2.11.0+cu128, transformers 4.57.6;
- **`.venv-train`** (A2.4 only): the same pins plus PEFT 0.21.0 and accelerate 1.15.0
  (`grounding/requirements-a24-train.txt`).

**Models:**
- 0.5B: `grounding\models\qwen2.5-0.5b-instruct\hf` (pinned `7ae5576...`);
- 7B: `...\qwen2.5-7b-instruct\hf` (`a09a354...`);
- the pinned 0.5B tokenizer: `quest-app\Assets\SecondEyes\Models\qwen2.5-0.5b-instruct`.

**Data** (RTX PC, under `second-eyes-data\iref-release\`):
- `zips\Scannet.zip`. ScanNet's terms: non-commercial research, **no redistribution**;
- `inventory-20261010-153906`;
- `spec-20261010-193912` (A2.6);
- `prep-20261010-211017` and `work-20261010-211017` (A2.6b). It took about 35 minutes for 104 scans with 12 workers,
  inferred from the folder's local time stamp read as EDT;
- A2.6c: `second-eyes-data\a26c-20261011-013341`, with `scores-v2` and `analysis-v2`.

**A2.4's outputs:** `second-eyes-data\a24-<stamp>`, named in `a24-current.txt`. Folders are never overwritten.

## 5. Working rules (learned the hard way)

1. **One consolidated delivery per step:**
   - a zip of the changed files;
   - a one-off records script (idempotent; line endings preserved; writes nothing if an anchor is missing);
   - the runbooks, committed with the phase.

   Small fixes are folded into larger deliveries.
2. **Runbooks:**
   - `. { … }` blocks that throw `STOP` on failure;
   - tests teed to a file and grepped for `FAIL`, `Traceback` or nonzero failures;
   - explicit `git add` lists;
   - Windows paths verified before use;
   - one block per machine;
   - an expected-output list after each block.
3. **Tests:** every change gets focused tests, run in the sandbox under Python 3.11 and 3.12. Fixtures come from the
   real chain where possible, and labelled doubles otherwise. Results are re-derived independently.
4. **Never overwrite outputs.** Long runs are resumable under identity locks. Estimate before starting, smoke-test
   before full runs, and pilot small.
5. **Windows:** worker processes are spawned, and transient file locks are retried.
6. **Boundaries:**
   - calibration, the project test and the final lab commands stay closed;
   - no Gate B claim and no Quest numerical-agreement claim;
   - annotations never enter model inputs;
   - the A2.6 benchmark's eligibility is never silently changed.
7. **Records:** errors are written honestly in `notes/failures.md` ("Claude's error" where it was). Decisions become
   D-rows.
8. **The project lead's standing constraints:**
   - propose first;
   - give options with a recommendation;
   - ask about inconsistencies rather than fixing them quietly;
   - one step at a time;
   - plan only up to the next gate;
   - every Quest computation counts;
   - Vicon and participants wait.
