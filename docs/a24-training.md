# A2.4: fine-tuning the grounder (`grounding/a24/`)

How A2.4 works now (D109–D111). What happened is in `notes/phases/A2.4_*.md`. Every setting is in
`grounding/a24/settings.v1.json`, frozen into the training specification by hash before anything is prepared or fitted.

## 1. Designs and configurations

LoRA on the pinned Qwen2.5-0.5B-Instruct (`7ae5576...`). The 7B stays A2.6's zero-shot reference. There are three
configurations, each fitted with three paired seeds (`20261011`, `20261012`, `20261013`), nine fits in all:

| Configuration | Learned input and output | Scene evidence at inference |
|---|---|---|
| `direct.coordinates_v2` | the D104 prompt, then one letter (A to J, or K) | the view's `coordinates_v2` document |
| `direct.coordinates_relations_v2` | the D104 prompt, then one letter | the view's relations document |
| `parser` | the command text, then a v1 query string | the same view, through the accepted resolver |

Model names are `a24.<configuration>.s<seed index>`.

The two direct formats share their rows, data order, initialization and exposure. The parser trains once per parent,
so its exposure differs, and the specification reports it.

## 2. The semantic output language (`language.py`, `language.v1.json`, `language.v1.gbnf`)

```text
TARGET | RELATION | ANCHOR            RELATION: closest N, farthest N (N = 1, 2, 3), near, below, on
TARGET | between | ANCHOR | ANCHOR    an object: CATEGORY, or CATEGORY [COLOUR]
```

**The vocabulary** is the pinned IRef-VLA `data/NYU_Object_Classes.csv` (commit `741cf02`, SHA-256 `3c65ff2e...`)
minus `unknown`: 893 labels, the category map's `model_vocabulary`. It is fixed before any scene is read and never
narrowed. Colours are the ten of the rules protocol.

**Why the colour follows the category.** Four labels begin with or equal a colour word ("orange juicer", "orange
plastic cap", "yellow pepper", "orange"), so a colour written first would make strings ambiguous. No label contains
`|`, `[` or `]`, so every accepted string has one reading.

**Outside v1:** directional relations, "above", "inside", size words, plurals, coreference, and ASK or unsupported
outputs.

**One specification, four uses:**
- the automaton the decoder walks;
- an independent reader (splitting and set membership);
- deterministic structural conversion to `grounding_query` (A2.2b's node layout);
- the GBNF for llama.cpp.

**What code supplies.** The query's identity fields, the action (INSPECT, the protocol's placeholder) and null frames
come from deterministic code, never from the model.

## 3. The constrained decoder (`decoder.py`)

**What the masks allow.** Every ordinary token is read as its bytes. A token is allowed when its bytes keep the
output a prefix of the language, so every valid tokenization stays possible. Special tokens are never allowed; the
end token `<|im_end|>` is allowed only at a complete string.

**Choice:** greedy, with ties going to the lowest token ID.

**Outcomes**, each distinct from ASK and from a resolver result:
- `completed`;
- `length_exhausted` (96 new tokens);
- `empty_allowed_set`;
- `premature_eos`.

A completed string is then validated by the independent reader and the accepted query validator. Nothing is repaired.

**Evidence:**
- an independently written oracle in the tests agrees with the masks on 21 prefixes over every ordinary token (frozen
  in `grounding/tests/fixtures/a24/mask-expectations.v1.json`);
- the pinned llama.cpp's own grammar validator (b11277, `test-gbnf-validator`) agrees on the test corpus (`gbnf_check.py`);
- `llama_sampler_init_grammar` is the later Quest route (A2.7).

This is language compatibility only: no Quest execution or numerical claim.

## 4. Labels (`labels.py`)

**The builder** maps a parent's annotation through a written table:
- `closest`, `second_closest` and `third_closest` become `closest 1`, `2` and `3` (likewise `farthest`);
- `near`, `below`, `on` and `between` map to themselves;
- anchors keep the annotation's order;
- a colour is kept only when the text used it.

**Refused parents** (each reason kept): a size word, `above` or `in`, a colour outside the ten, a class outside the
vocabulary, or entries that disagree.

**The checks** (none of them the builder):
- the annotation's consistency over entries;
- the accepted query validator;
- the accepted rule parser, as a diagnostic: a different structure quarantines the parent, which is an eligibility
  restriction and not a judgement of truth;
- per view: the target is offered, and the accepted resolver, given the label's query on that view's derived scene,
  resolves to exactly the target.

**What enters training.**
- The cohort is the parents that pass, with a supported source-known view.
- Each view whose own check passes is a direct-training row.
- One parser example per parent.
- No K (ASK) label is trained; K stays available at inference.

## 5. Preparation (`prep.py`)

**`freeze`:**
- inputs, by hash: A2.6's frozen specification (`9fa98001...`) and the inventory bytes (`87bacea9...f831`), plus the
  language, the settings and the code;
- the environments: every training environment, ordered by SHA-256(salt LF environment ID). The first 120 fit, the
  next 25 are the inner validation, and the rest is the ordered reserve;
- caps of 9 and 8 per environment, minimums of 900 and 180 parents;
- the reserve rule: a short cohort takes the next 10 reserve environments in order, at most four times; a remaining
  shortfall is reported.

**`prepare`**, per round:
1. Authorize the round's scans, then read them (section 1 of `iref-vla-development.md`).
2. Run A2.4's own wrapper over the shared import, audit and materialization in spawned workers, with A2.6b's lock
   retry.
3. Apply A2.6b's eligibility and the label checks in the same worker.
4. Sample per environment by A2.6b's round-robin, with the stratum order salted per environment (A2.4's sampling
   policy: name order favours earlier strata when a cap is smaller than the strata available).
5. Build D104 requests in both formats with A2.6b's `request_record`, and parser examples. Token boundaries are
   checked and nothing is truncated.

**Outputs:**
- counts by parent and by view, before and after filtering;
- `strata.jsonl` (eligible, retained and selected per environment and stratum, zero-selected included);
- `audit/` (every retained construction and every quarantine reason, up to three real parents each, with the
  model-visible objects);
- `pilot-spec.md`;
- a hashed manifest with UTC start and finish.

**Reusing an earlier attempt's scans** (`prepare --reuse-work FOLDER`, fix 1). When a preparation stops after its
scans are prepared, a new `prepare` can take the earlier work folder instead of `--work`. Each scan is reused only if
all twelve of its checks pass:
- its three stages are complete;
- today's importer reproduces its import byte for byte from the frozen source files (the inputs and the importer's
  code);
- the audit's outputs are unchanged, its recorded inputs are this import's, and its recorded code is today's;
- the accepted `verify_bundle` passes, and the materialization's recorded inputs, code, libraries, model description,
  tokenizer and runtime are this import's and today's.

A scan that fails any check stops the run before anything is published, naming the scan and the failed checks.
Nothing in the folder is changed, and scans the folder lacks are prepared fresh into it. Eligibility, labels, cohorts,
examples, the audit and the specification draft are always recomputed. The output's manifest and `scenes.jsonl` record
the reuse and every check per scan.

**Regenerating the report** (`report --prep FOLDER --out FOLDER`, fix 2). This recomputes the counts and
`pilot-spec.md` from a saved preparation into a new folder. It verifies the preparation's files against its manifest
first, records the hashes of the report it supersedes, and reruns nothing; the preparation is never changed.

An environment "contributes none" when its retained parents sum to zero. It falls in one of two groups: no parent
eligible under A2.6b, or eligible parents that were all quarantined.

**Re-exporting the audit** (`audit-export --prep FOLDER --work FOLDER --out FOLDER`, fix 3). This re-exports the
same audit examples from a saved preparation and its work records into a new folder. Each view's objects carry the
scene contract's geometry fields (`center_m`, `size_m`, `rotation_xyzw`) exactly as recorded, with their state and
full precision; `audit.md` rounds them to the centimetre for display only.

Each view links by SHA-256 to its original derived scene and command records, and each parent to its source
annotation entries. All three are stored once each under `evidence/`. The export is refused if a derived record
differs from its bundle's index, or a letter from the original audit's. Its manifest records the preparation's
identity, the superseded audit's hashes, and that the specification, cohorts, labels and six training-data files are
unchanged.

**Regenerating an analysis** (A2.4's closure). `analyze --supersedes FOLDER` writes a new analysis from the saved
scores and records the hash of the analysis it replaces, which is kept unchanged. `analysis-diff --old FOLDER --new
FOLDER` compares two analyses, A2.6c's or A2.4's. It reports each interval endpoint's change in percentage points, any
interval whose crossing of zero changed, and any other number that changed. The 95% interval takes exact tail
probabilities (1/40 and 39/40): the sorted replicates at indices 249 and 9,749 of 10,000.

## 6. Training (`train.py`, in `.venv-train`)

**A fit:**
- float32, eager attention;
- LoRA with r 16, alpha 32, dropout 0.05, on the seven projections;
- AdamW at 2e-4, with linear warmup over 5% of steps, then decay to 0;
- 16 examples per optimizer step, 3 epochs;
- the response-only loss, computed at the supervised positions only.

**Resume** is exact. The state (adapter, optimizer, scheduler, RNG, data position) is saved every 20 steps and at each
epoch's end. `lock.json` refuses a resume with different inputs, code, software or settings, and keeps the partial
evidence.

**Each epoch's end:**
- a hashed checkpoint;
- the inner validation: unweighted over the retained validation rows (direct: the accepted decision rule; parser:
  constrained decoding, then the resolver per retained view).

**`select`:** the best metric; a tie goes to the earlier checkpoint.

**`merge`:**
- merges in float32;
- checks adapter-loaded against merged on 16 direct rows (offered logits within 0.001, same decisions) or 8 parser
  parents (same tokens);
- copies the pinned tokenizer files;
- writes `a24-identity.json` (section 7).

**`continuation`** checks the mechanical conditions after seed 1, never accuracy.

**`smoke`** (Stage A only) runs three steps per configuration, an interrupted-and-resumed copy, validation, merge and
identity, then extrapolates measured times and peak GPU memory to Stage B.

## 7. The fine-tuned identity (`identity.py`)

**What `a24-identity.json` binds:**
- the base and its accepted evidence;
- the pinned tokenizer files;
- the specification, settings, preparation and data hashes;
- the fit's lock (seed, code, software);
- the selected checkpoint and its rule;
- the adapter files;
- the merge (procedure, software, check);
- every merged file.

**`finetuned_evidence`** is the accepted runner's evidence function for these folders:
- it re-hashes every file and refuses a changed, missing or extra one;
- it checks the tokenizer and architecture;
- its tie names the model.

The base model's `checkpoint_evidence` is unchanged. A local hash identifies a local artifact, not a published
checkpoint.

## 8. Evaluation (`evaluate.py`, in the accepted `.venv`)

After all nine fits and selections, one complete frozen development evaluation per selected checkpoint:

- **Direct:** A2.6c's preflight-bound `smoke` and `run`, with the fine-tuned identity, over all 10,860 requests.
  `model.json` binds the run to the model's name and identity. The trained format is the primary comparison; the
  other format's rows are a predeclared, labelled cross-format transfer analysis.
- **Parser:** one decode per parent (2,715), validated, then resolved in both views from that one decode.
- **Score:** A2.6c's `score`, with the base 0.5B and 7B re-scored from their saved runs; the parser by the rules'
  outcome rule.
- **Analyze:**
  - A2.6c's estimators and bootstrap, imported unchanged;
  - per seed, the seeds' mean, and the seed spread reported apart from environment uncertainty;
  - paired differences;
  - legacy apart;
  - the predeclared supplementary breakdown by the rules' saved per-row outcome.

## 9. Environments and commands

`.venv-train` holds PEFT 0.21.0 and accelerate 1.15.0 on the evaluation environment's pins
(`grounding/requirements-a24-train.txt`). The accepted `.venv` is unchanged.

```text
python -m grounding.a24 freeze|prepare|gbnf-check|smoke|fit|select|merge|continuation      (.venv-train)
python -m grounding.a24 prepare ... --reuse-work FOLDER   (instead of --work: verify, then reuse, an earlier attempt's scans)
python -m grounding.a24 report --prep FOLDER --out FOLDER  (the counts and pilot-spec.md regenerated, nothing rerun)
python -m grounding.a24 audit-export --prep FOLDER --work FOLDER --out FOLDER  (the audit re-exported with full geometry)
python -m grounding.a24 analyze ... --supersedes FOLDER    (a new analysis from saved scores; the earlier one kept)
python -m grounding.a24 analysis-diff --old FOLDER --new FOLDER  (what changed between two analyses)
python -m grounding.a24 locate|eval-direct|eval-parser|score|analyze                         (.venv)
```

Tests (sandbox; stand-in models labelled):

```text
python grounding/tests/test_a24_language.py --samples DIR [--gbnf-validator PATH]
python grounding/tests/test_a24_prep.py --samples DIR --official DIR
python grounding/tests/test_a24_train.py --samples DIR --official DIR
```

The runbooks are `notes/phases/A2.4_runbook_stage_a.md` and `A2.4_runbook_stage_b.md`.
