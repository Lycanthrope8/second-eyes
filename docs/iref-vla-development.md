# The IRef-VLA release, the development preparation and the baseline execution (A2.6)

How the A2.6 modules work now. What happened when they ran is in `notes/phases/A2.6*.md`; decisions are D97 and
D104–D108 (A2.6), D109 (A2.4's training mode).

## 1. The release reader (`grounding/adapters/iref_vla/release.py`, `iref_vla_release.v1`)

A versioned extension of A2.2a's one-scene adapter, which it leaves unchanged.

- **Reading.** `read_source(zip, "Scannet", only=...)` reads a scene's four members (objects, regions, scene graph,
  statements) from the release zip, or from a folder. `inspect_scene` and `build_inventory` count scenes, texts,
  relations and coverage; `group_of` puts rescans and derived regions of one physical environment into one group.
- **Partitions.** `propose_partitions` assigns every group: official lists first (ScanNet's train, val and test, pinned
  by commit), then a salted hash (`a26.partition.v1`). The legacy group `scannet:scene0010` is separate. A2.6's frozen
  specification records the assignment; nothing recomputes it later.
- **Importing a scene.** `import_release_scene` runs A2.2a's import under the scene's release identity
  (`release_identity`), with A2.2a's whole validation, after checking the scene's files against the inventory's
  recorded hashes.
  - **By default** only development and legacy scenes are converted (`CONVERTIBLE`). Training, calibration, test,
    unassigned and unknown scenes are refused before anything is read.
  - **A2.4's training mode** (D109). A training scene is converted only under a `TrainingAuthorization`, built from
    three files' bytes: A2.6's frozen specification, A2.4's training specification and the inventory. The
    authorization refuses unless A2.4's specification records the other two files' hashes, and A2.6's records the
    same inventory. It permits a scene only when:
    - its environment (found by this module's own grouping rule) is assigned to training;
    - the environment belongs to A2.4's fitting or inner-validation cohort, or to an activated reserve batch (activated
      in the frozen order only);
    - A2.4's specification lists that scan.

    Calibration and test stay refused in this mode too. `read_training_source` checks every scene before opening any
    of its members. A training scene's provenance reads "A2.4 training material (the ... cohort of the frozen training
    partition)", never "development material". Development and legacy imports are byte-identical to A2.6's.

## 2. The development preparation (`grounding/preparation/iref_vla_dev/`, A2.6b)

- **`freeze`** fixes the partition, the cap (64 per environment), eligibility, sampling, the desktop subset and the
  analysis plan with its literal bootstrap seed, in one hashed `spec.json` (the accepted one is `9fa98001...`).
- **`prepare`** runs the accepted chain per scene in spawned workers: A2.2a's import, A2.2c's selection audit, A2.2d's
  materialization. It retries a transient Windows file lock (recorded) and never retries a data error.
  - **Eligibility:** A2.3a's rule, 3 to 10 objects in both views, and one target over all of a parent's annotation
    entries. Entries are resolved, never taken in order.
  - **Sampling:** round-robin over relation strata in name order, salted within a stratum. N and n are recorded.
  - **Requests:** each sampled parent gives four D104 requests (two views by two formats), built as A2.5's golden
    builder builds them.
  - The request file (`dc725cfe...`), sampling and desktop subset are hashed before any inference. The targets sit in
    `reference_only/`.

## 3. The baseline execution and analysis (`grounding/inference/iref_vla_dev/`, A2.6c)

- **`preflight`**, run once (in parallel), checks the frozen requests read-only against their bundles: prompts, mappings,
  token IDs and each model's tokenizer. Its receipt binds the frozen hashes, the preparation, every bundle, this
  folder's code and the tokenizer identities. A changed file there (for example A2.4's docstring correction in
  `analysis.py`) needs a fresh preflight; old receipts keep their original code identities.
- **`smoke` and `run`** bind to the passed preflight and give A2.3d's accepted runner its context. The runner reads back
  one row per frozen request, so a run covers all 10,860. Each step takes an identity function: the accepted
  `checkpoint_evidence` by default, A2.4's `identity.finetuned_evidence` for fine-tuned folders.
- **`rules`**: the accepted parser and resolver, once per parent and view.
- **`score`** reads the targets (only here) and gives each request and each rules row its outcome; always-B is a
  labelled supplementary diagnostic.
- **`analyze`**:
  - estimators: command-weighted (N/n), environment-weighted and unweighted;
  - the paired environment-cluster bootstrap: `random.Random(20261010)`, 10,000 replicates;
  - 95% intervals by the nearest-rank convention (D97);
  - legacy reported apart, as point estimates.

## 4. Commands

```text
python -m grounding.adapters.iref_vla.release --help        (fetch-samples, fetch-official-splits, inventory, propose)
python -m grounding.preparation.iref_vla_dev freeze|prepare --help
python -m grounding.inference.iref_vla_dev preflight|smoke|run|rules|score|analyze --help
```

Tests: `grounding/tests/test_iref_vla_release.py`, `test_iref_vla_dev.py`, `test_iref_vla_baseline.py`; A2.4's training
mode in `test_a24_prep.py`.
