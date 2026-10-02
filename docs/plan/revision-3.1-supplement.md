> **Repository copy.** The Revision 3.1 supplement's text, verbatim, as exported to Markdown from its Claude Doc
> (https://claude.ai/artifact/QSRm7JDYsRaeu2ba9DscB8, revision 17) on 2026-10-01, and approved by the project lead
> (D63). The PDF of the same revision is in the Claude project's files:
> `Second_Eyes___Revision_3_1_supplement_A2_and_the_sections_it_touches.pdf`, SHA-256
> `2dbeda00465e2dc4ebf406bcd56b872d34ea1ca5121dbca0fad182bcd26518f0`. Not edited here: a change comes as a new
> revision (`docs/conventions.md`, rule 11). The one addition is Figure 5's text, marked as such under its placeholder.

---

# Second Eyes · Revision 3.1 supplement: A2 and the sections it touches

Oct 1, 2026 · @Jubayer Hossain

## About this supplement

This supplement replaces Revision 3's A2 phase card and every passage A2's changes touch; all other sections of Revision 3 stand unchanged. Sections follow Revision 3's numbering, and each says what it replaces or adds.

It reflects the project lead's decisions of 1 October 2026:

- A2 uses typed text commands only.
- A2 comes first, including its grounding-only headset checks. The detector and combined feasibility tests follow; on-device speech comes last.
- Section 7's study design is unchanged, speech conditions included.
- SpatialLM reaches the headset only if it passes Gate D and a compressed version then passes the headset budget.
- Published datasets are A2's main training source; no custom scene generator is built.
- Natural lab commands are used for evaluation only, never for training or selection.

**A1's step numbers.** This supplement refers to A1's remaining steps by number: A1.9 is on-device speech, A1.10 the object detector, and A1.11 everything together with the 30-minute soak. The Phases section lists all eleven.

**Claim changes.** Passages that narrow or reframe a Paper 1 claim end with a line marked *Claim change*.

## Revision summary and Executive Summary

*Replaces the Revision 3 description in the "Data" row of the What Changed table:*

Published spatial-reference datasets are A2's main training source, beginning with IRef-VLA; compatible datasets are added where measured gaps appear. A small set of natural lab commands is held out for evaluation. No custom scene generator is built.

*Adds to the "Gates and logistics" row, and to the end of the Executive Summary:*

Implementation is staged. A2 uses typed commands and includes grounding-only headset checks. The detector and combined feasibility tests (A1.10, A1.11) follow, and on-device speech (A1.9) comes last. The planned interaction study is unchanged.

## 1.2 Central claims

*Adds after the claims table:*

A2 provides evidence for the grounding component only. Its typed-command measurements do not establish Claim B for the full system; that still needs the detector, combined and speech tests. Claims A to C are otherwise unchanged.

## 2 Prior work and novelty boundary

### 2.1 Where the individual pieces already exist

*Adds to the end of the "Grounding language in 3D scenes" paragraph:*

Transcrib3D \[44\] already resolves 3D referring expressions through text descriptions of detected objects, including fine-tuning smaller models for local use. SORT3D \[45\] combines language models with a spatial reasoning toolbox for grounding and navigation. VLMaps \[46\] queries language-indexed spatial maps for navigation goals. REVERIE \[47\] grounds remote objects in navigation instructions, and SCOUT \[48\] studies dialogue with a remote robot exploring an unknown space. Second Eyes claims none of these components; it evaluates their integration under a wearable runtime budget, for targets the person has not seen.

### 2.2 Claim-by-claim novelty map

*Adds these rows:*

| Candidate claim | Closest prior work | Status |
| --- | --- | --- |
| Text descriptions connect 3D objects to language-model reference resolution; smaller models adapted for local use | Transcrib3D \[44\] | Established; not claimed |
| Language interpretation combined with deterministic spatial operations | SORT3D \[45\] | Established; not claimed |
| Stored spatial maps queried through language for navigation goals | VLMaps \[46\] | Established; not claimed |
| Language-guided navigation to a remote object | REVERIE \[47\] | Established task family; remote from the robot does not mean never seen by the person |
| Dialogue with a remote robot exploring an environment | SCOUT \[48\] | Established interaction setting; not claimed |
| C3: resource-bounded comparison of direct selection, query parsing and scene formats on a consumer MR headset | \[44\], \[45\], existing 3D grounding benchmarks | Paper 1 empirical systems contribution; the model's advantage is a hypothesis |
| Drone-acquired memory on the person's headset supports references to targets the person has not inspected | The five works above, plus the MR and drone precedents | Proposed system and evaluation contribution, subject to a final novelty check; A2 alone does not establish it |

### 2.3 Defensible contribution

*Keeps the paragraph and adds:*

This claim concerns the combined system and its controlled evaluation of information asymmetry. Text-based 3D grounding, spatial reasoning tools, map-based language navigation and remote exploration are prior art \[44\]–\[48\]. A2 supplies component evidence; it does not show the hidden-target outcome.

*Claim change:* C3 becomes an empirical accuracy, resource and design comparison, not an invention claim for grounding language with geometry (see 13).

## 3 Research questions and hypotheses

*Replaces RQ2:*

**RQ2 · Headset grounding.** Given the same structured scene evidence, how accurately and efficiently can a sub-billion-parameter model on the Quest ground unseen natural text commands, compared with a rule-based parser and a larger off-device reference model? Does direct target selection or query parsing with deterministic resolution give the better trade-off between accuracy and resources?

*Replaces RQ4:*

**RQ4 · Representation.** Under a fixed scene-information contract, how do coordinate and explicit-relation descriptions affect grounding accuracy, token cost, caching and measured headset latency? Whether richer learned structure improves downstream task success remains Track B's separate question.

*Replaces H2:*

**H2.** A 0.5B model fine-tuned mainly on compatible published spatial-reference data will outperform the rule-based parser on held-out natural lab commands and approach the larger reference model's target accuracy, under equivalent scene evidence. Gate B's numerical targets need the advisor's agreement before evaluation.

*Adds after the hypotheses:*

A2 tests typed commands. RQ3, H3 and section 7's speech-based study are unchanged, and H4's full-pipeline claim waits for Gate A's evidence.

*Claim change:* RQ2 and C3 are bounded by the supported commands and equivalent evidence; H2 stays a falsifiable hypothesis, not an expected success.

## 4 System architecture

### 4.1 and Figure 3

*Adds to section 4.1 and to Figure 3's caption:*

The figure shows the intended study system. In A2, typed text and the relevant scene and pose context go straight to the grounding interface; speech recognition is integrated last.

### 4.2 Division of labor

*Replaces the section's last sentence ("That keeps the model on the part of the problem it does well at this size… exact and instant."); the sentence before it, that the model never outputs distances or angles, stays:*

The two grounding designs differ in whether the model selects the target or supplies a structured query for deterministic resolution. Geometry stays exact but is not free: relation and goal computation count toward the measured command time.

### 4.3 Semantic memory

*Adds at the end:*

A2.1 fixes the exact identity, class, geometry, unknown-value and frame conventions, and training adapters and perception sources map to that contract. Provenance and observation history can stay in metadata outside the prompt. Relations are derived from the available geometry and the command-time frame; a dataset's own relation labels are never shown to the model as answers.

### 4.4 Hardware roles

*Replaces the RTX 6000 Pro workstation's role:*

Published-data preparation, limited gap-filling supervision, fine-tuning, the larger reference model, the offline SpatialLM work, and offline analysis.

### 4.5 Runtime budget

*Adds at the end:*

A2 measures the grounding path inside the Quest app, from typed submission to validated goal. It reports resident and peak memory, prompt and cache costs, scoring or decoding, and the deterministic work. Direct selection scores the candidates without first generating an answer. These measurements reserve no capacity for the detector, speech or a SpatialLM student; combined feasibility is tested afterwards.

## 6 Language grounding and goal execution

### 6.2 and 6.3 Output and grounding designs

*Replaces 6.2's confidence paragraph, and adds to 6.3:*

A2.1 fixes each design's constrained output. Direct selection chooses among permitted target IDs; query parsing fills supported semantic fields for the shared resolver. Invalid IDs and malformed queries are rejected. Candidate scores rank the candidates but do not guarantee that a valid, unique target exists, so non-selection is handled separately (6.6). A parser's token probabilities are not target confidence.

Training on object references alone does not teach an action vocabulary, so the INSPECT example stays illustrative. Both designs get equivalent evidence, and neither is chosen in advance.

*Claim change:* action understanding is claimed only for actions that are explicitly supervised and evaluated.

### 6.4 Describing the scene to a small model

*Replaces the section:*

A2 starts with two description families, both computed from the same scene evidence: compact room-fixed coordinates, and explicit relations. Zero-shot tests on development data shortlist serializations before fine-tuning. A small matched fine-tuning comparison of the two best then tests whether their order changes after training. Wider format sweeps are not needed for A2.

Pose-dependent descriptions use a pose snapshot taken when the text is submitted. A cached scene is reused only while the scene and the relevant pose stay valid. Relation construction, cache refresh, token processing and resolution all count toward command latency, and results report accuracy against measured headset latency and memory. RQ4's richer-structure comparison stays in Track B.

### 6.5 Relation definitions

*Replaces the introduction and the draft table:*

A2.1 defines and versions the supported relations before implementation and training. It separates directions anchored on the viewer's line of sight ("left of the table from where I am") from directions set by the user's heading ("on my left"). Drone-relative and intrinsic-object references are separate capabilities needing their own pose or semantic-front evidence; a box's geometric yaw alone does not give an object's front.

Candidate relations are behind and in front, left and right, beside and near, between, nearest and farthest, and vertical relations. Above and below are considered separately from on, which needs support and contact, not just greater height. Each supported relation states its required fields, frame, geometric test, tolerances and boundary behavior; missing evidence and ties stay explicit.

The implementation is checked against independently reviewed cases, because one function cannot both generate and verify its own labels. Published relation labels keep their original meaning in benchmark reporting; incompatible cases are filtered and documented.

**Revision 3's draft definitions, superseded.** Kept for the record:

| Relation | Frame | Draft definition |
| --- | --- | --- |
| behind(X, A) | user | X is farther from the user than A and within ±30° of the user-to-A direction |
| in\_front\_of(X, A) | user | X is nearer to the user than A and within ±30° of the user-to-A direction |
| left\_of(X, A) | user or drone | X lies to the left of A in that viewer's horizontal frame, lateral offset above 0.2 m |
| beside(X, A) | none | The footprint gap between X and A is below 0.5 m |
| between(X, A, B) | none | X lies within 0.3 m of segment AB and projects inside it |
| nearest(X, class, R) | none | X is the object of that class closest to R |

The draft fails on Revision 3's own example memory (4.3). At table\_1, its ±30° cone reaches 1.29 m to each side, about 2.6 m across. So chair\_1, 0.78 m to the side of the table and only 0.09 m deeper, counts as behind it; chair\_2, 0.76 m to the other side, counts as in front of it.

*Retains the elicitation commitment:* elicitation annotations can later test these definitions against how people use the words. Changes are versioned and never alter a locked test after the fact.

### 6.6 Ambiguity

*Replaces the section:*

The system separates four outcomes: one supported match, several matches, no match, and insufficient or unsupported evidence. Answerability is not inferred from normalized candidate probabilities alone. A2.1 decides whether non-selection is an explicit ASK output, a resolver status, confidence-based rejection, or a combination, with its calibration and threshold rules.

Calibration uses a separate public-data partition, never the held-out lab commands. A2 reports non-selection through the text and debug interface without a new dialogue policy, and the study's voice and tap clarification is unchanged. Wrong selections, unneeded clarification on answerable commands, and failures to reject unanswerable ones are reported separately. This stays Paper 1's simple mechanism; Paper 2B's ask-act-look policy is not part of A2.

### 6.8 Speech stays on the device

*Adds at the end:*

Speech remains part of the study system but is the last implementation item. A2 uses typed commands only; A1.10 and A1.11 follow A2, and speech integration follows those tests. End-of-speech latency is measured once speech is present.

Section 7 is unchanged.

## 9 Technical evaluation plan

### 9.1 Baselines

*Replaces the description of A2's baselines:*

A2 first compares three systems: rules with the shared resolver, zero-shot Qwen2.5-0.5B-Instruct, and a larger model on the RTX as an empirical reference. The larger model is not assumed correct or a strict upper bound. Fine-tuning then compares direct selection and query parsing under matched formats and data.

All systems get equivalent scene information and candidate sets chosen without the target. A2 uses annotated geometry to isolate grounding; results with estimated perception are a separate evaluation. A smaller model stays a fallback for resources, not a required first sweep. The SpatialLM comparison stays in Track B and never blocks A2.

### 9.2 Data

*Replaces the section:*

A2's training uses published spatial-reference data, beginning with IRef-VLA \[49\]. Compatible Sr3D, the released viewpoint generator \[50\], filtered Nr3D and ScanRefer, and Multi3DRefer \[51\] are added for specific gaps the results show. No custom scene generator is built; limited templates may fill documented residual gaps.

Adapters normalize geometry and identifiers, keep annotation provenance, and retain only references the available fields and frames support. Regions are cut from large rooms without using the labeled target; examples that would lose a needed distractor or anchor are rejected and counted. Physical scenes, rescans and derived variants stay together across splits, including scenes shared between datasets. Adapted subsets are named and reported apart from the original benchmarks, and dataset access is obtained before use.

A small set of natural lab commands is reserved for evaluation and never used for training, selection or calibration. The A3 elicitation corpus and recorded pre-scan replay keep their roles, and the A2 lab set never joins A3's training data.

*Claim change:* a custom synthetic training corpus is no longer a deliverable; results describe performance on the retained, documented subset.

### 9.3 Stratified test set

*Replaces the section:*

A2's tests are stratified by relation and frame, composition, superlatives, natural phrasing, candidate count and same-class distractors. They include explicit perspective checks, and separate one-match, several-match, no-match and missing-evidence cases. Negation, ordinals, vertical relations and references to earlier turns count as supported only if A2.1 supports them; otherwise they are unsupported cases that require non-selection.

Multiple-target references are valid, not annotation errors: their target sets are kept, and the single-goal policy is evaluated without picking an arbitrary first target. Public benchmark, adapted-subset and lab results stay separate. Speech-recognition noise is tested later, with speech.

### 9.4 Metrics

*Adds to the Grounding row:*

Accuracy over all uniquely answerable commands is reported apart from accuracy among accepted selections, with selection coverage, wrong selections and clarification behavior by answerability. A clarification on an answerable command is not a correct target. Parse fields and resolved targets are scored separately.

*Replaces the latency wording in the Headset row:*

From typed submission to validated goal for A2, and from the end of speech once speech is integrated. Stage times, cold and warm conditions, median and 90th percentile, with the active components named.

## 10 Execution plan

*Replaces the execution text that A2's new place affects:*

A1's language runtime (A1.1–A1.8) is established, so A2 runs before the remaining feasibility work. The order is A2.1 to A2.6, including grounding-only checks on the Quest; then A1.10, the detector; then A1.11, everything together with the 30-minute soak. On-device speech, A1.9, comes last, and must be in place before the pilot study (A9), whose spoken conditions need it.

How Gate A is judged while speech waits is a project decision, agreed with the advisor and recorded before A1.11. Revision 3's calendar is no longer a reliable estimate for A2: its length depends mainly on dataset access, so it is re-estimated in A2.2. Later phases and Track B keep their scope and gates; this supplement does not reschedule them.

&#91;embedded content: Figure 5 · order of work after this supplement\]

> **Added in this repository copy:** Figure 5's text, transcribed from page 10 of the PDF; the Markdown export keeps
> only the placeholder above.
>
> *A2 now runs before the remaining A1 feasibility work* · Track A · critical path, left to right:
>
> 1. **A1.1–A1.8** · Language runtime on the headset · Done
> 2. **A2** · Grounding with typed commands · A2.1–A2.6 · Prelim. Gate B
> 3. **A1.10** · Object detector
> 4. **A1.11** · Everything together and 30-minute soak · Gate A evidence
> 5. **A1.9** · On-device speech · Last
>
> From A1.11, a dashed arrow to **A3–A8** · Scope and gates unchanged (§10) · Timing follows the Gate A decision. From
> A1.9, an arrow to **A9** · Pilot study · needs A1.9.
>
> **Track B · SpatialLM offline on the RTX** · Never blocks Track A · a headset student only after Gate D and the
> headset budget
>
> Figure 5 · order of work after this supplement

*Figure 5, replacing Revision 3's.* A2 now follows A1's language runtime directly. The detector and combined tests come after it, and speech comes last, before the pilot study. Dates are left out until A2.2 re-estimates them.

*Replaces A2's row in the phase table:*

| Phase | Work | Exit condition |
| --- | --- | --- |
| A2 Grounding baselines | Scene and relation specification and implementation; published-data access and adapters; rules and zero-shot format screening; matched fine-tuning of selection and parsing; grounding-only Quest checks; held-out natural-command evaluation | Preliminary Gate B, pending the advisor's agreement on thresholds |

*In Figure 5 and the phase map:* A2's dependency becomes "the A1 language-runtime path", not all of A1. Completed Gate A is no longer drawn as a prerequisite for A2.

## 11 Go/No-go gates

### Gate A · Headset runtime

*Keeps the criteria and adds:*

A2 reports grounding-only measurements with typed commands and claims no Gate A pass. How Gate A is judged while speech waits is a project decision, agreed with the advisor and recorded before A1.11. One option is a typed-command part with a reserved speech allowance, then a speech part. Typed-submission latency is never relabeled as end-of-speech latency, and grounding-only memory is never reported as combined memory.

### Gate B · Language-model value

*Replaces the criterion:*

Proposed targets, pending the advisor's agreement: the fine-tuned model reaches at least 90% target accuracy on held-out natural commands, at least 10 percentage points above the rules, and within 5 points of the larger reference model, under equivalent scene evidence. A2 makes a preliminary assessment on its reserved lab commands, with public-data results reported separately.

Denominators, exclusions and how clarifications are scored are fixed before evaluation. Each criterion is reported separately, with its uncertainty. Unapproved criteria stay pending, and unmet ones stay unmet. If the rules match or beat the model, that is reported as a finding, and C3's model-value claim is reconsidered using the existing rule-based fallback.

### Gate D · SpatialLM value

*Clarifies the row; the threshold is unchanged:*

At least 10 percentage points improvement in hidden-target success on the specified relation- and layout-heavy set. Passing Gate D justifies compression work; only a compressed version that then passes the headset budget can enter deployment. This remains a condition, not an A2 task.

*Claim change:* A2 establishes at most a preliminary language-model value; it cannot establish hidden-target or full headset feasibility.

## 12 Risks and fallbacks

*Adds these rows:*

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Dataset access or source geometry is delayed | High | Start requests when A2 begins; use a compatible subset that is already accessible; check each source's access route separately |
| A description needs appearance, layout, viewpoint or semantic-front evidence the scene lacks | Medium | Filter and count it; never silently rewrite it or infer the missing evidence |
| Cutting rooms makes references easier or leaks the target | High | Build regions without the labels; keep needed distractors and anchors; reject what can't be kept; report the selection bias |
| Shared scenes or derived variants leak across datasets | High | Group by physical environment and ancestry before assigning splits |
| Dataset relations and the resolver disagree | Medium | Keep the original benchmark meaning; check compatibility independently; version the adapted task |
| High candidate scores hide an absent or ambiguous target | High | Apply the non-selection policy; report errors and coverage separately |
| A format's zero-shot rank fails after training | Medium | Compare the two shortlisted formats in a small matched fine-tuning run |
| Public-data success fails on natural commands | High | Reserve the lab set for evaluation; report transfer separately |
| Export, 8-bit rounding or longer prompts change predictions or costs | Medium | Run the reproduction check and grounding-only headset measurements across the intended range |
| Grounding fits, but the whole application does not | High | Keep component and combined resource claims apart; keep the later feasibility gates |

*Replaces the IRB-delay mitigation:* submit the existing protocol as planned. The A2 lab evaluation does not substitute for the approved A3 or section 7 studies, and its reserved commands are never repurposed for development or training.

The speech risk stays, tested when speech is implemented last. The rule-baseline risk stays, with strong rules reported as a finding.

## 13 Contributions and publication plan

*Replaces C3:*

**C3 · Grounding under a headset resource budget.** An empirical comparison of a sub-billion-parameter on-device grounder, a rule-based parser and a larger off-device reference model, under equivalent scene evidence. It tests direct target selection against query parsing with deterministic resolution, over shortlisted scene descriptions. It reports target accuracy, clarification behavior, and measured Quest latency and memory. Published-data training is evaluated separately from transfer to held-out natural lab commands. The model's advantage remains a hypothesis, and action claims are limited to explicitly supervised and evaluated actions.

*Adds to Claims to avoid:*

- First text-mediated 3D grounding, or first robot system combining language with spatial tools.
- A dataset contribution from converting published annotations.
- A Gate A pass from grounding-only tests.
- Speech robustness from typed commands.
- Hidden-target flight success from public grounding accuracy.

*Claim change:* C3 becomes a measured systems and design contribution. C1, C2 and C4 keep their claims and evidence requirements.

## Appendices A and B

### Appendix A · Immediate A2 checklist

*Replaces the Immediate Two-Week Checklist, retitled so its old heading no longer implies a duration:*

1. Record the established A1 language runtime and the outstanding feasibility items, keeping their run evidence.
2. Start the required data-access requests, and write the A2.1 scene, relation and evaluation specification.
3. Implement and independently check the serializer, validator, relation library and resolver.
4. Import a small compatible IRef-VLA subset; validate scene splits, IDs, geometry and the target-independent region rule.
5. Collect and reserve the small natural lab-command evaluation set, kept away from training and from model or format selection.
6. Run the rules, zero-shot Qwen and the larger reference on development data; shortlist formats and check perspective coverage.
7. Run the bounded matched fine-tuning pilot; add data only for gaps the results show.
8. Verify the export and measure the grounding-only path on the Quest.
9. Freeze the system and assess preliminary Gate B, reporting unresolved or failed criteria as they are.

After A2, A1.10 and A1.11 resume, and speech is the last implementation item. The study, Vicon and Track B keep their scope and gates; they are not immediate A2 tasks. Dataset access and existing ethics obligations are not postponed by this checklist.

### Appendix B · Assumptions to validate early

*Replaces or annotates these rows; hover-pose, study and SpatialLM assumptions are unchanged:*

| Assumption | Revised test or status |
| --- | --- |
| The chosen checkpoint runs on the headset | Established for the base model with llama.cpp; correctness and resources are rechecked for each fine-tuned export (A2.5) |
| Candidate scores and caching are usable | Established for the base model; rechecked for the adapted format and model |
| One model call per command is fast enough | Measured in A2 from typed submission to validated goal, deterministic work included; the speech-inclusive time is measured after speech arrives |
| Published labels fit the scene contract | Geometry, IDs, required attributes, frames and answerability checked before training; retained and rejected counts reported |
| Large rooms can be cut to the initial scene size without leaking the answer | The target-independent region rule is tested; cases whose meaning or context can't be kept are rejected |
| Public-data performance transfers to the application | The reserved lab commands and perspective cases are evaluated separately |
| Speech recognition runs fully on the device | The original offline test stays; it runs last, not as an A2 dependency |
| The full pipeline fits the Quest | Grounding-only evidence kept separate; detector and combined tests resume after A2, with Gate A judged as decided (11) |

## Phases

### A1 · Headset feasibility

*Adds to the A1 card:* the language runtime enables A2. A1.10 and A1.11 resume after A2, and A1.9 comes last. Intermediate results name their active components, and full Gate A status follows the Gate A decision (11). A1 works through these steps:

| Step | What | Status |
| --- | --- | --- |
| A1.1 | Headset baseline | Done |
| A1.2 | Headset cleanup | Closed with no changes: the busy processes are the XR system itself |
| A1.3 | App with passthrough, overlay, 72 Hz | Done |
| A1.4 | Event log on the headset, log pull | Done |
| A1.5 | Hand tracking and stop button | Done |
| A1.6 | Cost of the empty app | Measured within A1.7 |
| A1.7 | Language model with Meta's runtime (path A) | Done: correct but too slow; kept for comparison |
| A1.8 | Token probabilities and caching; llama.cpp (path B) | Done: llama.cpp is the runtime |
| A1.9 | On-device speech recognition, push-to-talk | Last |
| A1.10 | Object detector on passthrough | After A2 |
| A1.11 | Everything together, 30-minute soak | After A1.10; Gate A evidence |

### A2 · Grounding baselines (replaces the card)

**When** After A1.8, before A1.10 and A1.11. Revision 3 placed it in weeks 2–4; its length depends mainly on dataset access and is re-estimated in A2.2.

**Depends on** A1's language-runtime path, with candidate scoring and scene caching. Not on the detector, speech, a formal Gate A pass or Track B. Dataset access and format compatibility are checked before a source is committed.

**Ends at** Preliminary Gate B.

**Goal** Measure whether a sub-billion-parameter model on the Quest resolves natural text commands against a structured scene memory accurately and efficiently, compared with rules and a larger off-device model. Establish the value and cost of direct selection against query parsing, with published data as the main training source. Grounding is evaluated apart from perception, speech and flight.

**How**

1. **A2.1 · Specify the contract, then implement it.** First a short specification: required and optional scene fields; units, axes and transforms; identity and class mapping; when the pose is captured; and the supported command and output grammar. It separates "left of the table from where I am" from "on my left", states which user, drone and object frames are supported and what each needs, and decides vertical relations. It sets thresholds, boundary uncertainty, ties, and the outcome for one, several or no matches or missing evidence. Then the validator, deterministic serializer, relation library and shared resolver are built. They are checked with small, independently reviewed fixtures: the a17 example, viewpoint changes, duplicates, boundary cases and missing anchors. Fixtures are tests, not a training corpus. The policy and evaluation rules are frozen before the main comparisons.
2. **A2.2 · Obtain published data and build a small adapter.** Access requests start with A2, including ScanNet where the chosen files need it. Start with a small IRef-VLA subset \[49\], reusing its object records and annotations; normalize geometry; keep source IDs, targets, anchors, distractors, provenance and splits. Keep only references whose meaning the model-visible fields support. The region rule never uses the labeled target, keeps what a reference needs, and rejects examples that can't fit the initial 3–10 objects. Scenes, rescans and derived regions stay together across splits and datasets. As gaps appear, add compatible Sr3D, the viewpoint generator \[50\], filtered Nr3D and ScanRefer, or Multi3DRefer \[51\]; templates only for documented residual gaps. Begin collecting 50–100 natural lab-member text commands, with scene and pose context and independently checked targets. They are held out, never used for training, prompt or model selection, or calibration.
3. **A2.3 · Compare baselines and shortlist formats before fine-tuning.** Rules with the shared resolver, zero-shot Qwen2.5-0.5B-Instruct, and a larger model on the RTX as an empirical reference. All get equivalent scene evidence and candidate sets, never the targets, anchors or relation labels. Two format families, built from the same geometry and poses: compact room-fixed coordinates, and explicit relations. Check accuracy, token cost, relation-construction cost and caching on development data, plus a small perspective check from existing viewpoint resources. Shortlist two formats for A2.4; zero-shot rank is not evidence of post-training rank. The lab commands stay closed.
4. **A2.4 · Fine-tune both designs in a bounded comparison.** LoRA on compatible published examples, for direct selection and for query parsing with the shared resolver. Direct selection uses the target labels; parsing uses trustworthy relation, anchor and frame annotations where they exist, with their coverage reported. A pilot of about 1,000 examples checks the pipeline and transfer to unseen development scenes. The two shortlisted formats are then compared with matched examples, splits and budgets for each design. Format and design are chosen on grounding quality and expected headset cost. About 20,000 examples is a provisional next budget, used only when development failures call for more data. Partitions for training, development, calibration and test stay separate. Action supervision is recorded separately: object-reference labels don't show multi-action understanding.
5. **A2.5 · Verify the exported grounder in the Quest app.** Export the chosen model through the established llama.cpp path and check the headset against the floating-point reference. Where candidate scores exist, apply D56: same best candidate, same order among plausible candidates (at least a 1% share), and cached shares within a total variation distance of 0.05. For parsing, also check the constrained outputs and resolved targets. Measure the scores-only direct-selection path, and the parser's decoding plus resolution if it is kept. Time typed submission to validated goal, including relation construction, prompt processing, scoring or decoding, resolution and validation. Report cold and warm latency (median and 90th percentile), token counts, peak memory and MR frame behavior over the intended scene range. Test cache invalidation when the scene or pose changes. These are grounding-only measurements: no detector or speech, and no combined or Gate A pass is inferred.
6. **A2.6 · Report preliminary Gate B on frozen evaluations.** Freeze checkpoints, serializers, relation policies, prompts and clarification thresholds before opening the lab commands. Report public-data tests, viewpoint checks and lab commands separately, never as one headline number. Report correct targets, wrong selections, invalid outputs and clarifications, with errors by relation, frame, composition and candidate count, and uncertainty for the number of independent scenes and commands. Assess the advisor-agreed Gate B criteria, with the current defaults as targets. A clarification on a uniquely answerable command is not a correct selection; ambiguous and unsupported cases are scored separately. A strong rule baseline or an unmet margin is a result. Preliminary Gate B stays distinct from the later A3-based check and from full application feasibility.

**Deliverables** A versioned A2.1 specification; the validator, serializer, tested relation library and resolver; published-data adapters with access and provenance records, filtering counts and scene-disjoint split manifests; rule and model baselines; zero-shot format screening and the matched fine-tuning comparison; fine-tuned checkpoints and export records; the held-out natural-command set; an evaluation harness; and a grounding-only headset report. No custom scene generator.

**Exit** Preliminary Gate B, assessed against thresholds the advisor agreed before the locked evaluation, with public-data, natural-command and headset evidence reported separately. If the thresholds are still unapproved, results are reported against the proposed targets and Gate B is recorded as pending. A failed criterion is recorded with its implication for the model-value claim; the baseline, denominator or test set is never changed to manufacture a pass. A formal Gate A pass is not an A2 exit requirement.

**Note down**

- **Method** Schema and relation-policy versions; frames and grammar; output constraints; dataset releases, access terms, adapters, the region rule and leakage controls; zero-shot screening; matched fine-tuning settings; the candidate-scoring span; the calibration source; exact headset measurement boundaries.
- **Results** Accuracy and failure rates by model, design, format, relation, frame and candidate count; lab and public results separately; export agreement; measured latency, memory and MR behavior; each Gate B criterion's status.
- **Table** Dataset counts before and after filtering, with reasons; supported relations and frames; matched comparisons; accuracy and clarification; prompt and output tokens; measured median and 90th-percentile latency and memory.
- **Figure** Accuracy against measured grounding latency for the shortlisted configurations, cold and warm distinguished, with any estimates labeled as estimates.
- **Discussion** Ten to twenty representative failures, or all if fewer, separating language, frame, geometry, missing-evidence, adapter and export errors; where rules suffice; where model capacity or representation matters.
- **Limitations** Template-heavy public supervision; filtering and small-scene selection bias; unresolved viewpoint coverage; idealized geometry; the small lab set; no speech, perception or flight errors; any unverified combined budget; single-action tests don't show general action understanding.
- **Appendix** The final specification; adapter and split procedures; dataset release identifiers; the fixtures; prompt and serialization examples; filtering counts; any template or generator use; reproduction checks; run IDs.

## Open decisions

Sixteen choices are left to the A2.1 specification, and two to the advisor. Each is recorded with its reason and version before the implementation or locked evaluation that depends on it. None reopens the decisions listed at the top.

### For the A2.1 specification

| Decision | What to record |
| --- | --- |
| Object ID style and tokenization | Class-bearing or opaque IDs; a stable mapping; ordering; the candidate score span; checks for ID or token-length bias |
| Internal schema and model-visible fields | Required geometry; units, axes and handedness; box handling; how unknowns are written; the class map; optional metadata |
| Output grammar and non-selection | Whether ASK is a scored candidate, an explicit output, a resolver status or a combination; separate one, several, none and insufficient-evidence outcomes |
| Supported actions | Which commands are supported first; fixed INSPECT selection against learned action classification |
| Frame types and defaults | Exact frame types; anchored viewer against heading-relative behavior; the default for unqualified directions; whether drone and object frames are supported first |
| Pose and cache policy | Snapshot timing, transforms, invalidation triggers, reuse of pose-dependent relations |
| Vertical relations | Whether above, below and on are supported; the dimension, overlap and contact evidence each needs |
| Other relation definitions | Distance measure, thresholds, overlap tests, ties and uncertainty bands per relation; compatibility with dataset definitions |
| Supported composition | Anchor disambiguation, combined relations, ordinals, negation and earlier turns; unsupported forms rejected without a new dialogue task |
| Region rule | Independence from the target, region extent, keeping distractors and anchors, rejecting oversized cases, reporting retention bias |
| Dataset releases and eligibility | Which subset, its access route, filtering rules, class mapping, viewpoint availability, trustworthy parser labels |
| Split and evaluation manifests | Grouping of scenes, rescans and overlapping sources; development, calibration and test allocation; collecting, labeling and sealing the lab commands |
| Shortlisted serializations and training controls | Exact renderings; matched data and budgets; LoRA settings; when the pilot continues; permitted gap-filling templates |
| Reference model and comparison | Which larger checkpoint, its decoding and prompting; the evidence it gets; no assumption that it is right |
| Confidence and clarification thresholds | Calibration method and partition; scoring denominators; ties, absent targets and plural requests |
| Headset measurement protocol | Cold and warm conditions, workload, memory accounting, frame metrics, parser checks beyond D56 |

### For the advisor

- **Gate B:** the thresholds, denominators and exclusions, how uncertainty is reported, and what separates the preliminary from the final assessment.
- **Gate A while speech waits:** how an interim typed-command result is reported, which later evidence closes the speech-inclusive criterion, and which latency, memory, frame and thermal criteria apply.

## References

*Adds to Revision 3's list, continuing its numbering; \[1\] to \[43\] are kept. Provenance \[A\]: identified by the A2 amendment's source checks on 1 October 2026. Pin exact releases and verify every entry before submission.*

&#91;44\] J. Fang et al., "Transcrib3D: 3D Referring Expression Resolution through Large Language Models," in Proc. IROS, 2024. [Paper](https://arxiv.org/abs/2404.19221) · [Repository](https://github.com/ripl/Transcrib3D). \[A\]

&#91;45\] N. Zantout, H. Zhang, P. Kachana, J. Qiu, J. Zhang and W. Wang, "SORT3D: Spatial Object-centric Reasoning Toolbox for Zero-Shot 3D Grounding Using Large Language Models," in Proc. IROS, 2025. [Paper](https://arxiv.org/abs/2504.18684) · [Repository](https://github.com/nzantout/Sort3D). \[A\]

&#91;46\] C. Huang, O. Mees, A. Zeng and W. Burgard, "Visual Language Maps for Robot Navigation," in Proc. IEEE ICRA, 2023. [Project](https://vlmaps.github.io/). \[A\]

&#91;47\] Y. Qi et al., "REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments," in Proc. CVPR, 2020. [Paper](https://arxiv.org/abs/1904.10151) · [Publisher](https://openaccess.thecvf.com/content_CVPR_2020/html/Qi_REVERIE_Remote_Embodied_Visual_Referring_Expression_in_Real_Indoor_Environments_CVPR_2020_paper.html). \[A\]

&#91;48\] S. M. Lukin et al., "SCOUT: A Situated and Multi-Modal Human-Robot Dialogue Corpus," in Proc. LREC-COLING, 2024, pp. 14445–14458. [Publisher](https://aclanthology.org/2024.lrec-main.1259/). \[A\]

&#91;49\] H. Zhang et al., "IRef-VLA: A Benchmark for Interactive Referential Grounding with Imperfect Language in 3D Scenes," in Proc. IEEE ICRA, 2025. [Paper](https://arxiv.org/abs/2503.17406) · [Data and schema](https://github.com/HaochenZ11/IRef-VLA). \[A\]

&#91;50\] X. Shi, Z. Wu and S. Lee, "Viewpoint-Aware Visual Grounding in 3D Scenes," in Proc. CVPR, 2024. [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_Viewpoint-Aware_Visual_Grounding_in_3D_Scenes_CVPR_2024_paper.pdf) · [Implementation](https://github.com/Sxx1995/Viewpoint-Aware-3D-Grounding-). \[A\]

&#91;51\] Y. Zhang, Z. Gong and A. X. Chang, "Multi3DRefer: Grounding Text Description to Multiple 3D Objects," in Proc. ICCV, 2023. [Paper](https://arxiv.org/abs/2309.05251) · [Project](https://3dlg-hcvc.github.io/multi3drefer/). \[A\]
