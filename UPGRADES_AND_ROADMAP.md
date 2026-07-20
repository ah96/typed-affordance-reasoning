# Upgrades, Results, and Roadmap

What this project adds on top of the original submission — *"Can Frontier Vision-Language Models
Reason About the Interactable World? A Typed Affordance Evaluation"* (ECCV 2026 X-Reason) — and where
it should go next. This is the strategy/status companion to the technical [README](README.md) and the
GPU [LAB_PC_SETUP.md](LAB_PC_SETUP.md).

---

## 1. One-paragraph summary

The original paper introduced a **typed** view of affordances (name *why* an action is blocked, not
just whether) and evaluated four frontier VLMs plus o4-mini on it with two experiments: a
ground-truth study on ADE-Affordance (Experiment A) and a ground-truth-free inter-model agreement
study over SAM-proposed regions (Experiment B, SAM 2 area-ranked vs SAM 3 concept-targeted). Its
headline was that models **agree on *whether* an action is possible but diverge on *why***, and that
chain-of-thought helps but is not necessary. This project turns that single-finding protocol into a
**self-auditing evaluation study plus a model-building program**: it (a) audits what the agreement
signal is actually worth, (b) replaces near-blind n-gram explanation scoring with semantic measures,
(c) diagnoses *where* the typed reasoning breaks, (d) proves the break is largely a fixable
reason→type mapping problem, and (e) packages open-weight-model and spatial-grounding extensions —
all at **zero additional API cost**.

---

## 2. Original vs. upgraded — at a glance

| Axis | Original paper | This project |
|---|---|---|
| Models | 4 frontier VLMs + o4-mini (paid APIs) | + open-weight pool (Qwen3-VL, InternVL) for a **same-weights** CoT ablation and a widened agreement pool |
| Agreement metric | raw 4-way / pairwise / consensus | + **chance correction** (Fleiss κ, Krippendorff α, Cohen κ), tie-free typicality, conditional type agreement |
| Agreement vs. truth | flagged as a caveat | **measured**: ensemble-vs-GT test + model–model vs model–GT κ |
| Error analysis | aggregate mAcc | + **7×7 confusions, bootstrap CIs, constraint-axis decomposition** |
| Explanation scoring | BLEU/METEOR/ROUGE (≈0, "harsh") | + **embedding / NLI / BERTScore**, split by detection and by type-correctness |
| Explanation validity | LLM-judge rejected as biased | + **judge with a validity audit** (consistency / paraphrase / surface perturbations) |
| Spatial grounding | none (labels + text only) | + **OOAL saliency** re-ranking and AGD20K GT-grounded selection (KLD/SIM/NSS, recall@K) |
| Our own model | none | **Direction 5**: reason-first typed model (Stage 0 probe done, Stage 1 designed) |
| Reproducibility | raw predictions committed | + git-packaged repo, `LAB_PC_SETUP.md`, resumable run-ready drivers, zero-budget reanalysis |

---

## 3. Baseline recap (what the original established)

- **Taxonomy.** ADE-Affordance's 7 codes: 0 Positive, 1 Firmly-Negative, 2 Object-Non-functional,
  3 Physical-Obstacle, 4 Socially-Awkward, 5 Socially-Forbidden, 6 Dangerous. Codes 2–6 are rare
  exceptions (2–4% each) needing a one-sentence explanation and consequence.
- **Experiment A (GT).** 200 images, 13,512 (instance, action) pairs, 579 GT exceptions.
  mAcc-7 ranged 0.240–0.289 (Claude best), mAcc-3 0.471–0.531 (Gemini best).
- **Experiment B (GT-free).** SAM 2 area vs SAM 3 concept, K=3. 4-way agreement 7.2% overall,
  34.5% on Positive, 6.3% on exceptions. Concept-targeting raised agreement but lowered the
  exception rate.
- **Reasoning.** o4-mini beat three of four standard models on the 579 exceptions but standard
  Claude matched it → "CoT sufficient, not necessary."

Everything below reuses the **committed raw predictions** from these runs, so the audit cost nothing.

---

## 4. What's new — the upgrades in detail

### 4.1 An audit layer for the agreement signal (Direction 1, done, laptop)

`experiments/analysis/{analysis_agreement, analysis_confusion, analysis_ensemble}.py`

1. **Chance-corrected agreement.** Raw agreement flattered the consensus. Corrected, it is only
   *slight-to-fair*: Fleiss κ = **0.138** (area) / **0.246** (concept). Under area-ranking the 3-way
   κ (0.112) is *lower* than the 7-way (0.138), so the "agree on whether" claim does not survive
   chance correction in its blunt form — it survives conditionally (below).
2. **Label priors, not scenes, drive much of the dissent.** Per-model marginals differ
   systematically: Gemini uses Object-Non-functional for 5.5% of items vs. 41–56% for the others;
   Llama emits Dangerous and Socially-Awkward ≈0.1% each; GPT-5.5 never predicts Socially-Awkward in
   13,512 pairs. Gemini's "outlier" status is largely a prior clash, not per-scene disagreement.
3. **Conditional whether/why gap.** Given two models both flag an exception, they name the *same*
   type only **59.4%** (area) / **43.7%** (concept) of the time — the correct, sharper statement of
   the paper's thesis.
4. **Confusion structure + CIs.** 7×7 matrices expose per-model signatures: Claude refuses the
   residual bucket (Firmly-Negative row acc 4.0% vs GPT's 41.9%), GPT under-types (76.7% of
   exceptions collapse to 0/1), Gemini alone resolves within the social axis, Danger is the only type
   everyone but Llama can name. Bootstrap 95% CIs show the four models' mAcc-7 intervals **overlap**.
5. **Cross-axis errors dominate.** On the 579 exceptions, wrong-axis errors exceed right-axis-wrong-
   type errors for every model (e.g. Claude 0.485 wrong-axis vs 0.092 right-axis). The taxonomy's
   semantic axes are not the models' internal axes.
6. **Consensus ≠ correctness — measured, not caveated.** The majority-vote ensemble beats **no**
   single model and *loses* on exceptions (Claude Detect 0.763 → 0.554 under voting). Models agree
   with each other **2.8×** more than with the labels (mean model–model Cohen κ **0.214** vs
   model–GT κ **0.077**). The most central model in Exp B's exception consensus (GPT-5.5, 83.5%) is
   the *least* accurate exception detector (0.233).

### 4.2 Semantic explanation scoring (Direction 2a, done, laptop)

`experiments/analysis/analysis_explanations.py` — embedding cosine, NLI entailment, BERTScore over
the 579 exceptions, for both the explanation and consequence fields.

- **The "quality gap" was mostly a detection gap.** The n-gram ranking partly re-measured detection
  (missed exceptions score 0). Conditioned on detection, cosine similarity collapses from a
  0.109–0.387 span to **0.413–0.507**, and GPT-5.5 jumps from last to second.
- **Wrong label, right reason.** Explanations attached to the *wrong* type align with human
  references almost as well as those attached to the exact type (Claude 0.497 vs 0.520). Models often
  perceive the correct reason and file it under the wrong code.
- **Consequences converge more than reasons** (detected cosine 0.45–0.56) but entail less
  (0.08–0.22): a scene admits many valid outcomes but fewer valid reasons.

### 4.3 Judge + validity audit (Direction 2b, packaged, awaits one API key)

`experiments/analysis/judge_explanations.py` — a free-tier Gemini judge scoring reason-match, plus a
**validity audit** that re-judges under perturbations that should not change the verdict
(second pass, reference-shuffle, surface rewrite). This directly answers the 2026 critique that
LLM-judges are consistent-but-not-valid. Resumable; run with `GEMINI_API_KEY` set.

### 4.4 Reason-first probe (Direction 5, Stage 0, done, laptop)

`experiments/analysis/stage0_mapper.py` — a text-only classifier trained on 21,119 ADE-Affordance
train-split explanation→code pairs, applied to the models' *own* committed explanations.

- Re-mapping lifts the strongest-explanation model **Claude 0.256 → 0.350 Type (+37% relative)** — a
  new best, from re-reading its own words with no new queries. Gains track explanation quality
  (GPT/Llama flat or worse).
- The text-only route **saturates**: the mapper reaches only **0.374** on held-out *human*
  explanations (zero-shot centroid 0.215, chance 0.20). "It is a table, not a seat" is
  Firmly-Negative or Object-Non-functional depending on the image.
- **Conclusion:** typing needs the reason **and** the pixels → motivates a jointly trained
  reason-first model (Stage 1), not a post-hoc text patch.

### 4.5 Open-weight model extensions (Direction 3, packaged, GPU)

`experiments/local_vlms/` — `serve_vllm.sh`, `replay_regions.py`, and run-ready drivers.

- **Same-weights reasoning ablation.** Qwen3-VL-8B Instruct vs Thinking isolates chain-of-thought
  with weights held fixed — the clean test the o4-mini comparison could not give (it confounded
  reasoning with model identity).
- **Widened agreement pool.** Replays the *exact* committed Exp B regions to new models (SAM never
  re-runs), so open models judge identical inputs → more voters, fewer 2–2 ties, and frozen
  reproducible snapshots that answer the closed-API-drift limitation.

### 4.6 Spatial grounding via OOAL (Direction 4, packaged, GPU)

`experiments/ooal_grounding/` — revives the ablations cut from the submission using the unused
`ooal_models_amar/` checkpoints and the full AGD20K copy.

- **A third selection strategy** (`sam2_ooal`): OOAL affordance-saliency re-ranks SAM masks, compared
  against area-ranked and concept-targeted within one model pool.
- **GT-grounded selection eval**: scores selections against AGD20K's real spatial heatmaps
  (KLD/SIM/NSS, GT-mass recall@K) — converting a GT-free experiment into a partially grounded one and
  addressing the domain-mismatch limitation.

### 4.7 Engineering & reproducibility

Git-packaged repo (was a bare copy), `LAB_PC_SETUP.md`, committed image bundles, resumable per-call
caches, mock/dry-run paths tested on CPU, and self-contained lab-PC drivers with full server
lifecycle management.

---

## 5. Results so far — the story in numbers

| Claim | Evidence |
|---|---|
| Consensus is thinner than it looked | Fleiss κ 0.14 (area) / 0.25 (concept); 3-way κ ≤ 7-way κ under area |
| Divergence is partly systematic bias | per-model label priors differ 10–50× on some codes |
| Whether/why gap, stated correctly | both-exception same-type only 0.59 (area) / 0.44 (concept) |
| Consensus ≠ correctness | ensemble beats no single model; model–model κ 0.214 vs model–GT κ 0.077 (2.8×) |
| Errors cross constraint axes | wrong-axis > right-axis-wrong-type for all five models |
| Explanation gap ≈ detection gap | detected-only cosine span 0.41–0.51 (was 0.11–0.39) |
| Wrong code, right reason | wrong-axis explanations ≈ as aligned as exact-type ones |
| Mapping is the fixable bottleneck | text mapper lifts Claude Type +37%; ceiling 0.374 needs pixels |

**Net narrative.** The typed taxonomy exposes real divergence, but that divergence is (1) overstated
by uncorrected agreement, (2) partly a clash of priors, (3) partly a reason→type *mapping* failure
rather than a perception failure, and (4) **not** a reliable proxy for correctness. This is a
stronger, more defensible thesis than the original "they diverge on why," and it comes with a
constructive next step (reason-first typing).

---

## 6. Novelty and positioning

- **Exception-aware / "when-NOT-to-act" typed evaluation is an essentially unoccupied niche.** The
  2025–26 literature is dominated by *positive* affordance grounding — RL reasoners (Affordance-R1),
  zero-shot agentic grounding (A4-Agent, TokAG), and affordance-scaffolded VLAs (CoA-VLA, AffordVLA).
  None target broken/social/dangerous exceptions or scene-level "why not."
- **Judge-validity and calibrated, typed reporting are now expected hygiene** ("Reliability without
  Validity"; VLM-judges rank-but-can't-score). Our audit layer and semantic scoring meet that bar
  head-on, inside an annotation-free pipeline — which no one has audited before.
- **The reason-first diagnosis is new and actionable.** Showing that wrong types often carry right
  reasons, and that a text mapper recovers accuracy up to a pixel-bounded ceiling, reframes the
  problem from "models can't reason" to "models mis-map reasoning to a taxonomy" — and predicts what a
  purpose-built model should fix.
- **What we deliberately avoid** (per the field scan): another AGD20K heatmap benchmark, training a
  large VLA/RL model, or leaderboard accuracy with an unaudited judge — all treated as stale in 2026.

---

## 7. Experiments still to run — and what we expect

| # | Experiment | Where | Hypothesis / expected outcome | Risk |
|---|---|---|---|---|
| D2b | Judge + validity audit | laptop (free Gemini) | Judge agrees with semantic scores on reason-match; audit shows moderate consistency that **drops under paraphrase/surface** perturbation, quantifying judge (un)reliability per exception type. | Free-tier rate limits (handled by resumable cache). |
| D3a | Same-weights CoT ablation | lab PC | Thinking > Instruct on Detect/Type by a **modest** margin; the gap is smaller than the Claude–GPT gap → reinforces "CoT sufficient, not necessary" with identity controlled. Thinking should also shrink the **wrong-axis** error share. | vLLM/Qwen version support. |
| D3b | Widened agreement pool | lab PC | More voters cut the 21.2% tie rate and **lower** chance-corrected κ further (open models add prior diversity) → strengthens "consensus is thin." Frontier-vs-open axis emerges. | Free/open model quality. |
| D4-1 | OOAL checkpoint sanity | lab PC | KLD/SIM/NSS land near the OOAL paper (~1.07/0.46/1.14 Seen) → validates the adapter. **Gates** D4-2/3. | Upstream API drift (adapter fix isolated). |
| D4-2 | GT-grounded selection | lab PC | OOAL-ranked selection achieves **higher GT-mass recall@K** than area-ranked → affordance-aware selection is measurably better, not just higher-agreement. | — |
| D4-3 | `sam2_ooal` as 3rd strategy | lab PC | Recovers exception coverage lost by concept-targeting while keeping agreement up → a better reliability/coverage trade-off than either existing strategy. | — |
| D5-1 | Reason-first typed model (QLoRA Qwen3-VL) | lab PC | Jointly typing reason+vision beats the text-only ceiling (0.374) and the Claude+mapper baseline (0.350); reason-first ordering beats direct-label training on Type. | ADE label noise; domain shift. |

**Baselines the model must beat (already on disk, free):** the four frontier models' Type scores, the
majority-vote ensemble, and the Stage-0 text mapper (0.350). Ceiling if detection were perfectly
typed: 0.786 (Claude).

---

## 8. Limitations

**Inherited from the original (still true).**
- ADE ground truth covers only 3 of 6 actions; `cut`/`throw`/`ride` remain GT-free.
- The social/safety codes encode one normative frame; the Awkward-vs-Forbidden boundary is
  culturally contestable.
- AGD20K-vocabulary-on-ADE20K domain mismatch leaves some actions sparse (`ride` in 5/200 scenes).
- Rare exception classes (2–4%) mean small per-category samples.

**New / sharpened here.**
- The frontier predictions come from **closed, drifting APIs** (a reproducibility risk the open-model
  pool only partially offsets). We cannot re-query them under budget.
- Semantic scorers (embeddings/NLI/BERTScore) and the mapper are trained on **crowd-written** ADE
  explanations, so model-prose→annotator-prose domain shift bounds them — precisely what the 0.374
  ceiling measures.
- The reason-first mapper inherits ADE's normative frame and label noise; Stage 1 will bake this in.
- D4 depends on an **external OOAL codebase**; the adapter is untested on GPU by design (Step 1 gates
  it).
- Everything is **static/classification** evaluation; we do not close the loop to real action
  consequences (out of scope, and compute-prohibitive).

---

## 9. How to frame the paper (three viable scopes)

1. **Audit paper (ready now).** Directions 1+2 (+2b): "What is agreement-based, annotation-free typed
   affordance evaluation actually worth?" Self-contained, zero-budget, methodologically current.
2. **Audit + extensions (adds a few GPU days).** + D3 (open-model reasoning ablation, frontier-vs-open)
   and D4 (spatial grounding). Broader, still no new paid tokens.
3. **Diagnosis → model paper (the arc).** + D5 Stage 1: the reason-first model that the audit predicts.
   The strongest story, but needs training and careful baselining.

Recommended: aim the writing at **scope 2**, structured so that D5 can be appended as scope 3 if the
model works — the audit sections stand on their own if it does not.

---

## 10. Where to submit

Deadlines below are approximate cycle positions — **verify current CFPs**, as dates shift year to
year. Ordered by fit.

### Tier A — main venues (scope 2–3, highest bar/reward)

- **NeurIPS Datasets & Benchmarks track.** Best structural fit for an evaluation-methodology paper:
  it rewards rigorous benchmark critique, chance-corrected metrics, judge-validity audits, and
  reproducibility — exactly our strengths. Scope 2 is competitive here; scope 3 strengthens it.
- **TMLR (Transactions on Machine Learning Research).** No deadline, rolling review, values
  correctness and thoroughness over novelty-hype. Ideal if we want the audit out without racing a
  conference clock, and it accepts "solid, well-scoped" evaluation contributions.
- **CVPR / ICCV / ECCV main track.** Reachable only at scope 3 (a model + benchmark). As a pure audit
  it would likely be viewed as incremental for the main track; with the reason-first model and the
  grounding extension it becomes a full contribution.

### Tier B — evaluation / benchmark-leaning venues

- **COLM (Conference on Language Modeling).** Friendly to evaluation and analysis of (V)LM reasoning,
  judge reliability, and calibration. Good scope-2 home if framed around VLM evaluation methodology.
- **AAAI / an *AAAI Evaluation* or *Trustworthy AI* track.** Receptive to typed error taxonomies and
  reliability audits.

### Tier C — workshops (fast, lower bar, momentum-preserving)

- **BEAM 2 @ ECCV 2026 — Benchmarking Evidence-Aligned Multimodal Reasoning.** The single closest
  thematic match: evidence-aligned, verifiable multimodal reasoning is our audit's whole premise.
  Strong scope-1/2 target.
- **X-Reason @ ECCV 2026 (the original venue).** A natural home for the extended/camera-ready-plus
  version, especially the grounding (D4) and interactable-world framing.
- **eXCV @ ECCV 2026 / XAI4CV @ CVPR 2026.** Explainability-for-vision: the semantic explanation
  scoring, faithfulness-vs-plausibility angle, and reason-first probe fit well.
- **ERA @ CVPR 2026 / EMR @ ECCV 2026 (embodied reasoning).** Fit if we lean the "when-not-to-act"
  safety framing toward downstream action; weaker if we stay purely static.

### Recommendation

Two-track it. **Submit the audit (scope 2) to NeurIPS D&B or TMLR** for the archival, citable
version, and in parallel **put the extended work through BEAM 2 / X-Reason** for fast community
feedback and visibility. If D5 Stage 1 lands a model that beats the Stage-0 ceiling, **upgrade the
target to a CVPR/ICCV main-track submission** (scope 3). Keep everything reproducible-from-repo so the
same artifacts serve all three.

---

*Status snapshot: Directions 1, 2a, and 5-Stage-0 complete (laptop, zero cost); 2b awaits one API
key; 3 and 4 packaged and mock-tested for the lab PC; the paper source already carries the new
sections. See [project memory / follow-up state] for the live checklist.*
