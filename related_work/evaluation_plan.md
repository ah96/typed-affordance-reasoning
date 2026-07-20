# AffBench Evaluation Plan

**Paper:** AffBench: Benchmarking Frontier VLMs on Structured Affordance Reasoning  
**Target venue:** ECCV 2026 Workshop X-Reason  
**Models evaluated:** GPT-4.1-mini, Claude Sonnet 4.6, Gemini 2.5 Pro  
**Taxonomy:** 7-way label code (0=Positive, 1=Firmly Negative, 2=Object Non-functional, 3=Physical Obstacle, 4=Socially Awkward, 5=Socially Forbidden, 6=Dangerous)

---

## 1. What Is Being Evaluated

### Experiment A — Text-Only on ADE-Affordance Ground Truth

**Input:** Object + scene context (textual description OR image patch) + action query  
**Output per instance:** (a) label code 0–6, (b) exception description (free text), (c) consequence prediction (free text)  
**Ground truth:** ADE-Affordance dataset annotations:
- Binary affordance label (yes/no) for sit, run, grasp
- Exception type (physical constraint, social norm, safety hazard)
- Free-text exception description and predicted consequence

**Goal:** Measure whether VLMs correctly classify affordance type and whether their textual explanations align with human annotations.

---

### Experiment B — Vision Pipeline on Custom Images

**Input:** Full scene image processed by SAM (and optionally OOAL)  
**Pipeline:**
1. SAM automatic mask generator produces K instance proposals per image
2. (Optional) OOAL saliency filter re-ranks and selects top-K instances by action relevance
3. Each selected instance is cropped/masked and passed to VLM with action query
4. VLM outputs label code + explanation + consequence

**Ground truth (partial):** AGD20K spatial grounding maps for actions: sit_on, hold, carry, cut, throw, ride  
**Goal:** Measure how well the full pipeline (SAM → VLM reasoning) performs relative to spatial GT, and how much OOAL saliency guidance improves over SAM-only.

---

## 2. Metrics

### 2.1 Classification Metrics (both experiments)

| Metric | Definition |
|---|---|
| Per-class accuracy | Fraction of instances where predicted label code = GT code, per code 0–6 |
| Mean Accuracy (mAcc) | Macro-average of per-class accuracies across all 7 codes |
| Weighted Accuracy | Weighted average by class frequency (handles class imbalance) |
| Confusion matrix | 7×7 matrix of predicted vs GT label codes |
| Binary Accuracy | Collapse to positive (0) vs negative (1–6) |

### 2.2 Text Quality Metrics (Experiment A only — exception descriptions and consequences)

Applied to free-text fields against ADE-Affordance reference texts:

| Metric | Implementation | Notes |
|---|---|---|
| BLEU-4 | `sacrebleu` or `nltk.translate.bleu_score` | Standard n-gram precision |
| METEOR | `nltk.translate.meteor_score` | Handles synonymy via WordNet |
| ROUGE-L | `rouge_score` (longest common subsequence) | Recall-oriented |
| CIDEr | `pycocoevalcap` | Consensus-based; rewards specificity |

Compute per-model and report mean ± std across test instances.

### 2.3 Inter-Model Agreement (both experiments)

**Definition:** Rate at which independently queried models output the same label code for the same instance.

| Metric | Definition |
|---|---|
| Pairwise agreement (GPT vs Claude) | Cohen's κ and raw % agreement |
| Pairwise agreement (GPT vs Gemini) | Cohen's κ and raw % agreement |
| Pairwise agreement (Claude vs Gemini) | Cohen's κ and raw % agreement |
| 3-way agreement | % of instances where all three models agree |
| Consensus accuracy | Accuracy of majority-vote ensemble vs GT |

**Interpretation:** High inter-model agreement + high accuracy = reliable, grounded judgment. High agreement + low accuracy = systematic shared bias. Low agreement = uncertain / ambiguous cases; log these for qualitative analysis.

### 2.4 SAM vs SAM+OOAL Comparison (Experiment B)

For K ∈ {5, 10, 20} candidate masks per image:

| Configuration | Description |
|---|---|
| SAM-only (K=5) | Top-5 masks by SAM confidence score |
| SAM-only (K=10) | Top-10 masks by SAM confidence score |
| SAM-only (K=20) | Top-20 masks by SAM confidence score |
| SAM+OOAL (K=5) | Top-5 masks after OOAL saliency re-ranking |
| SAM+OOAL (K=10) | Top-10 masks after OOAL saliency re-ranking |
| SAM+OOAL (K=20) | Top-20 masks after OOAL saliency re-ranking |

Report per-configuration: mAcc, binary accuracy, inter-model agreement, and inference time.

---

## 3. Ground Truth Available

### ADE-Affordance (Experiment A GT)
- **Actions covered:** sit, run, grasp
- **Label types:** binary affordance (yes/no) + exception category + free-text description + consequence text
- **Scale:** ~37k object instances across ADE20K scenes
- **Availability:** Authors' website (Chuang et al. CVPR 2018); check for updated release
- **Mapping to 7-way taxonomy:**
  - Affordance = yes → code 0 (Positive)
  - Physical constraint → code 3 (Physical Obstacle)
  - Social norm (mild) → code 4 (Socially Awkward)
  - Social norm (forbidden) → code 5 (Socially Forbidden)
  - Safety hazard → code 6 (Dangerous)
  - Object non-functional → code 2 (Object Non-functional)
  - No affordance, no exception → code 1 (Firmly Negative)
  - **Note:** This mapping must be validated manually on a sample before running evaluation.

### AGD20K (Experiment B spatial GT)
- **Actions covered:** sit_on, hold, carry, cut, throw, ride (seen); + 25 unseen
- **Label types:** spatial heatmap (pixel-level action region)
- **Usage in AffBench:** Soft spatial GT for checking whether VLM-selected instances overlap with GT affordance regions
- **Metric:** KL-divergence between OOAL saliency map and AGD20K spatial GT (as used in the OOAL paper)
- **Availability:** Li et al. CVPR 2024 (OOAL project page)

---

## 4. What Is Missing / Needs to Be Prepared

### 4.1 Data Preparation

| Task | Status | Priority |
|---|---|---|
| Download and verify ADE-Affordance dataset | Not confirmed | HIGH |
| Map ADE-Affordance exception categories to 7-way taxonomy codes | Not done | HIGH |
| Prepare ADE-Affordance test split (ensure no train/test overlap with any VLM training data) | Not done | HIGH |
| Download AGD20K dataset | Not confirmed | MEDIUM |
| Collect/select custom images for Experiment B (scenes with diverse affordance types) | Not done | MEDIUM |
| Prepare image annotation template for any new ground-truth labeling needed | Not done | LOW |

### 4.2 Pipeline Implementation

| Task | Status | Priority |
|---|---|---|
| SAM automatic mask generator: confirm batch inference works on target hardware | Partial (from existing code) | HIGH |
| OOAL integration: load DINOv2+CLIP, run saliency re-ranking on SAM masks | Partial | HIGH |
| VLM API calls: GPT-4.1-mini, Claude Sonnet 4.6, Gemini 2.5 Pro endpoints + rate limiting | Partial | HIGH |
| Structured output parsing: extract label code (0–6) + exception text + consequence text from VLM response | Not done | HIGH |
| Retry logic and cost tracking for API calls | Not done | MEDIUM |
| Caching layer for VLM responses (to avoid re-running identical queries) | See `outputs/cache_llm.json` (deleted in git) — needs recreation | MEDIUM |

### 4.3 Evaluation Code

| Task | Status | Priority |
|---|---|---|
| Per-class accuracy + mAcc computation | Not done | HIGH |
| Confusion matrix visualization | Not done | HIGH |
| BLEU-4, METEOR, ROUGE-L, CIDEr computation for text fields | Not done | HIGH |
| Pairwise + 3-way Cohen's κ and agreement % | Not done | HIGH |
| KL-divergence between OOAL saliency and AGD20K GT | Not done | MEDIUM |
| Ablation over K ∈ {5,10,20} for SAM/OOAL | Not done | MEDIUM |

### 4.4 Baselines

| Baseline | Description | Priority |
|---|---|---|
| Random baseline | Uniform random over 7 codes | LOW |
| Majority class | Predict most frequent code | LOW |
| GNN (ADE-Affordance paper) | Chuang et al. 2018 model predictions on their test set (if available) | MEDIUM |
| OOAL (spatial only, no VLM) | Compare spatial accuracy vs VLM relational accuracy | MEDIUM |

---

## 5. Step-by-Step Execution Checklist

### Phase 1: Data Setup

- [ ] 1.1 Download ADE-Affordance dataset from Chuang et al. CVPR 2018 project page
- [ ] 1.2 Inspect annotation format; document field names and value ranges
- [ ] 1.3 Create mapping table: ADE-Affordance exception category → 7-way AffBench code
- [ ] 1.4 Validate mapping on 50 random samples with manual review
- [ ] 1.5 Create `data/ade_affordance/test.json` with fields: `image_id`, `object_id`, `action`, `gt_code`, `gt_exception_text`, `gt_consequence_text`
- [ ] 1.6 Download AGD20K dataset from OOAL project page
- [ ] 1.7 Select N=200 images for Experiment B from custom image set (ensure diversity across all 7 label codes)

### Phase 2: Pipeline Setup

- [ ] 2.1 Install/verify SAM: `pip install segment-anything`; test on 5 images
- [ ] 2.2 Set up OOAL: clone repo, load DINOv2 + CLIP weights, test saliency on 5 images
- [ ] 2.3 Configure API keys for all three VLMs; verify rate limits and token budgets
- [ ] 2.4 Write `src/vlm_query.py`: takes (image_crop, action_label) → calls VLM → parses structured response
- [ ] 2.5 Write `src/parse_response.py`: extracts label code (integer 0–6) + texts from VLM output
- [ ] 2.6 Write `src/instance_selector.py`: SAM masks → (optional OOAL re-rank) → top-K instances
- [ ] 2.7 Test full pipeline end-to-end on 10 images; manually verify outputs
- [ ] 2.8 Implement LLM response cache (JSON keyed by `hash(image_id + action + model)`)

### Phase 3: Experiment A — ADE-Affordance GT Evaluation

- [ ] 3.1 Run GPT-4.1-mini on ADE-Affordance test split; save to `outputs/expA_gpt.json`
- [ ] 3.2 Run Claude Sonnet 4.6 on same split; save to `outputs/expA_claude.json`
- [ ] 3.3 Run Gemini 2.5 Pro on same split; save to `outputs/expA_gemini.json`
- [ ] 3.4 Compute per-model: per-class accuracy, mAcc, confusion matrix
- [ ] 3.5 Compute BLEU-4, METEOR, ROUGE-L, CIDEr for exception texts vs GT
- [ ] 3.6 Compute pairwise + 3-way agreement (Cohen's κ and raw %)
- [ ] 3.7 Generate confusion matrix plots for each model + ensemble
- [ ] 3.8 Identify top-10 disagreement instances; annotate qualitatively

### Phase 4: Experiment B — Vision Pipeline Evaluation

- [ ] 4.1 Run SAM-only pipeline with K=5,10,20 on custom image set; save outputs
- [ ] 4.2 Run SAM+OOAL pipeline with K=5,10,20 on same images; save outputs
- [ ] 4.3 For each configuration × model, compute: mAcc, binary accuracy, inter-model agreement
- [ ] 4.4 Compute KL-divergence of OOAL saliency vs AGD20K GT for overlapping actions
- [ ] 4.5 Generate table: rows = configurations (6 rows: SAM/OOAL × K), columns = metrics
- [ ] 4.6 Profile inference time per configuration

### Phase 5: Analysis and Write-Up

- [ ] 5.1 Identify code categories where inter-model agreement is highest/lowest
- [ ] 5.2 Identify code categories where VLMs are systematically wrong (high agreement, low accuracy)
- [ ] 5.3 Analyse failure modes by code: are social codes harder than physical codes?
- [ ] 5.4 Qualitative examples: select 3–5 representative success and failure cases with images
- [ ] 5.5 Compute error bars: report 95% confidence intervals for all metrics
- [ ] 5.6 Draft Results section tables (one per experiment)
- [ ] 5.7 Draft Discussion section covering model-specific patterns and taxonomy difficulty
- [ ] 5.8 Update `paper/refs.bib` with new cite keys from `related_work/related_work_draft.md`
- [ ] 5.9 Submit draft to co-authors for review

---

## 6. Budget Estimate

| Item | Estimate |
|---|---|
| GPT-4.1-mini API (Exp A, 37k queries) | ~$15–25 at ~$0.40–0.60/1k tokens |
| Claude Sonnet 4.6 API (Exp A, 37k queries) | ~$30–50 |
| Gemini 2.5 Pro API (Exp A, 37k queries) | ~$20–40 |
| Exp B (N=200 images × 3 models × 6 configs) | ~$15–25 total |
| **Total estimated API cost** | **$80–140** |

> Note: Use caching aggressively. If Experiment A runs successfully first, results can be re-used for ablation table construction in Experiment B as an internal consistency check.

---

## 7. Prompt Template (Reference)

```
You are an affordance reasoning assistant. You will be shown an image of a scene
(or a description of a scene and object) and asked whether a specified action is
appropriate.

Respond with a JSON object containing exactly three fields:
{
  "code": <integer 0–6>,
  "exception": "<free text: why the action is constrained, or 'N/A' if code 0>",
  "consequence": "<free text: what would happen if the action were taken anyway, or 'N/A' if code 0>"
}

Label codes:
0 = Positive (action is appropriate and safe)
1 = Firmly Negative (action is impossible or clearly inappropriate; no specific exception type)
2 = Object Non-functional (the object cannot support the action due to its state/type)
3 = Physical Obstacle (a physical constraint prevents or endangers the action)
4 = Socially Awkward (the action is possible but socially inappropriate in this context)
5 = Socially Forbidden (the action is prohibited by explicit social or legal rule)
6 = Dangerous (the action poses a safety risk to the agent or others)

Scene: {scene_description}
Object: {object_description}
Action: {action}
```

Adjust the template for vision-only input by replacing the text description with the image crop passed to the VLM's vision API.
