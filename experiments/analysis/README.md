# Reanalysis of the committed raw predictions

Everything here recomputes from the committed `raw_*.jsonl` (Experiment A) and
`<model>_<mode>_K3.jsonl` (Experiment B) files — no cache, no API keys, no GPU, stdlib only.
Outputs land in `out/*.json`.

```bash
python3 analysis_agreement.py   # Exp B: chance-corrected agreement, typicality, label priors
python3 analysis_confusion.py   # Exp A: 7x7 confusions, bootstrap CIs, exception-axis errors
python3 analysis_ensemble.py    # Exp A: majority-vote ensemble vs GT, model-model vs model-GT
```

## 1. Chance-corrected agreement (Experiment B)

Raw pairwise agreement overstates consensus because the label space is skewed. Chance-corrected:

| mode | raw pairwise 7-way | Fleiss κ-7 | Fleiss κ-3 | Kripp. α-7 | mean pair Cohen κ-7 |
|---|---|---|---|---|---|
| sam2_area | 0.360 | **0.138** | 0.112 | 0.138 | 0.167 |
| sam3_concept | 0.462 | **0.246** | 0.340 | 0.247 | 0.263 |

- Corrected agreement is *slight-to-fair* (Landis-Koch), far below what the raw rates suggest.
- In area mode the 3-way κ (0.112) is **lower** than the 7-way (0.138): once chance is removed,
  models do not even agree on the coarse verdict more than on the fine one there.
- **Label priors, not scene reading, drive much of the disagreement.** Area mode:
  Gemini uses ObjectNonFunctional for 5.5% of items vs 41–56% for the other three, and prefers
  FirmlyNegative (47.1%) + PhysicalObstacle (18.4%); Llama emits Dangerous and SociallyAwkward
  almost never (0.1% each); GPT-5.5 emits SociallyAwkward for 0.4%. Gemini's "outlier" status in
  the paper is largely a marginal-prior clash, not per-scene dissent.
- Tie-free typicality (mean agreement with the other three, no 2-2 exclusions) reproduces the
  paper's ordering — GPT 0.395, Claude 0.393, Llama 0.367, Gemini 0.285 (area) — so the
  27.6% tie exclusion did not distort the ranking, it only hid how weak the majority is.
- Given two models both flag an exception, they name the same type 59.4% (area) / 43.7%
  (concept) of the time.

## 2. Confusion structure (Experiment A, bootstrap 95% CIs by image cluster)

| model | mAcc-7 | mAcc-3 | Detect | Type |
|---|---|---|---|---|
| GPT-5.5 | 0.251 [0.215, 0.281] | 0.480 [0.456, 0.504] | 0.233 [0.173, 0.296] | 0.110 [0.059, 0.153] |
| Claude Sonnet 5 | 0.289 [0.256, 0.322] | 0.504 [0.469, 0.532] | 0.763 [0.710, 0.814] | 0.256 [0.213, 0.303] |
| Gemini 3.5 Flash | 0.277 [0.250, 0.300] | 0.531 [0.495, 0.561] | 0.551 [0.495, 0.604] | 0.180 [0.140, 0.213] |
| Llama 4 Maverick | 0.240 [0.211, 0.267] | 0.471 [0.448, 0.495] | 0.371 [0.313, 0.428] | 0.128 [0.086, 0.168] |
| o4-mini (579 exc. only) | — | — | 0.701 [0.654, 0.748] | 0.229 [0.189, 0.269] |

The mAcc-7 CIs of all four standard models overlap; Claude vs Llama is the only near-separated
pair. Detect separates cleanly (Claude ≫ Gemini ≫ Llama ≫ GPT).

Model-specific signatures from the 7×7 confusions:

- **Claude refuses the residual bucket**: FirmlyNegative row accuracy 4.0% (vs GPT 41.9%) —
  it converts flat negatives into typed exceptions (37.7% ObjectNonFunctional, 28.3% Dangerous).
  Over-typing, not under-detection, is its failure mode; it is also why its raw agreement with
  the 88%-FirmlyNegative GT is only 0.097.
- **GPT-5.5 under-types**: 76.7% of GT exceptions collapse to Positive/FirmlyNegative, and it
  *never* predicts SociallyAwkward (0 of 13,512). Social exceptions: 0.6% exact.
- **Gemini is the only model with real within-social resolution** (right-axis 0.151 on social
  GT), but it misses half of all exceptions (0.494).
- **Danger is the only type every model can name** (exact: Claude 0.724, o4-mini 0.592,
  Gemini 0.447, GPT 0.395; except Llama 0.092, which barely emits Dangerous at all).
- **Errors cross constraint axes far more than they stay within them.** For every model,
  wrong-axis errors exceed right-axis-wrong-type errors (e.g. Claude functional GT: 0.596
  wrong-axis vs 0.117 right-axis). The taxonomy's axes are not the models' internal axes.
- o4-mini never uses FirmlyNegative on the exception subset and mistypes toward Dangerous
  (54.2% of PhysicalObstacle GT) — high Detect (0.701), but half its detections land on the
  wrong axis (0.503).

## 3. Does consensus track correctness? (Experiment A)

Majority vote of the four standard models on all 13,512 GT-labelled pairs:

| | mAcc-7 | mAcc-3 | Detect | Type | n |
|---|---|---|---|---|---|
| best single (Claude) | 0.289 | 0.504 | 0.763 | 0.256 | 13,512 |
| ensemble, ties dropped | 0.279 | 0.525 | 0.431 | 0.163 | 10,650 |
| ensemble, ties→most severe | 0.279 | 0.516 | 0.554 | 0.192 | 13,512 |

- **The ensemble beats no single-model column.** On exceptions it erases the best model's
  edge (Detect 0.554 vs Claude's 0.763): the majority drags typed judgments toward the
  shared prior, not toward the ground truth.
- 21.2% of items are tied under 4 voters.
- **Models agree with each other far more than with the labels**: mean inter-model Cohen κ
  0.214 vs mean model-GT κ 0.077 (2.8×; raw agreement 0.359 vs 0.307). This is the number
  behind the paper's "agreement is a reliability signal, not correctness" caveat — and the
  ensemble rows show the gap is not benign.

## 4. LLM judge + validity audit (`judge_explanations.py`)

An independent judge (Gemini 3.1 Flash-Lite, **not** one of the evaluated models) rules whether each
detected exception's explanation gives the same reason as the human references, over 1,517 rows.

- **Corroborates the semantic ranking**: same-reason rate Claude 0.507 > GPT-5.5 0.385 >
  Gemini 0.329 > o4-mini 0.288 > Llama 0.200 — the same order as the embedding/NLI metrics.
- **Sharpens reason-first**: same-rate is higher on exact-type than wrong-axis rows for most models
  (Claude 0.685 vs 0.449), a wider gap than the embedding cosine — so wrong-type explanations carry
  the right reason less often than surface similarity implied, but still substantially (0.19–0.45).
- **Validity audit (the headline)**: re-judging identical input is perfectly stable (1.000), but
  reversing reference order or a meaning-preserving surface rewrite each flip **11.3%** of verdicts
  (stability 0.887). The judge is *reliable but not valid* — hence reported only alongside the
  reference-based metrics, never as ground truth.

## Takeaway

Experiment B's consensus measures reward typicality of priors. The two most "central" models
there (GPT-5.5, and GPT/Claude under concept mode) sit at opposite extremes of GT accuracy in
Experiment A, and voting hurts exactly where the paper's central finding lives (typed
exceptions). Agreement-based evaluation of typed affordances needs chance correction and a
correctness anchor.
