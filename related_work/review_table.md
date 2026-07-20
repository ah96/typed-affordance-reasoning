# Comparative Review Table — AffBench Related Work

This table reviews all 26 unique papers across the dimensions most relevant to AffBench.

**Column definitions:**
- **Aff. Type**: spatial grounding (S) = where on object/image; relational reasoning (R) = whether/why appropriate in context
- **Dataset**: dataset(s) used for training or evaluation
- **Taxonomy / Labels**: affordance label scheme
- **VLM/LLM**: foundation model used, if any
- **XAI Component**: explainability / interpretability component
- **Relation to AffBench**: how closely the paper connects to AffBench goals (High / Medium / Low)

---

| # | Paper | Aff. Type | Dataset | Taxonomy / Labels | VLM/LLM Used | XAI Component | Relation to AffBench |
|---|---|---|---|---|---|---|---|
| 1 | Jamone et al. 2018 — *Affordances in Psychology, Neuroscience, and Robotics: A Survey* | Both (theoretical) | N/A — survey | Gibson's ecological theory; multidisciplinary taxonomy | None | None | Low — foundational theory reference |
| 2 | Zech et al. 2017 — *Computational Models of Affordance: Taxonomy and Classification* | Both (theoretical) | N/A — survey | Proposed robotics taxonomy (perception, learning, action) | None | None | Low — taxonomy inspiration |
| 3 | Myers et al. 2015 — *Affordance Detection of Tool Parts from Geometric Features* | S (part-level) | UMD Part-Affordance Dataset | Multi-label ranked affordances (cut, grasp, contain, …) | None | None | Low — early spatial baseline |
| 4 | Nguyen et al. 2016 — *Detecting Object Affordances with CNNs* | S (pixel-level) | RGB-D objects dataset | Affordance region labels (grasp, contain, support, …) | None | None | Low — CNN-era baseline |
| 5 | Nguyen et al. 2017 — *Object-Based Affordances Detection with CNN+CRF* | S (pixel-level) | Custom RGB-D dataset | Affordance region labels + object class | None | None | Low — improved CNN baseline |
| 6 | Do et al. 2018 — *AffordanceNet: End-to-End Deep Learning* | S (pixel-level) | IIT-AFF, ADE20K subset | 9 affordance classes (grasp, contain, support, …) | None | None | Low — state-of-the-art baseline |
| 7 | Duc et al. 2020 — *Learning Affordance Segmentation: An Investigative Study* | S (pixel-level) | IIT-AFF | Affordance region labels | None | None | Low — segmentation focus |
| 8 | Lakani 2018 — *Affordance-Driven Visual Object Representation* (PhD) | S (part-level) | Custom grasping datasets | Functional affordance labels (per part) | None | None | Low — robotic manipulation focus |
| 9 | Lakani et al. 2019 — *Parts and Affordance for Robot Manipulation* | S (part-level) | UMD Part-Affordance Dataset | Multi-label affordance per part | None | None | Low — robotic grasping focus |
| 10 | Chu et al. 2020 — *Recognizing Affordances for Scene Reasoning* | S (pixel + symbolic) | IIT-AFF + UMD | Category-agnostic affordance labels | None | None | Low-Medium — scene reasoning angle |
| 11 | Chu et al. 2019 — *Affordance Detection and Ranking on Novel Objects* | S (pixel + ranked) | IIT-AFF, UMD | Ranked affordance labels | None | None | Low — ranking extension |
| 12 | Jiang et al. 2022 — *A4T: Hierarchical Affordance for Transparent Objects* | S (hierarchical pixel) | TRANS-AFF (custom) | Positional affordance labels for transparent parts | None | None | Low — niche domain (transparent objects) |
| 13 | Engelbracht et al. 2024 — *SpotLight: Robotic Scene Understanding* | R (action-level) + S | Custom light-switch dataset (715 images) | Binary affordance + scene graph relations | VLM (unspecified, GPT-4 class) | None | Medium-High — VLM for affordance prediction; scene-level reasoning |
| 14 | Zeng et al. 2022 — *Robotic Pick-and-Place with Multi-Affordance Grasping* | S (pixel-level) | Amazon Robotics Challenge | 4 grasp primitive affordances | None | None | Low — physical grasping only |
| 15 | Chu et al. 2019 — *Affordance Segmentation via Synthetic Images* | S (pixel-level) | Synthetic UMD | Affordance region labels | None | None | Low — domain adaptation focus |
| 16 | Chuang et al. 2018 — *Learning to Act Properly* (ADE-Affordance) | R (relational + textual) | **ADE-Affordance** (ADE20K-based) | sit / run / grasp + exception codes (physical, social, safety) + free-text explanations | GNN (no VLM) | Explanation generation | **High — primary GT dataset for AffBench Experiment A** |
| 17 | Zhang et al. 2024 — *Self-Explainable Affordance Learning with Embodied Caption* (SEA) | S + textual explanation | Custom SEA dataset (images + heatmaps + captions) | Action region labels + embodied text captions | Vision-language model (CLIP-based) | Embodied caption generation; BLEU/METEOR evaluation | **High — dual affordance+caption output mirrors AffBench VLM output format** |
| 18 | Mirnateghi et al. 2024 — *Towards Explainability of Affordance Learning in Robot Vision* | S (pixel-level) + textual | Multi-view RGB affordance dataset | Affordance class labels | GPT-4 (for textual explanation) | CAM heatmap + GPT-4 textual explanation | **High — GPT-4 used to explain affordance predictions; directly relevant to AffBench** |
| 19 | Bhattacharyya et al. 2023 — *Visual Affordance Recognition: XAI Study* | S (object-level classification) | CAD-120 (modified) | Object affordance classes | None (pretrained CNN) | Smooth Grad-CAM++ post-hoc | **Medium-High — XAI motivation aligns with AffBench's interpretability goals** |
| 20 | Luo et al. 2021 — *One-Shot Affordance Detection* | S (pixel-level) | PAD (Purpose-driven Affordance Dataset, 4k images) | 31 affordance categories, 72 objects | None | None | Medium — one-shot predecessor to OOAL used in AffBench |
| 21 | Li et al. 2024 — *One-Shot Open Affordance Learning with Foundation Models* (OOAL) | S (spatial saliency maps) | **AGD20K** + UMD | Open-vocabulary affordance (36+25 categories) | DINOv2 + CLIP | None | **High — directly integrated into AffBench pipeline for saliency-guided selection** |
| 22 | Yang et al. 2023 — *Grounding 3D Object Affordance from 2D Interactions* | S (3D point cloud) | 3D-AffordanceNet dataset | 3D affordance regions | Vision encoder (no LLM) | None | Low-Medium — 3D spatial grounding; orthogonal to AffBench's relational reasoning |
| 23 | Li et al. 2023 — *Beyond Object Recognition: Object Concept Learning* (OCL) | R (causal reasoning) | **OCL benchmark** (ICCV 2023) | Category + attribute + affordance + causal relations | None (OCRN model) | Causal intervention (interpretable model) | **High — closest benchmark to AffBench; lacks normative/danger dimensions** |
| 24 | Zhang et al. 2024 — *Inpaint2Learn: Self-Supervised Affordance Learning* | S (spatial) | In-the-wild images | Human pose, object location, 6D pose affordances | Inpainting model (no LLM) | None | Low-Medium — self-supervised; complementary to AffBench's zero-shot approach |
| 25 | Pan et al. 2025 — *ACKnowledge: Human Compatible Affordance Planning* | R (social + physical) | Real-world interaction logs | Physical + social norm affordances | LLM (GPT class, for planning) | Plan explanation | **High — social norm reasoning aligns with AffBench taxonomy codes 4 (Socially Awkward) and 5 (Socially Forbidden)** |
| 26 | Wang et al. 2025 — *ACE: Concept Editing in Diffusion Models* | N/A | Image generation benchmark | Concept labels | Diffusion model | None | None — out of scope |

---

## Key Observations

### Papers with highest relevance to AffBench

1. **Chuang et al. 2018 (ADE-Affordance)** — primary GT source for Experiment A
2. **Li et al. 2024 (OOAL)** — integrated module in AffBench pipeline; AGD20K as Experiment B GT
3. **Li et al. 2023 (OCL)** — nearest benchmark competitor; AffBench extends it with normative/danger taxonomy
4. **Pan et al. 2025 (ACKnowledge)** — social norm reasoning; motivates taxonomy codes 4-5
5. **Mirnateghi et al. 2024 (Explainability)** — uses GPT-4 for affordance explanation; AffBench generalises this
6. **Zhang et al. 2024 (SEA)** — embodied caption output format mirrors AffBench VLM output

### Gaps AffBench fills

- No prior benchmark jointly evaluates **multiple frontier VLMs** on **structured 7-way affordance taxonomy**
- No prior work uses **inter-model agreement** as a GT-free reliability signal for affordance judgments
- Social, normative, and safety affordance categories are underrepresented in existing detection benchmarks
- Text quality evaluation (BLEU-4, METEOR, ROUGE-L, CIDEr) for affordance exception explanations has not been applied at scale across frontier VLMs
