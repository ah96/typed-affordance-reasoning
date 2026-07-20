# Paper Summaries — AffBench Related Work

## Overview

This document covers all 30 PDF files found across the three subdirectories of the `related_work/` folder:
- `Initial Papers/` (19 PDFs)
- `Learning Affordances Papers/` (7 PDFs, including 4 duplicates from Initial Papers)
- `XAI Papers/` (4 PDFs, all duplicates from Initial Papers)

**Unique papers: 26** (4 papers appear in multiple folders).

---

## Topic Taxonomy

| # | Topic Bucket | Abbreviation |
|---|---|---|
| A | Affordance Theory & Surveys | THEORY |
| B | Affordance Detection & Segmentation | DETECT |
| C | Learning Affordances / Open-Vocabulary | LEARN |
| D | VLMs for Scene Understanding | VLM |
| E | Affordance Reasoning with LLMs | LLM-AFF |
| F | XAI & Explainability | XAI |
| G | Datasets & Benchmarks | DATA |
| H | Instance Segmentation / SAM | SAM |

---

## Per-Paper Summaries

---

### 1. Affordances in Psychology, Neuroscience, and Robotics: A Survey

**Authors:** Lorenzo Jamone, Emre Ugur, Angelo Cangelosi, Luciano Fadiga, Alexandre Bernardino, Justus Piater, José Santos-Victor  
**Venue/Year:** IEEE Transactions on Cognitive and Developmental Systems, 2018  
**Topic:** THEORY  
**Cite key:** `jamone2018affordances`

Gibson's 1966 concept of affordance is the lens through which this survey reviews four decades of work spanning ecological psychology, neuroscience, and robotic perception. The authors synthesise major theoretical formalizations and review neuroscientific evidence (mirror neurons, canonical neurons) that supports action-grounded object perception. A comprehensive robotics section covers learning, perception, and developmental systems that operationalize affordances for robot-environment interaction.

**Relation to AffBench:** Provides the theoretical backbone for the 7-way affordance taxonomy used in AffBench. Crucial for grounding the distinction between "positive" and the five negation categories in Gibson's ecological framework.

---

### 2. Computational Models of Affordance in Robotics: A Taxonomy and Systematic Classification

**Authors:** Philipp Zech, Simon Haller, Safoura Rezapour Lakani, Barry Ridge, Emre Ugur, Justus Piater  
**Venue/Year:** Adaptive Behavior, 2017  
**Topic:** THEORY  
**Cite key:** `zech2017computational`

After conducting a systematic literature review of affordance models in robotics, the authors propose a taxonomy rooted in the most prominent theoretical frameworks (Gibson, Chemero, Turvey). Each reviewed paper is classified along dimensions including affordance type, perception modality, learning mechanism, and evaluation setting. The paper identifies open research gaps and provides a roadmap for future work on robot autonomy.

**Relation to AffBench:** The taxonomy dimensions proposed here partially overlap with AffBench's 7-way label structure (positive vs. negation types). Useful reference for motivating a structured label taxonomy.

---

### 3. Affordance Detection of Tool Parts from Geometric Features

**Authors:** Austin Myers, Ching L. Teo, Cornelia Fermüller, Yiannis Aloimonos  
**Venue/Year:** ICRA 2015  
**Topic:** DETECT  
**Cite key:** `myers2015affordance`

Two supervised methods — superpixel-based hierarchical matching pursuit (S-HMP) and structured random forests (SRF) — predict affordance labels on tool parts from local RGB-D geometric features. A new dataset of kitchen, workshop, and garden tools with multi-labeled, ranked affordance annotations is introduced. Experiments demonstrate strong performance on cluttered scenes with occlusions and viewpoint changes.

**Relation to AffBench:** Pre-deep-learning affordance detection baseline. AffBench moves well beyond part-level spatial labeling to scene-level relational reasoning, representing a qualitatively different level of abstraction.

---

### 4. Detecting Object Affordances with Convolutional Neural Networks

**Authors:** Anh Nguyen, Dimitrios Kanoulas, Darwin G. Caldwell, Nikos G. Tsagarakis  
**Venue/Year:** IROS 2016  
**Topic:** DETECT  
**Cite key:** `nguyen2016detecting`

A CNN with encoder-decoder architecture learns affordance region maps from RGB-D images end-to-end. Multiple input modalities (RGB + depth) are fused to improve feature learning. The method outperforms geometric-feature baselines by 20% and is demonstrated on the WALK-MAN humanoid robot for grasp planning.

**Relation to AffBench:** Represents the CNN-era affordance detection baseline. AffBench asks VLMs to reason about affordances without being given object categories or spatial regions, a considerably harder task requiring language-grounded contextual understanding.

---

### 5. Object-Based Affordances Detection with Convolutional Neural Networks and Dense Conditional Random Fields

**Authors:** Anh Nguyen, Dimitrios Kanoulas, Darwin G. Caldwell, Nikos G. Tsagarakis  
**Venue/Year:** IROS 2017  
**Topic:** DETECT  
**Cite key:** `nguyen2017object`

Building on their 2016 work, the authors add an object-detector front-end to generate bounding-box proposals, then apply CNN features on those proposals followed by dense CRF post-processing to sharpen affordance boundary predictions. A new, more challenging RGB-D dataset is introduced and the pipeline is coupled to a grasping controller on WALK-MAN.

**Relation to AffBench:** The object-detector-plus-affordance-segmenter pipeline is conceptually related to AffBench's SAM-based instance discovery followed by VLM affordance query, but operates at the pixel level rather than the semantic reasoning level.

---

### 6. AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection

**Authors:** Thanh-Toan Do, Anh Nguyen, Ian Reid  
**Venue/Year:** ICRA 2018 (arXiv 2017)  
**Topic:** DETECT  
**Cite key:** `do2018affordancenet`

AffordanceNet jointly detects and localises multiple objects and their affordance regions in a single forward pass using two branches: object detection and pixel-wise affordance labeling. Key design choices include deconvolutional layers, a robust resizing strategy, and multi-task loss. Inference runs at 150 ms/image, suitable for real-time robotics.

**Relation to AffBench:** Establishes the benchmark for integrated object-affordance detection. AffordanceNet's affordance vocabulary (grasp, contain, support, etc.) covers physical affordances only; AffBench's taxonomy additionally covers social, normative, and danger categories.

---

### 7. Learning Affordance Segmentation: An Investigative Study

**Authors:** Chau Nguyen Duc Minh, Syed Zulqarnain Gilani, Syed Mohammed Shamsul Islam, David Suter  
**Venue/Year:** Edith Cowan University (journal TBD)  
**Topic:** DETECT  
**Cite key:** `duc2020learningseg`

This study identifies two limiting factors in affordance segmentation accuracy: backbone feature quality and information poverty in the Region Proposal Network. The paper proposes backbone replacement and a multiple-alignment strategy in the RPN, improving over prior state-of-the-art.

**Relation to AffBench:** Marginal relevance. Focuses narrowly on the visual segmentation sub-problem; AffBench operates at a higher semantic level of affordance judgment.

---

### 8. Affordance-Driven Visual Object Representation

**Authors:** Safoura Rezapour Lakani  
**Venue/Year:** PhD Dissertation, University of Innsbruck, 2018  
**Topic:** DETECT / THEORY  
**Cite key:** `lakani2018phd`

This dissertation proposes affordance-centric object representations for robotic manipulation and grasping in indoor scenes. It develops both part-based and holistic representations that tie visual features directly to action possibilities, validated through manipulation hardware experiments.

**Relation to AffBench:** Foundational work on connecting visual representations to affordance types. The part-based approach contrasts with AffBench's scene-level, language-mediated affordance reasoning.

---

### 9. Towards Affordance Detection for Robot Manipulation Using Affordance for Parts and Parts for Affordance

**Authors:** Safoura Rezapour Lakani, Antonio J. Rodríguez-Sánchez, Justus Piater  
**Venue/Year:** Autonomous Robots, 2019 (Springer)  
**Topic:** DETECT  
**Cite key:** `lakani2019parts`

A bidirectional RGB-D method where affordance labels inform part segmentation and part boundaries help localise affordances. Trained on the Myers et al. dataset, the method outperforms baselines on novel object instances by 14% on average and is validated in a real grasping scenario.

**Relation to AffBench:** Part-affordance co-inference at the pixel level; contrasts with AffBench's semantic, language-based reasoning about whether an action is appropriate in context.

---

### 10. Recognizing Object Affordances to Support Scene Reasoning for Manipulation Tasks

**Authors:** Fu-Jen Chu, Ruinian Xu, Chao Tang, Patricio A. Vela  
**Venue/Year:** arXiv 2020  
**Topic:** DETECT  
**Cite key:** `chu2020recognizing`

AffContext, a category-agnostic region proposal network with self-attention, predicts affordance labels for scene understanding linked to PDDL-based symbolic action planning. A key contribution is reducing the performance gap between object-agnostic and object-informed affordance recognition.

**Relation to AffBench:** Integration of affordance recognition into symbolic planning is conceptually analogous to AffBench's use of affordance judgments for agent decision-making, though AffBench uses natural language taxonomy labels rather than planning symbols.

---

### 11. Toward Affordance Detection and Ranking on Novel Objects for Real-World Robotic Manipulation

**Authors:** Fu-Jen Chu, Ruinian Xu, Landan Seguin, Patricio A. Vela  
**Venue/Year:** IEEE Robotics and Automation Letters, 2019  
**Topic:** DETECT  
**Cite key:** `chu2019ranking`

A region-based affordance segmentation system that additionally ranks detected affordances by KL-divergence, enabling non-primary affordances to supplement primary ones when needed. Category-agnostic design generalises to unseen objects.

**Relation to AffBench:** Affordance ranking is tangentially related to AffBench's taxonomy, where the 7 categories encode an implicit priority ordering (e.g., Dangerous overrides all others).

---

### 12. A4T: Hierarchical Affordance Detection for Transparent Objects Depth Reconstruction and Manipulation

**Authors:** Jiaqi Jiang, Guanqun Cao, Thanh-Toan Do, Shan Luo  
**Venue/Year:** IEEE Robotics and Automation Letters, 2022  
**Topic:** DETECT  
**Cite key:** `jiang2022a4t`

A4T uses a hierarchical AffordanceNet for transparent objects, where affordance maps encode relative positions of object parts and guide a multi-step depth reconstruction pipeline. A new TRANS-AFF dataset with depth and affordance labels for transparent objects is contributed.

**Relation to AffBench:** Niche sub-problem (transparent objects). The hierarchical affordance architecture is loosely related to AffBench's structured output, but the settings and goals differ substantially.

---

### 13. SpotLight: Robotic Scene Understanding through Interaction and Affordance Detection

**Authors:** Tim Engelbracht, René Zurbrügg, Marc Pollefeys, Hermann Blum, Zuria Bauer  
**Venue/Year:** arXiv 2024  
**Topic:** VLM / DETECT  
**Cite key:** `engelbracht2024spotlight`

SpotLight integrates VLM-based affordance prediction with scene graph construction to enable a robot to interact with functional elements (light switches). VLM predictions inform motion primitives, achieving 84% operation success. The robot learns through physical interaction, updating its scene graph with newly discovered relationships.

**Relation to AffBench:** One of the few papers to directly use VLMs for affordance prediction in robotic scenes, complementing AffBench by demonstrating the downstream utility of VLM affordance reasoning in real robot systems.

---

### 14. Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching

**Authors:** Andy Zeng, Shuran Song, Kuan-Ting Yu, et al.  
**Venue/Year:** International Journal of Robotics Research, 2022 (original 2019)  
**Topic:** DETECT  
**Cite key:** `zeng2022pickplace`

A complete robotic pick-and-place pipeline using pixel-wise affordance probability maps for four grasp primitives without task-specific training data. Object recognition uses cross-domain image matching against product images. The system handles a wide range of novel objects without retraining.

**Relation to AffBench:** Shows how pixel-level affordance maps drive physical robot actions. AffBench operates at the semantic reasoning level above physical grasping, evaluating appropriateness and generating explanations rather than predicting grasp locations.

---

### 15. Learning Affordance Segmentation for Real-World Robotic Manipulation via Synthetic Images

**Authors:** Fu-Jen Chu, Ruinian Xu, Patricio A. Vela  
**Venue/Year:** IEEE Robotics and Automation Letters, 2019  
**Topic:** DETECT  
**Cite key:** `chu2019synthetic`

A domain-adaptation framework trains affordance segmentation on auto-generated synthetic images and adapts to real-world data without supervision. Domain-invariant region proposal networks and task-level adaptation components close a 30% gap to fully supervised performance.

**Relation to AffBench:** Synthetic-to-real transfer for affordance learning is orthogonal to AffBench's zero-shot, language-mediated approach. AffBench avoids any domain-specific training entirely.

---

### 16. Learning to Act Properly: Predicting and Explaining Affordances from Images

**Authors:** Ching-Yao Chuang, Jiaman Li, Antonio Torralba, Sanja Fidler  
**Venue/Year:** CVPR 2018  
**Topic:** DATA / XAI  
**Cite key:** `chuang2018act` (= `ade_affordance` in refs.bib)

**The ADE-Affordance paper** — the primary ground-truth dataset for AffBench Experiment A. The paper introduces ADE-Affordance, built on ADE20K, annotating object instances with sit/run/grasp affordance labels and rich free-text exception explanations tied to physical, social, and safety constraints. A Graph Neural Network model propagates scene context to reason about per-object affordances.

**Relation to AffBench:** Direct predecessor. AffBench Experiment A evaluates VLMs against ADE-Affordance ground truth. AffBench's 7-way taxonomy extends the exception categories in ADE-Affordance into a structured classification schema.

---

### 17. Self-Explainable Affordance Learning with Embodied Caption

**Authors:** Zhipeng Zhang, Zhimin Wei, Guolei Sun, Peng Wang, Luc Van Gool  
**Venue/Year:** arXiv 2024 (ETH Zurich / NPU)  
**Topic:** XAI / LEARN  
**Cite key:** `zhang2024sea`

SEA introduces a self-explainable affordance learning framework where a model generates embodied captions alongside affordance heatmaps, bridging the gap between visual predictions and human-understandable explanations. A new dataset integrating images, heatmaps, and natural language captions is released.

**Relation to AffBench:** The embodied caption idea directly parallels AffBench's requirement that VLMs produce not just a label but also a textual explanation (exception description + consequence). SEA evaluates BLEU/METEOR-style metrics on captions, which AffBench also proposes for evaluating explanation quality.

---

### 18. Towards Explainability of Affordance Learning in Robot Vision

**Authors:** Nima Mirnateghi, Syed Mohammed Shamsul Islam, Syed Afaq Ali Shah  
**Venue/Year:** DICTA 2024 (IEEE)  
**Topic:** XAI  
**Cite key:** `mirnateghi2024explainability`

A post-hoc multimodal explainability framework generates Class Activation Map (CAM) heatmaps for affordance predictions and then queries GPT-4 to generate textual explanations. Evaluated on a large-scale multi-view RGB dataset, the framework demonstrates that LLMs can articulate the behaviour of affordance learning systems in zero-shot fashion.

**Relation to AffBench:** Directly relevant — uses GPT-4 to explain affordance predictions using visual saliency as a bridge. AffBench inverts this: it uses VLMs to make and explain affordance judgments from scratch, rather than post-hoc explaining existing detector outputs.

---

### 19. Visual Affordance Recognition: A Study on Explainability and Interpretability for Human Robot Interaction

**Authors:** Rupam Bhattacharyya, Alexy Bhowmick, Shyamanta M. Hazarika  
**Venue/Year:** Springer (book chapter / workshop, year TBD)  
**Topic:** XAI  
**Cite key:** `bhattacharyya2023visual`

Three pretrained vision models are tested on object affordance classification (modified CAD-120 dataset) without affordance heatmaps as training signal. Predictions are post-hoc explained using Smooth Grad-CAM++. The study highlights that high accuracy does not imply correct affordance understanding, motivating the need for interpretability frameworks.

**Relation to AffBench:** The core insight — that performance metrics alone do not reveal whether a model truly understands affordance — directly motivates AffBench's inter-model agreement and explanation quality metrics as complementary signals.

---

### 20. One-Shot Affordance Detection

**Authors:** Hongchen Luo, Wei Zhai, Jing Zhang, Yang Cao, Dacheng Tao  
**Venue/Year:** IJCAI 2021 (arXiv)  
**Topic:** LEARN  
**Cite key:** `luo2021oneshot`

Given a support image showing an action purpose, the OS-AD network transfers the action intent to detect all objects affording the same action in scene images. A new Purpose-driven Affordance Dataset (PAD) of 4k images across 31 affordances and 72 object categories is contributed.

**Relation to AffBench:** Predecessor to OOAL. AffBench uses OOAL (which generalises this to open-vocabulary via foundation models) for saliency-guided instance selection.

---

### 21. One-Shot Open Affordance Learning with Foundation Models (OOAL)

**Authors:** Gen Li, Deqing Sun, Laura Sevilla-Lara, Varun Jampani  
**Venue/Year:** CVPR 2024  
**Topic:** LEARN / VLM  
**Cite key:** `ooal`

OOAL trains on just one example per base object category and generalises to novel objects and affordances using DINOv2 visual features and CLIP text embeddings. Experiments on two affordance segmentation benchmarks (including AGD20K) show competitive results with less than 1% of full training data.

**Relation to AffBench:** OOAL is directly integrated into AffBench's pipeline as the saliency-guided instance selection module. The SAM-only vs. SAM+OOAL comparison is a key ablation. AGD20K used by OOAL provides spatial ground truth for AffBench Experiment B.

---

### 22. Grounding 3D Object Affordance from 2D Interactions in Images

**Authors:** Yuhang Yang, Wei Zhai, Hongchen Luo, Yang Cao, Jiebo Luo, Zheng-Jun Zha  
**Venue/Year:** ICCV 2023  
**Topic:** LEARN  
**Cite key:** `yang2023grounding3d`

The paper introduces grounding 3D affordance regions from 2D interaction images, addressing the gap between 2D demonstrations and 3D geometric affordance prediction. An intention-driven module (IDAR) aligns 2D interaction cues with 3D point-cloud affordance regions.

**Relation to AffBench:** 3D affordance grounding is orthogonal to AffBench's scene-level relational reasoning, but the human-interaction-driven approach shares the insight that affordance understanding benefits from modelling agent intent.

---

### 23. Beyond Object Recognition: A New Benchmark towards Object Concept Learning

**Authors:** Yong-Lu Li, Yue Xu, Xinyu Xu, Xiaohan Mao, Yuan Yao, Siqi Liu, Cewu Lu  
**Venue/Year:** ICCV 2023  
**Topic:** DATA / LEARN  
**Cite key:** `li2023ocl`

The Object Concept Learning (OCL) benchmark requires models to jointly reason about object category, attribute, and affordance, plus the causal relations between these three levels. OCRN uses concept instantiation and causal intervention to infer all three levels simultaneously.

**Relation to AffBench:** OCL is the closest benchmark to AffBench in jointly requiring contextual affordance reasoning. AffBench specifically adds normative and danger dimensions absent in OCL, and evaluates frontier VLMs rather than purpose-built models.

---

### 24. Inpaint2Learn: A Self-Supervised Framework for Affordance Learning

**Authors:** Lingzhi Zhang, Weiyu Du, Shenghao Zhou, Jiancong Wang, Jianbo Shi  
**Venue/Year:** CVPR (~2024), University of Pennsylvania  
**Topic:** LEARN  
**Cite key:** `zhang2024inpaint2learn`

Inpaint2Learn uses image inpainting as a self-supervised scaffold to automatically generate affordance labels without human annotation. Applied to three tasks: human affordance prediction (pose placement), Location2Object (object insertion), and 6D object pose hallucination.

**Relation to AffBench:** Self-supervised affordance learning via inpainting is complementary to AffBench's zero-shot VLM approach. Both avoid manual affordance labels but via different strategies.

---

### 25. ACKnowledge: A Computational Framework for Human Compatible Affordance-based Interaction Planning in Real-world Contexts

**Authors:** Ziqi Pan, Xiucheng Zhang, Zisu Li, Zhenhui Peng, Mingming Fan, Xiaojuan Ma  
**Venue/Year:** CHI 2025  
**Topic:** LLM-AFF / VLM  
**Cite key:** `pan2025acknowledge`

ACKnowledge uses LLMs to plan human-compatible affordance-based interactions in real-world contexts, integrating knowledge about social norms and user preferences to generate interaction plans that respect both physical affordances and social constraints.

**Relation to AffBench:** Highly relevant — ACKnowledge explicitly reasons about social norms and human compatibility in affordance-based planning, aligning with AffBench's "Socially Awkward" and "Socially Forbidden" taxonomy categories. ACKnowledge is an application framework; AffBench evaluates the underlying VLM reasoning capability.

---

### 26. ACE: Concept Editing in Diffusion Models without Performance Degradation

**Authors:** Ruipeng Wang, Junfeng Fang, Jiaqi Li, Hao Chen, Jie Shi, Kun Wang, Xiang Wang  
**Venue/Year:** ACM Multimedia 2025  
**Topic:** OUT OF SCOPE  
**Cite key:** `wang2025ace`

ACE proposes a method for editing semantic concepts in diffusion models without degrading generation performance.

**Relation to AffBench:** Not directly related to affordance reasoning or VLM evaluation. Likely included in the folder by mistake.

---

## Proposed Subfolder Organization

The following subfolder structure is proposed for the PDF files. **No files should be moved** — this is a proposed structure only.

```
related_work/
├── 01_Theory_Surveys/
│   ├── Affordances in Psychology, Neuroscience, and Robotics_ A Survey.pdf
│   └── Computational models of affordance in robotics_ a taxonomy and systematic classification.pdf
│
├── 02_Affordance_Detection_Segmentation/
│   ├── Affordance Detection of Tool Parts from Geometric Features.pdf
│   ├── Detecting Object Affordances with Convolutional Neural Networks.pdf
│   ├── Object-Based Affordances Detection with Convolutional Neural Networks and Dense Conditional Random Fields.pdf
│   ├── AffordanceNet_ An End-to-End Deep Learning Approach for Object Affordance Detection.pdf
│   ├── Learning Affordance Segmentation_ An Investigative Study.pdf
│   ├── Affordance-Driven Visual Object Representation.pdf
│   ├── Towards affordance detection for robot manipulation using affordance for parts and parts for affordance.pdf
│   ├── Recognizing object affordances to support scene reasoning for manipulation tasks.pdf
│   ├── Toward Affordance Detection and Ranking on Novel Objects for Real-World Robotic Manipulation.pdf
│   └── A4T_ Hierarchical Affordance Detection for Transparent Objects Depth Reconstruction and Manipulation.pdf
│
├── 03_Robotic_Applications/
│   ├── SpotLight_ Robotic Scene Understanding through Interaction and Affordance Detection.pdf
│   ├── Robotic pick-and-place of novel objects in clutter with multi-affordance grasping and cross-domain image matching.pdf
│   └── Learning Affordance Segmentation for Real-World Robotic Manipulation via Synthetic Images.pdf
│
├── 04_Learning_Open_Vocabulary/
│   ├── One-Shot Affordance Detection.pdf
│   ├── One-Shot Open Affordance Learning with Foundation Models.pdf
│   ├── Grounding 3D Object Affordance from 2D Interactions in Images.pdf
│   ├── Inpaint2Learn_ A Self-Supervised Framework for Affordance Learning.pdf
│   └── ACKnowledge_ A Computational Framework for Human Compatible Affordance-based Interaction Planning in Real-world Contexts.pdf
│
├── 05_Datasets_Benchmarks/
│   ├── Learning to Act Properly_ Predicting and Explaining Affordances from Images.pdf  [ADE-Affordance]
│   └── Beyond Object Recognition_ A New Benchmark towards Object Concept Learning.pdf   [OCL]
│
├── 06_XAI_Explainability/
│   ├── Self-Explainable Affordance Learning with Embodied Caption.pdf
│   ├── Towards Explainability of Affordance Learning in Robot Vision.pdf
│   └── Visual Affordance Recognition - A Study on Explainability and Interpretability for Human Robot Interaction.pdf
│
└── 07_Other/
    └── ACE_ Concept Editing in Diffusion Models without Performance Degradation.pdf
```
