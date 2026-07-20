# Related Work — AffBench (LaTeX-ready draft)

> This draft is intended as a drop-in replacement for Section 2 of `paper/main.tex` (lines 147–237).
> Cite keys match `paper/refs.bib`; new keys are marked with [NEW].
> Word count: approximately 720 words.

---

```latex
\section{Related Work}
\label{sec:related}

\paragraph{Affordance theory and surveys.}
Gibson's ecological concept of affordance~\cite{gibson1979ecological} defines perception
as inherently action-oriented: agents do not see objects but the interaction possibilities
they offer.  This idea has inspired decades of computational work across cognitive science
and robotics~\cite{jamone2018affordances,zech2017computational}.  Jamone et al.\
\cite{jamone2018affordances} provide a multidisciplinary survey spanning psychological
theory, neuroscientific evidence, and robotic implementations, while Zech et al.\
\cite{zech2017computational} propose a systematic taxonomy of computational models along
dimensions of perception modality, affordance type, and learning mechanism.  AffBench
builds directly on this theoretical tradition: its 7-way label schema (Positive,
Firmly Negative, Object Non-functional, Physical Obstacle, Socially Awkward, Socially
Forbidden, Dangerous) operationalises the Gibson framework at the level of structured
relational judgment rather than pixel-level detection.

\paragraph{Affordance detection and segmentation.}
Early vision-based approaches detect interaction regions from RGB-D data using
geometric feature classifiers on tool parts~\cite{myers2015affordance}.  Deep learning
superseded this with CNN encoder-decoder architectures that predict dense affordance
maps~\cite{nguyen2016detecting}, subsequently improved by object-detector front-ends
and dense CRF post-processing for sharper boundaries~\cite{nguyen2017object}.
AffordanceNet~\cite{do2018affordancenet} unified object detection and affordance
segmentation in a single end-to-end network at real-time speed, establishing a
strong baseline for physical affordances (grasp, contain, support, cut).  More
recently, Chu et al.\ \cite{chu2020recognizing} connected category-agnostic
affordance recognition to symbolic PDDL planning, showing that affordance detection
can drive higher-level robot decision-making.  All of these methods produce spatial
affordance maps for \emph{where} to interact; they do not address the relational
judgment of \emph{whether} an action is appropriate and \emph{why} it may be
constrained by physical, social, or safety factors --- the question AffBench evaluates.

\paragraph{Open-vocabulary and learning-based affordances.}
A parallel line of work exploits vision-language alignment to generalise beyond fixed
affordance vocabularies.  Luo et al.\ \cite{luo2021oneshot} tackle one-shot affordance
detection, transferring action intent from a support image to new scenes.  OOAL
\cite{ooal} scales this to open-vocabulary settings by combining DINOv2 visual features
with CLIP text conditioning, outperforming fully-supervised baselines with under 1\%
of training data on AGD20K.  Grounding affordances in 3D from 2D interaction images
\cite{yang2023grounding3d} and self-supervised inpainting-based affordance discovery
\cite{zhang2024inpaint2learn} further expand the space of approaches.  We incorporate
OOAL into AffBench as a saliency-guided instance selector, but shift the evaluation
axis: where these models predict \emph{where} affordances exist, AffBench evaluates
whether frontier VLMs can reason about \emph{why} an affordance is or is not appropriate.

\paragraph{Affordance datasets and benchmarks.}
The ADE-Affordance dataset~\cite{ade_affordance}, built atop ADE20K, remains the
richest source of relational affordance supervision: it annotates object instances
with sit, run, and grasp labels and free-text explanations covering physical, social,
and safety exception categories.  This makes it the ground-truth source for AffBench
Experiment~A.  AGD20K~\cite{ooal} provides large-scale egocentric affordance grounding
maps for 36 seen and 25 unseen action categories.  The Object Concept Learning
benchmark (OCL)~\cite{li2023ocl} requires joint reasoning about category, attribute,
and affordance with causal relation annotations, making it the closest prior benchmark
to AffBench; however, OCL does not cover normative or danger affordance categories
and evaluates purpose-built models rather than frontier VLMs.  AffBench fills this
gap with an explicit 7-way taxonomy that extends the social and safety dimensions
present in ADE-Affordance, and evaluates three deployed frontier VLMs simultaneously.

\paragraph{Affordance reasoning with LLMs and VLMs.}
Language models carry substantial commonsense knowledge about object functions and
action consequences.  AffordanceLLM~\cite{affordancellm} combines language-model
reasoning with visual encoders, achieving strong generalisation to novel objects.
ACKnowledge~\cite{pan2025acknowledge} uses LLMs to plan human-compatible interactions
in real-world contexts, explicitly integrating social norm constraints --- directly
motivating AffBench's Socially Awkward and Socially Forbidden categories.  Engelbracht
et al.\ \cite{engelbracht2024spotlight} show VLMs can predict functional affordances
for robot interaction with over 80\% operational success.  ProbeAff~\cite{probeaff}
probes vision foundation models for the geometric interaction cues underlying affordance
reasoning, finding partial encoding but failures on fine-grained cases.  AffBench is
complementary: rather than probing or training affordance modules, it \emph{directly
benchmarks the structured reasoning output} of deployed frontier VLMs across a nuanced
7-way taxonomy, revealing where GPT-4.1-mini, Claude Sonnet, and Gemini 2.5 Pro
agree and diverge on affordance judgments.

\paragraph{Explainability in affordance systems.}
Bhattacharyya et al.\ \cite{bhattacharyya2023visual} demonstrate that high affordance
classification accuracy does not imply correct affordance understanding, motivating
the need for interpretability analysis.  Mirnateghi et al.\ \cite{mirnateghi2024explainability}
address this with a post-hoc framework: CAM heatmaps identify salient regions and
GPT-4 generates textual explanations of detector behaviour.  SEA
\cite{zhang2024sea} goes further, proposing \emph{self-explainable} affordance
learning where a model simultaneously outputs an action heatmap and an embodied
caption, evaluated with BLEU and METEOR metrics.  AffBench inherits this philosophy:
it requires VLMs to produce not just a taxonomy label but also a free-text exception
description and consequence prediction, which are scored with BLEU-4, METEOR, ROUGE-L,
and CIDEr against ADE-Affordance reference explanations.

\paragraph{Multi-model evaluation and inter-model agreement.}
Evaluating generative models without dense ground truth is an active methodological
challenge.  LLM-as-judge frameworks use one model to score another's outputs
\cite{zheng2023judging}, but introduce their own biases.  Inter-model agreement ---
the rate at which independently queried models converge on the same answer --- provides
a complementary GT-free signal: high agreement across diverse model families suggests
the judgment is grounded in shared, reliable visual evidence rather than idiosyncratic
model bias.  We report both pairwise and 3-way agreement alongside consensus accuracy,
providing a richer picture of where frontier VLMs are reliable versus uncertain on
affordance judgments.

\paragraph{Segment Anything.}
SAM~\cite{kirillov2023sam} enables class-agnostic, prompt-free instance segmentation
at scale, producing dense mask proposals from a single forward pass.  Combined with
open-vocabulary detectors in Grounded SAM~\cite{grounded_sam}, it has become a key
building block for open-world scene parsing.  We exploit SAM's automatic mask generator
as an annotation-free instance proposal mechanism, decoupling object discovery from
object classification and making our benchmark applicable to arbitrary real-world
scenes without category-specific detectors.  An ablation over $K\in\{5,10,20\}$
masks, with and without OOAL saliency filtering, quantifies the sensitivity of
VLM affordance judgments to instance selection strategy.
```

---

## New cite keys required (not yet in refs.bib)

| Key | Paper |
|---|---|
| `jamone2018affordances` | Jamone et al. 2018 — Affordances Survey, IEEE TCDS |
| `zech2017computational` | Zech et al. 2017 — Computational Models of Affordance, Adaptive Behavior |
| `nguyen2016detecting` | Nguyen et al. 2016 — Detecting Object Affordances with CNNs, IROS |
| `do2018affordancenet` | Do, Nguyen, Reid 2018 — AffordanceNet, ICRA |
| `chu2020recognizing` | Chu et al. 2020 — AffContext, arXiv |
| `luo2021oneshot` | Luo et al. 2021 — One-Shot Affordance Detection, IJCAI |
| `yang2023grounding3d` | Yang et al. 2023 — Grounding 3D Affordance, ICCV |
| `zhang2024inpaint2learn` | Zhang et al. 2024 — Inpaint2Learn, CVPR |
| `li2023ocl` | Li et al. 2023 — OCL Benchmark, ICCV |
| `pan2025acknowledge` | Pan et al. 2025 — ACKnowledge, CHI |
| `engelbracht2024spotlight` | Engelbracht et al. 2024 — SpotLight, arXiv |
| `mirnateghi2024explainability` | Mirnateghi et al. 2024 — Explainability of Affordance Learning, DICTA |
| `bhattacharyya2023visual` | Bhattacharyya et al. 2023 — Visual Affordance XAI, Springer |
| `zhang2024sea` | Zhang et al. 2024 — SEA: Self-Explainable Affordance, arXiv (ETH/NPU) |
