# 2 Literature review

Deploying deep learning in medical image analysis is constrained above all by data: patient-privacy
regulation, the cost of expert annotation, and the rarity of many pathologies together cap the
labelled data that any single institution can accumulate. Synthetic image generation has become a
leading response to this bottleneck, and diffusion models have largely displaced generative
adversarial networks as the generative backbone of choice because of their more stable training
and their better preservation of fine anatomical detail. A closely related obstacle is domain
shift, in which a model trained at one institution degrades when deployed at another owing to
differences in scanner vendor, acquisition protocol, and patient population. This has produced a
growing body of synthetic-to-real (S2R) work in which diffusion-generated images, whose labels are
known by construction, are used to train or adapt models intended for real, unseen clinical data.
Most of this work is concerned with generation itself: image fidelity, structural conditioning,
and domain alignment. The question addressed in this thesis arises afterwards, given a large pool
of generated images of uneven quality: which of them should be admitted to a real training set,
and in what per-class balance. In line with supervisory guidance, the review below is restricted
to work published from 2024 onward; the few earlier papers that are the original source of a
technique implemented here are named once, for attribution only, at the end of the section.

Recent generation-side research continues to focus on aligning domains before or during synthesis.
Ji and Chung proposed Stochastic Step Alignment, a diffusion-based unsupervised domain adaptation
method that aligns the intermediate trajectory of the denoising process between source and target
domains rather than only the final image [3]. They reported Dice similarity coefficients of 86.5%
for MR-to-CT and 88.3% for CT-to-MR abdominal segmentation on the CHAOS benchmark, outperforming
previous domain-adaptation approaches. Zeng et al. addressed the stricter, privacy-driven setting
in which source pixels cannot be shared, introducing Reliable Source Approximation, which uses
edge-conditioned diffusion to synthesize source-like images and then applies an uncertainty and
prediction-consistency filter to retain only reliable pseudo-labels before fine-tuning [4]. Their
method achieved 77.83% Dice on vestibular schwannoma segmentation without ever accessing source
data. Gong et al. separated appearance alignment from structural alignment in a 3D diffusion
framework for volumetric segmentation, reporting accuracy close to a target-trained upper bound
[14]. Cho et al. released MediSyn, a generalist text-guided latent diffusion model spanning ten
imaging modalities and six medical specialties, and showed through blinded physician review that
its outputs were frequently indistinguishable from real images while improving downstream
classifier performance in data-scarce settings [1]. In the chest-radiograph domain most relevant
to this thesis, Prakash et al. conditioned a latent diffusion model on text and segmentation masks
and used proxy modelling together with radiologist feedback to raise the training utility of
synthetic CheXpert images, reporting F1 gains of up to 0.150 for classification [15]. Their
emphasis, however, is on the conditioning process rather than on selecting individual images. The
CXR-LT 2024 challenge consolidated long-tailed, multi-label, and zero-shot chest-radiograph
methods, several of which use generative augmentation for rare findings, but as a benchmark it
prescribes no selection procedure [16].

A recurring finding in this recent work is that generation-quality metrics are poor proxies for
clinical usefulness. A 2025 review in The Lancet Digital Health argued that downstream task
performance, not the Fréchet Inception Distance (FID), must be the acceptance criterion for
medical synthetic data [6]. An empirical study on retinal image synthesis showed directly that a
lower FID does not translate into better downstream classification or segmentation when the
synthetic images are used for augmentation [7]. Domain-adapted alternatives have been proposed,
such as the Fréchet Radiomic Distance, which compares real and synthetic medical datasets in a
space of quantitative radiomic features [17], but these remain distributional metrics rather than
per-image signals. Taken together, recent generation-side work either aligns domains at synthesis
time or filters generated data by a single criterion, and none learns which individual synthetic
images most improve a downstream multi-label classifier.

A smaller but rapidly growing body of work addresses the post-generation selection problem
directly. Wang et al. observed that diffusion-generated chest radiographs contain noisy and
uninformative samples, and proposed Informative Data Selection, an Information-Bottleneck-grounded
scheme that assigns a higher training weight to more informative synthetic images without
discarding any [2]. Their approach shares the modality, the generator family, and the goal of
this thesis, but differs in three respects: it uses a single information-theoretic criterion
rather than several heterogeneous signals; it re-weights softly rather than making a hard
admission decision; and it applies no per-class adaptive threshold. Tang et al. proposed a
training-free selection method for semantic segmentation that combines a perturbation-based CLIP
similarity score with a class-balanced annotation-similarity filter, halving the synthetic dataset
while improving the segmenter [8]. This is a close analogue of the similarity front end used
here, but relies on CLIP alone, targets segmentation rather than multi-label classification, and
learns neither a ranking model nor a threshold. Zhang et al. used a classifier-predicted-likelihood
semantic-quality metric to filter synthetic fundus images, adding them per class to
under-represented diabetic-retinopathy grades, and reported an increase in balanced accuracy from
66.8% to 74.2% [9]. Their result validates both the intended-label-agreement signal and the
per-class handling adopted in this thesis, although they use one signal and a fixed per-class
quota rather than a learned threshold. In a non-diffusion setting, the CosSIF method removes
synthetic skin-lesion images whose cosine-similarity feature distributions are less
class-discriminative than those of real images, improving downstream AUC by 1.88% on ISIC-2016
[18], which motivates a within-class-versus-between-class similarity cue.

Two further studies frame the selection problem theoretically. Rezaei et al. showed analytically
that, for linear models, the covariance shift between the target and synthetic distributions
drives generalisation error while the mean shift does not, and that matching the covariance is a
near-optimal selection criterion that also transfers to deep networks [19]. Their work supplies a
principled single-criterion baseline against which a multi-signal approach can be contrasted.
Nguyen et al. proposed Targeted Diffusion Augmentation, which generates faithful synthetic images
only for the real training examples that a model fails to learn early, and showed that augmenting
only 30–40% of the data outperforms augmenting everything [12]. Despite its title, the method
selects which real examples to augment rather than which generated images to admit, a distinction
this thesis states explicitly. Kolossov et al. proved that training on a well-chosen subset can
beat training on the full sample, and that unbiased-reweighting and influence-function selectors
can be substantially sub-optimal [13], which provides the formal justification for comparing a
selected synthetic subset against the use of all synthetic data.

The ranking component of the proposed framework draws on recent work in learned set functions and
data valuation. Xie et al. showed that when the utility of a subset depends on the ground set it
is drawn from, the subset representation must incorporate a permutation-invariant sufficient
statistic of that superset, and that this term accounts for most of the observed gain, with the
largest improvements when subsets are small relative to the pool [10]. That regime matches the
roughly one hundred measured subsets used to train the set-utility model here. The DUPRE framework
makes the same efficiency argument that motivates this thesis's selection stage, learning a model
that predicts subset utility so that a proxy classifier need not be retrained for every candidate
subset [20]. For the per-image target, Sun et al. introduced an out-of-bag marginal-contribution
estimator that is fast and robust to the stochasticity of training and that detects fine-grained
outliers [21]. Xu et al. valued a whole data distribution from small samples using a
maximum-mean-discrepancy formulation, supporting the premise that data carry unequal value but
operating at a coarser granularity than the per-image mechanism used here [22]. Chi et al. unified
Shapley-, Banzhaf-, and leave-one-out-type values and optimised them for the selection objective
directly [23]. Yang et al. provided recent, training-free evidence that fusing a
similarity-matching score with an image-quality-assessment term outperforms either alone [24], and
Loizou and Tsoumakos showed that chunk-level contribution can be estimated with single-iteration
proxies at very large speed-ups, paralleling the bounded-budget subset measurement adopted here
[25]. The Set-LLM architecture demonstrates that architectural permutation-invariance with formal
guarantees remains an active design principle for set-structured inputs [26].

The adaptive-threshold component builds on recent class-adaptive thresholding in semi-supervised
learning, adapted from a per-training-step mechanism into a one-time data-curation decision. The
AllMatch method applies class-adaptive confidence thresholds that exploit all unlabelled data
[27]. Zhao et al. decomposed the per-class threshold into a global term and a local term and added
an explicit class-fairness regulariser that prevents rare classes from being starved of
pseudo-labels [11]; this directly motivates the range-normalised per-class leniency and the hard
per-class selection floor used in this thesis. For the distinctiveness signal, Tan et al.
formulated coreset selection as maximising information content minus an importance-weighted
redundancy term defined by pairwise similarity, solved as a discrete quadratic program [28], and
Maharana et al. showed through graph message passing that raw diversity must be balanced against
per-sample difficulty [29]. Finally, and specific to the medical setting, Dar et al. demonstrated
that unconditional latent diffusion models replicate or near-replicate real training patients,
with near-duplication detectable at a DINOv2 cosine similarity above 0.95 [5]. Their result
grounds the near-duplicate rejection threshold used in this thesis in an independent source rather
than a hand-chosen constant.

Despite this progress, there remain clear shortcomings across the recent literature. Existing
methods for selecting synthetic data act on a single criterion, whether an information-theoretic
bound [2], a CLIP similarity score [8], a classifier likelihood [9], or a distributional
distance [19], and none combines several complementary perceptual signals. Most operate at the
level of the individual image or the whole distribution and do not learn a set-level utility that
is then distilled onto individual images; where a marginal-contribution value is used it is
computed for general classification tasks [21, 23] rather than for synthetic medical images.
Class-adaptive thresholds are studied almost exclusively as a per-training-step mechanism for
semi-supervised learning [27, 11] rather than as a one-time curation decision for a multi-label
medical classifier. Few studies use predictive uncertainty or explanation quality as an explicit
selection signal, and most report predictive accuracy while providing limited interpretability,
which reduces their usefulness for clinical decision support. Therefore, there remains a research
gap in developing a multi-signal, class-adaptive, and interpretable framework for selecting
diffusion-generated medical images for a downstream multi-label classifier.

To address these limitations, this thesis proposes ASISM (Adaptive Quality-Aware Synthetic Data
Selection), a framework that operates on the output of a LoRA-fine-tuned diffusion foundation
model. Each synthetic image is scored on six complementary signals: DINOv2 k-nearest-neighbour
similarity to real references, no-reference and domain-specific image quality, Monte-Carlo-dropout
predictive uncertainty, Grad-CAM alignment with expected pathology regions, agreement between the
generated image and its intended labels, and within-class distinctiveness. Each signal must pass an
independent go/no-go admission gate before it can enter the model. A permutation-invariant
set-utility network is trained on measured subset-level AUROC changes, and its value is distilled
into a small multi-signal ranking network that produces one scalar utility score per image; the
distillation target is a size-normalised leave-one-out marginal, with a Data-Banzhaf
Maximum-Sample-Reuse estimate computed from the same measured subsets as a robust alternative. A
per-class admission threshold is then decided once, before final classifier training, under a
three-tier governance rule that uses a learned threshold network only where sufficient
proxy-verified, image-disjoint evidence exists and otherwise falls back to a verified proxy
threshold or a fixed baseline. All tuning decisions are made on a dedicated proxy split, and the
final evaluation split is never opened during selection. Table 1 summarises the reviewed methods,
their strengths, and their limitations relative to this problem.

## Table 1  Summary of the reviewed 2024–2026 methods, strengths, and limitations

| Authors | Method | Strength | Limitation relative to this thesis |
|---|---|---|---|
| Cho et al. [1] | MediSyn (generalist text-guided latent diffusion) | Broad modality coverage; physician-rated realism | Generation only; no post-generation selection |
| Wang et al. [2] | Informative Data Selection (IB re-weighting) | Same modality and goal; down-weights uninformative synthetic CXR | One criterion; soft re-weighting not hard admission; no per-class threshold; no learned set-utility |
| Ji and Chung [3] | Stochastic Step Alignment (diffusion UDA) | Aligns the denoising trajectory between domains; strong cross-modality Dice | Alignment at generation time; no per-image admission decision |
| Zeng et al. [4] | Reliable Source Approximation (source-free UDA) | Uncertainty and consistency filter for reliable pseudo-labels; no source access | One filtering criterion; segmentation; filters pseudo-labels, not generated images by downstream effect |
| Dar et al. [5] | Memorisation analysis of medical LDMs | Shows near-duplication at DINOv2 cosine > 0.95 | A diagnostic study, not a selection framework |
| The Lancet Digital Health review [6] | Review of generative AI for medical synthesis | Argues downstream performance, not FID, is the right acceptance criterion | Review; proposes no method |
| Retinal FID study [7] | Empirical analysis of FID for augmentation | Shows lower FID does not imply better downstream performance | Diagnostic; not a selector |
| Tang et al. [8] | Training-free CLIP-based selection | Perturbation-based CLIP similarity + class-balanced filter; halves the set | CLIP-only (two filters); segmentation; no learned ranker; no threshold learning |
| Zhang et al. [9] | Classifier-likelihood filtering for DR grading | Per-class filtering of synthetic fundus images; balanced accuracy 66.8 → 74.2% | One signal; fixed per-class quota, not a learned threshold; no learned ranker; no valuation |
| Xie et al. [10] | Superset-conditioned neural subset selection | Permutation-invariant superset statistic is the main source of gain | Drug-discovery subsets; no downstream-AUROC target; no thresholds |
| Zhao et al. [11] | SST self-adaptive thresholding | Global × local per-class threshold with a class-fairness regulariser | Per-training-step SSL; not a one-time curation threshold |
| Nguyen et al. [12] | Targeted Diffusion Augmentation | Augmenting 30–40% of the data beats augmenting everything | Selects which real examples to augment, not which generated images to admit |
| Kolossov et al. [13] | Statistical theory of data selection | Proves a well-chosen subset can beat the full sample | Theory; surrogate-guided; not multi-signal, not synthetic/medical |
| Gong et al. [14] | Diffuse-UDA (appearance + structure aligned 3D diffusion) | Approaches the target-trained upper bound in 3D segmentation | No selection of generated samples by downstream effect |
| Prakash et al. [15] | Mask-conditioned LDM for chest radiographs | Proxy modelling and radiologist feedback raise synthetic utility; CheXpert | Conditioning-focused; no per-image selection framework |
| CXR-LT 2024 [16] | Long-tailed multi-label CXR challenge | Consolidates generative augmentation methods for rare findings | A benchmark, not a selection method |
| Fréchet Radiomic Distance [17] | Radiomic-feature distributional metric | Domain-adapted real-vs-synthetic dataset comparison | A metric, not a per-image signal |
| CosSIF [18] | Cosine-similarity image filtering | Removes less class-discriminative synthetic images; +1.88% AUC (ISIC-2016) | Single modality (skin); GAN generator; one criterion |
| Rezaei et al. [19] | Covariance-matching synthetic-data selection | Analytic criterion; transfers from linear models to deep networks | Single distribution-level criterion; not per-image; no thresholds |
| DUPRE [20] | Data utility prediction for valuation | Predicts subset utility to avoid per-subset retraining | General valuation; no perceptual signals; no ranking distillation |
| Sun et al. [21] | 2D-OOB joint valuation | Out-of-bag marginal-contribution estimator; fast; robust to stochastic training | General/tabular; per-sample; no set-utility distillation; no thresholds |
| Xu et al. [22] | Data distribution valuation | MMD-based valuation of a distribution from small samples | Distribution-level, not per-image; not selection-oriented |
| Chi et al. [23] | Unified data values for selection | Optimises Shapley/Banzhaf/LOO values for the selection objective | General classification; not synthetic/medical; no perceptual signals |
| Yang et al. [24] | GMValuator (similarity + IQA valuation) | Training-free fusion of similarity matching and image-quality assessment | Values real training data for a generator, not generated images for a classifier; two signals |
| Loizou and Tsoumakos [25] | Chunked Data Shapley | Chunk-level contribution with single-iteration proxies; large speed-ups | Tabular focus; no perceptual signals |
| Set-LLM [26] | Permutation-invariant architecture | Formal permutation-invariance guarantees for set inputs | Text-set setting; not utility regression over image sets |
| AllMatch [27] | Class-adaptive SSL thresholding | Class-adaptive thresholds using all unlabelled data | Per-training-step SSL loop; single-label |
| Tan et al. [28] | InfoMax data pruning | Coreset = information − importance-weighted redundancy, as a QP | Single objective; no downstream-utility target; no thresholds |
| Maharana et al. [29] | D² Pruning | Graph message passing balancing diversity and difficulty | Real-data pruning; no synthetic/medical; no learned utility model |

## References

[1] Cho C, et al. MediSyn: a generalist text-guided latent diffusion model for synthetic medical
image generation. arXiv:2405.09806; 2024.

[2] Wang Y, Goel R, Jojic M, Silva AC, Wu T, Yang Y. Informative synthetic data generation for
thorax disease classification. Conference on Uncertainty in Artificial Intelligence (UAI). PMLR
vol. 286; 2025. p. 4489–514.

[3] Ji W, Chung ACS. Stochastic step alignment for diffusion-based unsupervised domain adaptation
in medical image segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI);
2024.

[4] Zeng Y, et al. Reliable source approximation: source-free unsupervised domain adaptation for
vestibular schwannoma MRI segmentation. Medical Image Computing and Computer-Assisted Intervention
(MICCAI); 2024.

[5] Dar SUH, et al. Unconditional latent diffusion models memorize patient imaging data:
implications for openly sharing synthetic data. Nature Biomedical Engineering; 2025.
doi:10.1038/s41551-025-01468-8.

[6] Generative artificial intelligence in medical image synthesis: opportunities, challenges, and
future directions. The Lancet Digital Health; 2025.

[7] A pragmatic note on evaluating generative models with Fréchet Inception Distance for retinal
image synthesis. arXiv:2502.17160; 2025.

[8] Tang H, Yu S, Pang J, Zhang B. A training-free synthetic data selection method for semantic
segmentation. Proceedings of the AAAI Conference on Artificial Intelligence; 2025.
doi:10.1609/aaai.v39i7.32777. arXiv:2501.15201.

[9] Zhang H, Heinke A, Nagel ID, Bartsch DUG, Freeman WR, Nguyen TQ, An C. Class-conditioned
image synthesis with diffusion for imbalanced diabetic retinopathy grading. Medical Image
Computing and Computer-Assisted Intervention (MICCAI); 2025.

[10] Xie B, Bian Y, Zhou K, Chen Y, Zhao P, Han B, Meng W, Cheng J. Enhancing neural subset
selection: integrating background information into set representations. International Conference on
Learning Representations (ICLR); 2024. arXiv:2402.03139.

[11] Zhao S, Huang H, Li X, Chen X, Wang R. SST: self-training with self-adaptive thresholding for
semi-supervised learning. Information Processing & Management. 2025;62(5):104158. arXiv:2506.00467.

[12] Nguyen D, Li J, Zheng J, Mirzasoleiman B. Do we need all the synthetic data? Targeted image
augmentation via diffusion models. International Conference on Learning Representations (ICLR); 2026
(accepted). arXiv:2505.21574.

[13] Kolossov G, Montanari A, Tandon P. Towards a statistical theory of data selection under weak
supervision. International Conference on Learning Representations (ICLR); 2024 (Oral).
arXiv:2309.14563.

[14] Gong H, et al. Diffuse-UDA: addressing unsupervised domain adaptation in medical image
segmentation with appearance and structure aligned diffusion models. arXiv:2408.05985; 2024.

[15] Prakash E, Valanarasu JMJ, Chen Z, Reis EP, Johnston A, Pareek A, Bluethgen C, Gatidis S,
Olsen C, Chaudhari A, Ng A, Langlotz C. Evaluating and improving the effectiveness of synthetic
chest X-rays for medical image analysis. Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV) Workshops; 2025. p. 4413–21.

[16] CXR-LT 2024: a MICCAI challenge on long-tailed, multi-label, and zero-shot disease
classification from chest X-ray. MICCAI 2024 challenge report; arXiv:2506.07984.

[17] Fréchet Radiomic Distance (FRD): a versatile metric for comparing medical imaging datasets.
arXiv:2412.01496; 2024.

[18] CosSIF: cosine similarity-based image filtering to overcome low inter-class variation in
synthetic medical image datasets. Computers in Biology and Medicine. 2024;172:108317.
arXiv:2307.13842.

[19] Rezaei P, et al. High-dimensional analysis of synthetic data selection. International
Conference on Learning Representations (ICLR); 2026 (accepted). arXiv:2510.08123.

[20] DUPRE: data utility prediction for efficient data valuation. International Conference on
Autonomous Agents and Multiagent Systems (AAMAS); 2025.

[21] Sun Y, Shen X, Kwon Y. 2D-OOB: attributing data contribution through joint valuation
framework. Advances in Neural Information Processing Systems (NeurIPS); 2024. arXiv:2408.03572.

[22] Xu X, Wu Z, Sim RHL, Low BKH, et al. Data distribution valuation. Advances in Neural
Information Processing Systems (NeurIPS); 2024. arXiv:2410.04386.

[23] Chi H, et al. Unifying and optimizing data values for selection via sequential
decision-making. International Conference on Machine Learning (ICML); 2026 (Spotlight; accepted).
arXiv:2502.04554.

[24] Yang J, Deng W, et al. GMValuator: similarity-based data valuation for generative models.
International Conference on Learning Representations (ICLR); 2025. OpenReview:WncnpvJk83.
arXiv:2304.10701.

[25] Loizou A, Tsoumakos D. Chunked Data Shapley: a scalable dataset quality assessment for
machine learning. ACM International Conference on Information and Knowledge Management (CIKM); 2025.
doi:10.1145/3746252.3761305. arXiv:2508.16255.

[26] Set-LLM: a permutation-invariant LLM. Advances in Neural Information Processing Systems
(NeurIPS); 2025. arXiv:2505.15433.

[27] AllMatch: exploiting all unlabeled data for semi-supervised learning. International Joint
Conference on Artificial Intelligence (IJCAI); 2024.

[28] Tan H, Wu S, Huang W, Zhao S, Qi X. Data pruning by information maximization. International
Conference on Learning Representations (ICLR); 2025. OpenReview:93XT0lKOct. arXiv:2506.01701.

[29] Maharana A, Yadav P, Bansal M. D² pruning: message passing for balancing diversity and
difficulty in data pruning. International Conference on Learning Representations (ICLR); 2024.
arXiv:2310.07931.

## Method-origin references (attribution only, pre-2024)

The following papers are the original source of a technique implemented in this thesis. They are
cited once, in the methodology, solely so that the framework can attribute its own components; they
carry no claim about the current state of the art, and the operative citation in each case is the
2024–2026 paper indicated. Zaheer et al. introduced the permutation-invariant set function
`ρ(Σ φ(x))` (Deep Sets, NeurIPS 2017), applied here through the recent set-utility work of Xie et
al. [10] and Set-LLM [26]. Wang and Jia introduced the Data-Banzhaf value and its
Maximum-Sample-Reuse estimator (AISTATS 2023), used here in the form recently unified for selection
by Chi et al. [23] and Sun et al. [21]. Class-adaptive confidence thresholding originates with
FlexMatch (Zhang et al., NeurIPS 2021) and FreeMatch (Wang et al., ICLR 2023), adapted here
following AllMatch [27] and Zhao et al. [11]. Grad-CAM (Selvaraju et al., ICCV 2017) is used only
as a tool; no 2024–2026 work uses explanation–region alignment as a data-selection signal, which
is one contribution of this thesis. Multi-gate Mixture-of-Experts (KDD 2018) and gradient-based
multi-objective optimisation work are deliberately not cited: the multi-signal ranking network has
a single scalar output head and one training target, and is not a multi-task or Pareto model.
