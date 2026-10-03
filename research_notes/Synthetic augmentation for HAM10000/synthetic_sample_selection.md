# Selecting / Filtering Synthetic Training Samples (positioning ASISM)

Provenance key: [V] = content checked in this session (abstract/page/search snippet fetched 2026-10-02). [K] = from prior knowledge, citation believed correct but details NOT re-verified this session; check before citing in the thesis.

## Q1. Which signals are used to filter synthetic samples?

### Takeaway
The dominant signals are (a) classifier agreement/confidence with the intended label (a classifier trained on real data, CLIP zero-shot, or an ensemble), and (b) feature-space proximity to real data (class centroids, nearest real clusters, cosine similarity). Newer work (2025-2026) moves toward distribution-level criteria (covariance matching, coverage) and training-dynamics signals (gradient alignment with real data, "not learned early" examples). Sample-level fidelity/diversity metrics (alpha-precision/beta-recall, density/coverage, Vendi) exist but are mostly used for evaluation, rarely as the filter itself.

### Cited Findings
**Classifier confidence / label agreement**
- He et al., "Is Synthetic Data from Generative Models Ready for Image Recognition?", ICLR 2023: uses CLIP-based filtering of generated images (semantic similarity to class text/real examples), mainly in zero-/few-shot settings, alongside language-enhanced prompts — [arXiv 2210.07574](https://arxiv.org/abs/2210.07574); signal summary as described by [FROST related work](https://arxiv.org/html/2609.29988v1). [V: existence and CLIP-feature signal; exact threshold/numbers NOT verified, OpenReview PDF blocked]
- ALIA (Dunlap et al., NeurIPS 2023, "Diversify Your Vision Datasets with Automatic Diffusion-Based Augmentation"): two filters: a CLIP semantic filter removes obvious edit failures; a confidence-based filter removes edits that a classifier trained on the original data predicts confidently (i.e., edits that add little new information) — [arXiv 2305.16289](https://ar5iv.labs.arxiv.org/html/2305.16289); [project page](https://lisadunlap.github.io/alia-website/). [V] Note the direction: ALIA drops *high-confidence* samples, the opposite of "keep confident samples" filters.
- Lin et al., "Explore the Power of Synthetic Data on Few-shot Object Detection", CVPRW 2023: CLIP-based filtering of generated instances (false positives) — [arXiv 2303.13221](https://arxiv.org/pdf/2303.13221). [V existence; filter details K]
- Akrout et al., "Diffusion-based Data Augmentation for Skin Disease Classification" (MICCAI workshop / Springer LNCS 2023): 30,000 generated images per class; a binary EfficientNet removes non-skin images (accepts >99%); an ensemble (EfficientNetV2 + RegNet + Swin) pretrained on the real dataset keeps only synthetic images it classifies as the intended label — [arXiv 2301.04802](https://arxiv.org/abs/2301.04802); [Springer](https://link.springer.com/chapter/10.1007/978-3-031-53767-7_10). [V]

**Feature-space distance to real data**
- Xue et al., "Synthetic Augmentation and Feature-Based Filtering for Improved Cervical Histopathology Image Classification", MICCAI 2019 — [Springer](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_43). [V existence]
- Xue et al., "Selective Synthetic Augmentation with HistoGAN for Improved Histopathology Image Classification", Medical Image Analysis 2021: two-step selection: (1) keep samples confidently classified (entropy-based label-confidence), (2) compute ResNet18 features, class centroids = mean of real training features, sort remaining synthetic samples by distance to their class centroid and keep the closer half. Reported +6.7% (cervical) and +2.8% (metastatic cancer) accuracy — [arXiv 2111.06399](https://arxiv.org/pdf/2111.06399). [V] This is the closest medical precedent to a multi-signal (confidence + centroid distance) selector.
- CosSIF (Islam, Zunair, Mohammed), Computers in Biology and Medicine vol. 172, 2024: cosine-similarity filtering; FBGT removes real images with cross-class similarity before GAN training, FAGT removes synthetic images lacking discriminative power after training. Evaluated on ISIC-2016 and HAM10000; FAGT on ISIC-2016 +1.59% sensitivity, +1.88% AUC; HAM10000 max accuracy 94.44%, recall +13.75% — [arXiv 2307.13842](https://arxiv.org/abs/2307.13842). [V] Directly on HAM10000: must-cite.
- DS3 (2025): samples from synthetic feature clusters nearest to real examples; CovMatch (2026): greedily matches the selected subset's feature covariance to real data — as described in [FROST related work](https://arxiv.org/html/2609.29988v1) and [search snippet](https://arxiv.org/pdf/2609.29988). [V second-hand only; original DS3/CovMatch papers not fetched]
- Training-free synthetic data selection for semantic segmentation: scores images by similarity and a Pseudo-Class Similarity (PCS) score, keeps high-fidelity ones — [arXiv 2501.15201](https://arxiv.org/pdf/2501.15201). [V snippet]

**Distribution-level / theoretical**
- Rezaei, Kovacevic, Locatello, Mondelli, "High-dimensional Analysis of Synthetic Data Selection" (arXiv Oct 2025): in high-dimensional linear analysis, covariance shift between synthetic and target distributions affects generalization error but mean shift surprisingly does not; covariance matching outperforms competing selection approaches across architectures, datasets and generators — [arXiv 2510.08123](https://arxiv.org/abs/2510.08123). [V]

**Training-dynamics / utility signals**
- FROST (Wu et al., arXiv Sep 2026), "Let Training Guide Selection: Online Synthetic Data Filtering via Real-Anchored Utility": per-sample utility = alignment of the synthetic sample's gradient with an EMA of real-data gradients; batch-level z-score routing; IQR thresholds inside out-of-band batches. CIFAR-100 +0.58 to 1.14% while filtering 20-30% of synthetic data (FLUX.1, SANA, SD v1.4) — [arXiv 2609.29988](https://arxiv.org/html/2609.29988v1). [V]
- TADA (Nguyen, Li, Zheng, Mirzasoleiman), "Do We Need All the Synthetic Data? Targeted Image Augmentation via Diffusion Models" (arXiv May 2025, rev. Mar 2026): augment only examples not learned early in training (30-40% of data), up to +2.8% on CIFAR-10/100, TinyImageNet, ImageNet; with SGD beats SAM on some benchmarks — [arXiv 2505.21574](https://arxiv.org/abs/2505.21574). [V] Selection is on which *real* examples to synthesize for, a "which" decision driven by learning difficulty.
- Data-pruning signals borrowed by synthetic-selection work: GraNd/EL2N (Paul et al., NeurIPS 2021), GRAD-MATCH (2021), LESS (2024), GREATS (2024) — listed in [FROST related work](https://arxiv.org/html/2609.29988v1). [V second-hand]

**Sample-level fidelity/diversity metrics (candidates as filters)** [K]
- Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models" (density & coverage), ICML 2020 — [arXiv 2002.09797](https://arxiv.org/abs/2002.09797).
- Alaa et al., "How Faithful is your Synthetic Data? Sample-level Metrics for Evaluating and Auditing Generative Models" (alpha-precision, beta-recall, authenticity), ICML 2022 — [arXiv 2102.08921](https://arxiv.org/abs/2102.08921). Explicitly proposes sample-level auditing, i.e., discarding low-quality samples post hoc.
- Friedman & Dieng, "The Vendi Score: A Diversity Evaluation Metric for Machine Learning", TMLR 2023 — [arXiv 2210.02410](https://arxiv.org/abs/2210.02410).
- Data valuation: Ghorbani & Zou, "Data Shapley", ICML 2019 — [arXiv 1904.02868](https://arxiv.org/abs/1904.02868); Pruthi et al., "TracIn", NeurIPS 2020 — [arXiv 2002.08484](https://arxiv.org/abs/2002.08484).

**Medical/other domain snippets**
- Brain MRI (BRISC 2025, StyleGAN2-ADA): synthetic pool 2.5x real size, quality control then deterministic farthest-point sampling to pick a diverse subset and avoid collapse onto high-density look-alikes — [arXiv 2605.23094](https://arxiv.org/pdf/2605.23094). [V snippet]
- Another work combines prototype filtering, classifier confidence and Mahalanobis distance to choose synthetic samples (cited via search summary; exact paper not identified) — [search result context, arXiv 2605.10916](https://arxiv.org/pdf/2605.10916). [uncertain attribution]

### Inferences
- Nearly all existing filters are *fixed-threshold, single- or two-signal, label-consistency* filters. ASISM's combination of classifier agreement + DINOv2/CLIP distance + fidelity + diversity + novelty, with an adaptive decision of how many to keep, sits between HistoGAN-style two-step filtering and distribution-level selectors (covariance matching, HO/HE splitting).
- A DINOv2/CLIP judge separate from the downstream classifier is a design choice not commonly made explicit; most medical filters reuse the downstream classifier (or a sibling) as judge, which risks confirmation bias.

### Gaps
- Exact CLIP-filter thresholds and ablation numbers in He et al. (OpenReview blocked).
- Original DS3 and CovMatch papers not located; only second-hand descriptions.
- Did not find a paper that uses Naeem density/coverage or Vendi *as the per-sample filter* for classifier augmentation; may exist but not found.
- No found paper that applies Data Shapley/TracIn specifically to select synthetic images for image classification (gradient-alignment in FROST is the nearest).

## Q2. Which papers combine multiple signals or learn a selection policy (and how do the listed landmark papers handle selection)?

### Takeaway
Most landmark general-CV synthetic-data papers (Sariyildiz, Azizi, Real-Fake, Diversify-Don't-Fine-Tune, Training on Thin Air, DataDream) improve the *generator or prompts* rather than filter samples; explicit multi-signal selection is found mainly in ALIA (CLIP + classifier confidence), HistoGAN (entropy + centroid distance), Akrout (domain classifier + ensemble agreement), HO/HE curation (fidelity + anti-redundancy), and learned/online selectors (FROST, SunGen, active generation).

### Cited Findings
- Diversify, Don't Fine-Tune (Yu, Zhu, Culatana, Krishnamoorthi, Xiao, Lee; arXiv 2312.02253): uses off-the-shelf generators with LLM-driven contextual + style diversification; gains keep growing up to 6x ImageNet size of synthetic data, unlike fine-tuned-generator approaches where performance declined once synthetic outnumbered real — [arXiv 2312.02253](https://arxiv.org/abs/2312.02253). [V] Focus is diversity at generation time, not post-hoc filtering (any filtering step not verified).
- Post-Generation Curation via Homogeneous-Heterogeneous Splitting (Liu, Liang, Song, Yin; arXiv Jul 2026): splits each real class into canonical (HO) and non-redundant (HE) subsets; scores synthetic images with a fidelity-diversity criterion that rewards semantic alignment and penalizes canonical redundancy; generator-agnostic; matches real-data performance with up to 40% fewer synthetic samples — [arXiv 2607.02637](https://arxiv.org/abs/2607.02637). [V] Closest conceptual neighbour to ASISM's fidelity + diversity + novelty combination.
- FROST: learned/online utility-based selection with batch routing (see Q1) — [arXiv 2609.29988](https://arxiv.org/html/2609.29988v1). [V]
- Diffusion Curriculum (Liang et al., ICCV 2025): generates synthetic-to-real interpolations at different image-guidance levels and selects guidance levels over training as a curriculum — [CVF](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Liang_Diffusion_Curriculum_Synthetic-to-Real_Data_Curriculum_via_Image-Guided_Diffusion_ICCV_2025_paper.pdf). [V snippet]
- Fan et al., "Scaling Laws of Synthetic Images for Model Training ... for Now", CVPR 2024 [K] — [arXiv 2312.04567](https://arxiv.org/abs/2312.04567).
- Landmark papers, mostly no sample filtering [K, not re-verified]:
  - Sariyildiz et al., "Fake it till you make it", CVPR 2023 — prompt design and guidance scale for ImageNet clones; [arXiv 2212.08420](https://arxiv.org/abs/2212.08420).
  - Azizi et al., "Synthetic Data from Diffusion Models Improves ImageNet Classification", TMLR 2023 — fine-tuned Imagen; gains come from generator tuning (resolution, guidance); [arXiv 2304.08466](https://arxiv.org/abs/2304.08466). Performance degrades as synthetic outnumbers real (reported by [Yu et al.](https://arxiv.org/abs/2312.02253) [V]).
  - Yuan et al., "Real-Fake: Effective Training Data Synthesis Through Distribution Matching", ICLR 2024 — distribution-matching view of generator training rather than sample filtering; [arXiv 2310.10402](https://arxiv.org/abs/2310.10402).
  - Zhou, Sahak, Ba, "Training on Thin Air: Improve Image Classification with Generated Data" (Diffusion Inversion), 2023 — [arXiv 2305.15316](https://arxiv.org/abs/2305.15316).
  - Kim et al., "DataDream: Few-shot Guided Dataset Generation", ECCV 2024 — LoRA-tunes the generator on few-shot real data — [arXiv 2407.10910](https://arxiv.org/html/2407.10910v2).
  - Trabucco et al., "Effective Data Augmentation With Diffusion Models" (DA-Fusion), ICLR 2024 — [arXiv 2302.07944](https://arxiv.org/abs/2302.07944).
- Learned weighting/policy precedents [K]: SunGen, Gao et al., "Self-Guided Noise-Free Data Generation for Efficient Zero-Shot Learning", ICLR 2023 — bilevel learned per-sample weights on synthetic (text) data, [arXiv 2205.12679](https://arxiv.org/abs/2205.12679); Askari-Hemmat et al., "Feedback-guided Data Synthesis with Diffusion Models" (classifier loss/entropy as generation guidance), [arXiv 2310.00158](https://arxiv.org/abs/2310.00158) (venue uncertain); Huang et al., "Active Generation for Image Classification" (ActGen), ECCV 2024, [arXiv 2403.06517](https://arxiv.org/abs/2403.06517).
- "Not all synthetic data are equal": no paper with that exact title was found for image classification; search returned only blogs and related benchmarks — [search context](https://arxiv.org/html/2406.05184v3).

### Inferences
- ASISM's novelty claim is defensible as: (i) >2 signals combined at sample level, (ii) a judge in a foundation-model space (DINOv2) independent of the downstream classifier, (iii) deciding *how many* adaptively. Closest priors to contrast: HistoGAN (2 signals, fixed keep-half), HO/HE curation (fidelity + anti-redundancy, general CV), FROST (online, training-signal), covariance matching (distribution-level).
- Many positive general-CV results come from generator/prompt diversity, which argues that selection should not undo diversity (see Q3).

### Gaps
- Did not verify whether DataDream, Real-Fake, or Diversify-Don't-Fine-Tune include any CLIP filtering step; check their method sections.
- Could not identify the exact paper referenced as "Not all synthetic data are equal".

## Q3. What is known about filtering being harmful?

### Takeaway
Evidence consistently shows a quality-diversity trade-off: aggressive or confidence-maximizing filters remove useful hard/diverse samples and collapse onto canonical modes; moderate filtering (about 20-40% removed) or filters that explicitly reward diversity/anti-redundancy perform best. Using the downstream classifier as judge invites confirmation bias, and ALIA's choice to drop *confidently* predicted samples is direct evidence that "easy" synthetic samples add little.

### Cited Findings
- FROST: aggressive filtering (narrow symmetric utility bands) removes useful supervision despite noisy synthetic data; highest-utility batches peak early but are unstable; moderate-utility bands are more stable long-term — [arXiv 2609.29988](https://arxiv.org/html/2609.29988v1). [V]
- Filtering rates that are too high reduce synthetic diversity, too low leave harmful low-quality images (trade-off stated in a work surfaced by search; attribution to a specific paper uncertain) — [search context incl. He et al. OpenReview](https://openreview.net/pdf?id=nUmCcZ5RKF). [uncertain]
- Generators over-produce canonical class modes and underrepresent intra-class variation; curation should penalize canonical redundancy — [arXiv 2607.02637](https://arxiv.org/abs/2607.02637). [V]
- Brain MRI study adds farthest-point sampling specifically to stop the filtered set collapsing onto many similar high-density samples — [arXiv 2605.23094](https://arxiv.org/pdf/2605.23094). [V snippet]
- ALIA removes edits the real-trained classifier predicts confidently, treating them as uninformative — [ALIA](https://lisadunlap.github.io/alia-website/). [V]
- TADA: the useful synthetic data are those for hard (not-early-learned) examples — [arXiv 2505.21574](https://arxiv.org/abs/2505.21574). [V]
- Mean-matching (keep samples near real data) is not what matters in the high-dimensional analysis; covariance matching is — [arXiv 2510.08123](https://arxiv.org/abs/2510.08123). [V] Implies centroid-proximity filters can be suboptimal by shrinking spread.
- Retrieved real images match or beat synthetic: 15K retrieved aircraft images need ~500K synthetic to match — Geng et al., "The Unmet Promise of Synthetic Training Images" — [arXiv 2406.05184](https://arxiv.org/html/2406.05184v3). [V snippet]
- Chest X-ray: ~42% of synthetic images incorrectly indicated COVID (hallucination), motivating validity checks — [arXiv 2312.06979](https://arxiv.org/html/2312.06979). [V snippet]

### Inferences
- For ASISM on HAM10000: a pure "classifier agrees" filter will favour canonical-looking minority samples and can reinforce the DenseNet nv-sink; the project's own diagnostics (synthetic mel/bkl read as nv) are an instance of judge-classifier bias. Literature supports including diversity/novelty terms and an independent judge.
- Keep-rates in the literature that worked: ~50% (HistoGAN), 60-80% (FROST), and selecting a 30-40% target subset (TADA); none supports tuning keep-rate on test results.

### Gaps
- No controlled medical-imaging study found that quantifies diversity loss caused by confidence filtering (beyond the MRI FPS remark).
- Confirmation-bias literature from self-training/pseudo-labeling (e.g., Arazo et al. 2020) not fetched; relevant analogue but unverified here.

## Q4. Medical-imaging-specific selection methods

### Takeaway
Medical selection methods are mostly classifier-agreement filters (Akrout dermatology ensemble), feature-similarity filters (CosSIF on HAM10000/ISIC; HistoGAN centroid distance in histopathology), and occasional diversity sampling (MRI farthest-point). Chest X-ray work emphasizes validity/hallucination checks more than formal selection.

### Cited Findings
- Dermatology: CosSIF on ISIC-2016 and HAM10000 — [arXiv 2307.13842](https://arxiv.org/abs/2307.13842) [V]; Akrout et al. ensemble filter — [arXiv 2301.04802](https://arxiv.org/abs/2301.04802) [V]. Other dermatology generation work (no verified selection details): SkinGenBench (generative model and preprocessing effects on melanoma augmentation) — [arXiv 2512.17585](https://arxiv.org/pdf/2512.17585); "From Majority to Minority" diffusion augmentation for underrepresented groups — [arXiv 2406.18375](https://arxiv.org/pdf/2406.18375); DermaFlux (rectified flows) — [arXiv 2603.16392](https://arxiv.org/pdf/2603.16392).
- Histopathology: Xue et al. 2019 (MICCAI) and HistoGAN 2021 (MedIA) — [arXiv 2111.06399](https://arxiv.org/pdf/2111.06399) [V]; Ye et al., "Synthetic Augmentation with Large-scale Unconditional Pre-training" (HistoDiffAug), MICCAI 2023 — [arXiv 2308.04020](https://arxiv.org/pdf/2308.04020) [V existence; selection component not verified].
- Brain MRI: quality control + farthest-point sampling — [arXiv 2605.23094](https://arxiv.org/pdf/2605.23094) [V snippet].
- Chest X-ray: hallucination/validity of synthetic CXR — [arXiv 2312.06979](https://arxiv.org/html/2312.06979); diffusion classifiers give uncertainty, and filtering uncertain predictions improves remaining accuracy — [arXiv 2502.03687](https://arxiv.org/html/2502.03687); concept-coverage-driven generation for CXR models — [arXiv 2603.15525](https://arxiv.org/pdf/2603.15525); diffusion-synthesized CXR improving fairness — [PLOS Digital Health](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001277). [V snippets; none confirmed as a per-sample synthetic selection method]

### Inferences
- No found medical method combines 5 signals or adaptively sets the count; ASISM's closest medical comparators are HistoGAN (2 signals) and CosSIF (similarity filter, same dataset). CosSIF should be a baseline or at least discussed since it reports HAM10000 results.

### Gaps
- Not verified whether Ktena et al. (Nature Medicine 2024) or Sagers et al. (dermatology LDM augmentation) filter samples.
- No chest-X-ray paper found that does explicit per-sample selection of synthetic training images before classifier training.
