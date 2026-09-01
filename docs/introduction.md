# 1 Introduction

Automated classification of medical images is a central task in clinical machine learning, with
broad applications across diverse settings such as disease screening, patient triage, and
diagnostic decision support [1]. Unlike image generation, which produces new samples, classification
assigns a categorical label to an image: a capability that is essential for scalable, automated
diagnosis. Its performance, however, is bounded by the availability of large, diverse, and
accurately annotated training data, which is difficult to obtain in practice because
patient-privacy regulation restricts data sharing, expert annotation is costly, and many
pathologies are rare, so the labelled data any single institution can accumulate is limited [2].
Synthetic image generation has emerged as a leading response to this bottleneck, and diffusion
models in particular have displaced generative adversarial networks as the generative backbone of
choice because of their more stable training and their better preservation of fine anatomical
detail. Recent diffusion models produce chest radiographs and other medical images that are
frequently indistinguishable from real ones in blinded physician review, and can improve downstream
classifiers in data-scarce settings [1]. This has made synthetic-to-real learning, in which a model
is trained or adapted on generated images whose labels are known by construction and then deployed
on real, unseen clinical data, a practical strategy for addressing both data scarcity and
cross-institution domain shift [3, 4].

The output of a diffusion model, however, is uneven. Some generated images are uninformative or
artefactual, some do not express the pathology they were conditioned on, and some are
near-duplicates of specific real training patients: unconditional latent diffusion models have
been shown to replicate or near-replicate real patient images, detectable at a high
feature-similarity threshold [5]. Adding all generated images to a real training set
indiscriminately can therefore degrade rather than improve generalisation, and commonly used
generation-quality metrics such as the Fréchet Inception Distance do not predict downstream
diagnostic performance [6, 7]. The question of which synthetic images should be admitted to a real
training set, and in what per-class balance, is treated only in pieces by recent work: through
single-criterion soft re-weighting for synthetic chest radiographs [2], training-free CLIP-based
filtering for segmentation [8], or classifier-likelihood filtering for an imbalanced grading task
[9]. No existing method combines several complementary quality signals, learns which individual
images most improve a downstream multi-label classifier, and adapts the admission threshold to
each class.

To address this gap, this paper proposes ASISM (Adaptive Quality-Aware Synthetic Data Selection),
a framework that curates the output of a LoRA-fine-tuned diffusion foundation model before it is
used to train a real classifier. ASISM has three components:

1. A multi-signal quality assessment that scores every synthetic image on six complementary
   signals: k-nearest-neighbour similarity to real references in a self-supervised feature space,
   no-reference and domain-specific image quality, Monte-Carlo-dropout predictive uncertainty,
   Grad-CAM alignment with expected pathology regions, agreement between the generated image and
   its intended labels, and within-class distinctiveness. Each signal must independently pass a
   go/no-go admission gate before it can enter the model.

2. A learned utility estimator. A permutation-invariant set-utility network is trained on measured
   subset-level changes in classification AUROC, and its value is distilled into a small
   multi-signal ranking network that assigns one scalar utility score to each image [10]. The
   distillation target is a size-normalised leave-one-out marginal contribution; a Data-Banzhaf
   Maximum-Sample-Reuse estimate, computed from the same measured subsets at no additional cost, is
   provided as a noise-robust alternative.

3. An adaptive per-class admission threshold. A learned threshold network predicts a separate
   admission threshold for each disease label, decided once before final classifier training,
   under a three-tier governance rule that uses the learned threshold only where sufficient
   proxy-verified, image-disjoint evidence exists and otherwise falls back to a verified proxy
   threshold or a fixed baseline [11].

All tuning decisions in ASISM are made on a dedicated proxy split, and the final evaluation split
is never accessed during selection. The framework is evaluated end-to-end on a held-out patient
set by comparing a classifier trained on real data only, a classifier trained on real data plus
all synthetic images, and a classifier trained on real data plus the ASISM-selected subset,
together with a matched-random control that draws the same number of synthetic images per class at
random. This design isolates the benefit of intelligent selection from the effect of simply using
fewer synthetic images, an effect that recent work shows can itself be substantial [12, 13].

The main contributions of this paper are summarized as follows:

i. A multi-signal quality assessment for diffusion-generated medical images that combines six
   heterogeneous perceptual signals: semantic similarity, image quality, predictive uncertainty,
   explanation localisation, intended-label agreement, and within-class distinctiveness, under an
   explicit signal-governance gate. To the best of our knowledge, this is the first framework to
   combine this many complementary signals for the selection of synthetic medical images, and the
   first to use explanation–region alignment as a data-selection signal.

ii. A learned set-level utility model, trained on measured downstream AUROC changes and distilled
   into a per-image ranking network, which avoids the pseudo-replication of copying a set-level
   score onto its member images. A Data-Banzhaf Maximum-Sample-Reuse target is derived from the
   same measured subsets with no additional proxy training, as a noise-robust alternative to the
   leave-one-out marginal.

iii. A learned class-specific admission threshold governed by a conservative three-tier rule,
   together with a strictly proxy-based tuning protocol in which selection decisions never touch
   the final evaluation data. The framework is evaluated against a matched-random control that
   separates the contribution of intelligent selection from that of dataset size, an ablation
   absent from comparable prior work.

The remainder of this paper is organized as follows. Section 2 reviews recent work on
diffusion-based synthetic medical image generation and on post-generation synthetic-data
selection. Section 3 details the proposed ASISM framework, including the six signals, the
set-utility and ranking networks, and the adaptive-threshold governance. Section 4 describes the
dataset, the patient-disjoint split design, and the training and proxy-evaluation configuration.
Section 5 presents the experimental results and the ablation studies. Section 6 concludes the
paper, summarizing the key findings, discussing their implications for synthetic-data curation in
medical imaging, and outlining future research directions. A summary of the proposed framework is
presented in Fig. 1.
