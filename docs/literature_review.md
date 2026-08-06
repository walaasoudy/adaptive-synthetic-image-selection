# Literature Review

The scarcity of large, diverse, and accurately annotated datasets remains one of the central
obstacles to deploying deep learning in medical image analysis. Patient privacy regulation, the
cost of expert annotation, and the rarity of many pathologies together limit the amount of
labelled data any single institution can accumulate. Synthetic data generation has emerged as a
leading strategy for addressing this bottleneck, and diffusion models in particular have rapidly
displaced Generative Adversarial Networks (GANs) as the generative backbone of choice, owing to
their more stable training and their ability to preserve fine anatomical detail. A closely related
obstacle is domain shift: models trained at one institution routinely degrade when deployed at
another, owing to differences in scanner vendor, acquisition protocol, and patient population.
This has motivated a growing body of work on synthetic-to-real (S2R) learning, in which
diffusion-generated images, whose labels are known by construction or transferred from a source
domain, are used to train or adapt models intended for deployment on real, unseen clinical data.

Guan and Liu surveyed the broader field of domain adaptation for medical image analysis and
taxonomized existing approaches into shallow feature-alignment methods and deep, adversarial
(GAN-based) methods such as CycleGAN, noting that multi-vendor scanner variation (e.g., Philips
versus Siemens) remains the primary driver of domain shift in clinical practice [1]. Ji and Chung
proposed a diffusion-based unsupervised domain adaptation (UDA) framework for abdominal
multi-organ segmentation, introducing Stochastic Step Alignment (SSA), which aligns the
intermediate trajectory of the diffusion denoising process between source and target domains
rather than only the final generated image. The proposed method achieved DSC values of 86.5% for
MR→CT and 88.3% for CT→MR on the CHAOS benchmark, outperforming previous domain adaptation
approaches [2]. Gong et al. extended this line of work to 3D volumetric segmentation with
Diffuse-UDA, which separately aligns appearance, using frequency-domain amplitude swapping, and
structure, using a deformable augmentation module, reporting a DSC of 84.1% on CT-to-MR cardiac
segmentation, close to the fully supervised upper bound of 84.5% [3].

Cho et al. introduced MediSyn, an open-access, text-guided latent diffusion model trained on 1.26
million public image-text pairs spanning ten imaging modalities and six medical specialties, and
showed through blinded physician review that the model's outputs were frequently indistinguishable
from real images while improving downstream classifier performance in data-scarce settings [4].
Zeng et al. addressed the stricter, privacy-driven scenario in which source-domain pixels cannot
be shared at all, proposing Reliable Source Approximation (RSA), a source-free UDA method that
uses edge-conditioned diffusion to synthesize source-like approximations of target-domain images
and filters the resulting pseudo-labels using a reliability criterion before fine-tuning; the
method achieved 77.83% Dice on vestibular schwannoma MRI segmentation without ever accessing
source pixels, substantially outperforming the prior source-free baseline (Fourier Style Mining,
56.12%) [5]. Zhang et al. tackled the related problem of institutional stain shift in digital
pathology, conditioning a latent diffusion model (PathLDM) on foundation-model (UNI) feature
embeddings to preserve tissue morphology while allowing staining appearance to drift toward the
target institution, improving macro F1 score from 0.641 to 0.716 on a cross-cohort lung
adenocarcinoma classification task [6].

Saragih et al. proposed a fully automated pipeline for polyp segmentation that clusters training
images by similarity before training one diffusion model per cluster, using a four-channel (image
plus mask) input to generate spatially aligned synthetic image-label pairs without any manual
annotation; the method achieved an FID of 65.99, compared with 118.73 for a GAN-based baseline,
and improved Dice scores in low-data settings (N=16) from 0.39 to 0.58 [7]. Khader et al. presented
one of the first large-scale evaluations of diffusion models for 3D medical volume generation,
combining VQ-GAN latent compression with a 3D DDPM to synthesize MRI and CT volumes; radiologist
review rated the majority of generated volumes as realistic and anatomically correct, and
pretraining on synthetic volumes improved breast segmentation Dice from 0.91 to 0.95 in low-data
regimes, though at the cost of approximately seven days of training per model [8]. Domínguez et al.
argued that the standard Gaussian noise schedule used in most diffusion models is physically
unrealistic for ultrasound imaging and proposed a noise schedule derived from acoustic wave-decay
physics (B-maps), reducing the FID of synthetic liver ultrasound images from 20.75 to 0.19 relative
to a standard diffusion baseline [9].

Beyond generation quality, several studies have examined whether synthetic images preserve the
diagnostic content required for clinical use. Pozzi et al. introduced a three-tier evaluation
pipeline for synthetic digital pathology tiles combining statistical fidelity metrics, Concept
Relevance Propagation (CRP) to verify that classifiers rely on correct morphological features, and
blinded assessment by professional pathologists; their explainability analysis revealed a case of
cross-organ concept entanglement, in which synthetic pancreas tiles activated features associated
with lung tissue, illustrating that visual realism does not guarantee correct diagnostic grounding
[10]. Hosseini and Serag investigated whether diffusion models preserve concealed clinical
biomarkers, such as lung markings or retinal drusen, by training classifiers exclusively on
synthetic data and testing on real, unseen patients across radiology, ophthalmology, and
histopathology, reporting AUC and F1 scores of 0.8–0.99, providing direct empirical evidence that
synthetic-only training can generalize to real clinical populations [11]. Fei et al. proposed
UniMIE, a training-free medical image enhancement framework that treats enhancement as an
inversion problem solved using a diffusion model pretrained only on natural (ImageNet) images,
demonstrating robust zero-shot performance across thirteen imaging modalities without any
modality-specific fine-tuning [12].

Rehman et al. addressed demographic bias in diagnostic AI, using Low-Rank Adaptation (LoRA) to
fine-tune Stable Diffusion on tabular-to-text prompts describing patient age, sex, and race, in
order to generate demographically balanced Chest X-ray training sets; Grad-CAM analysis confirmed
that classifiers trained on the resulting mixed real-and-synthetic data shifted attention away
from a demographic shortcut (the shoulder region) toward the true disease region (the
cardiomediastinum), while reducing GPU memory requirements from 24GB to 8GB relative to full
fine-tuning [13]. Azad et al. [16] conducted a systematic review of 68 diffusion-based medical
diagnosis studies, analyzing methods, clinical integration, explainability, and future directions.
Their review found that only 10.30% of the studies addressed real-time clinical deployment and
only 22.06% incorporated explainable AI (XAI) components, highlighting hybrid
diffusion–foundation-model architectures and physics-informed generation as important future
research directions.

Niemeijer et al. [15] introduced TSynD, a targeted synthetic data generation framework that shifts
augmentation from random sample generation toward uncertainty-guided synthesis. By optimizing
synthetic samples according to classifier epistemic uncertainty, TSynD generated informative
examples that improved classification performance under limited data conditions. However, the
framework focuses on selecting informative synthetic samples rather than optimizing their
integration with real samples, leaving the synthetic-real balancing problem unresolved.

Taken together, these studies establish diffusion models as a powerful generative approach for
medical image synthesis and domain adaptation. They demonstrate that structural conditioning,
including edges, segmentation masks, foundation-model features, and physics-informed constraints,
is essential for reducing anatomical inconsistencies and improving the reliability of generated
medical images. However, despite these advances, the optimal utilization of synthetic data after
generation remains an open research challenge. Most existing studies focus on improving
generation quality or reducing domain discrepancy, while the strategy for selecting, weighting,
and integrating synthetic samples with real clinical data is rarely investigated. Current
approaches that combine synthetic and real data typically rely on manually selected fixed ratios
[7, 8], without considering whether the selected proportion is optimal for the target task.
Although Niemeijer et al. [15] introduced uncertainty-guided synthetic data generation to identify
more informative synthetic samples, the method does not address the subsequent synthetic-real
data balancing problem. Furthermore, commonly used image quality metrics such as FID do not
necessarily reflect the impact of synthetic samples on downstream diagnostic performance
[10, 11]. Therefore, developing adaptive strategies for optimizing synthetic-real data composition
represents an important research direction for improving the effectiveness of diffusion-based
synthetic data augmentation in medical image analysis.

## Table 1. Summary of reviewed diffusion-based medical imaging studies and their limitations relative to synthetic-to-real (S2R) learning

| Study | Method | Dataset/Modality | Key Contribution | Limitation Relative to S2R |
|---|---|---|---|---|
| Guan and Liu [1] | Domain Adaptation Survey | Multi-modality medical imaging | Taxonomy of domain adaptation methods; identifies scanner/vendor shift as a major driver of domain discrepancy | Does not focus on diffusion-based synthetic data integration |
| Ji and Chung [2] | Stochastic Step Alignment (DDPM-based UDA) | CHAOS (CT/MRI) | Aligns intermediate diffusion trajectories between source and target domains; 86.5% DSC (MR→CT), 88.3% DSC (CT→MR) | Focuses on domain alignment rather than synthetic-real data optimization |
| Gong et al. [3] | Diffuse-UDA (Conditional DDPM) | MMWHS, FeTA (3D CT/MRI) | Separates appearance and structural alignment for cross-modality segmentation; 84.1% DSC | Does not optimize synthetic data utilization strategy |
| Cho et al. [4] | MediSyn (Text-guided Latent Diffusion Model) | 10 imaging modalities, 6 medical specialties | Large-scale medical image generation (1.26M image-text pairs) with physician evaluation | Focuses on generation quality rather than downstream synthetic-real balancing |
| Zeng et al. [5] | Reliable Source Approximation (Edge-conditioned DDPM) | Vestibular Schwannoma MRI | Source-free domain adaptation via diffusion-generated source approximation; 77.83% Dice | Synthetic data selection and mixing strategy remain unexplored |
| Zhang et al. [6] | PathLDM (Foundation-model conditioned LDM) | NLST/TCGA Histopathology | Preserves morphology while adapting stain appearance; Macro-F1 improved 0.641→0.716 | Limited evaluation of synthetic-real composition |
| Saragih et al. [7] | Cluster-based 4-channel DDPM | HyperKvasir, CVC-ClinicDB, PolypGen | Generates aligned image-mask pairs; Dice improved 0.39→0.58 in low-data settings | Uses predefined augmentation strategy without optimized mixing ratio |
| Khader et al. [8] | VQ-GAN + 3D DDPM | MRI/CT volumes | Realistic 3D medical volume synthesis; segmentation Dice improved 0.91→0.95 | Uses synthetic pretraining without studying optimal synthetic-real ratio |
| Domínguez et al. [9] | Physics-guided DDPM Noise Schedule | Ultrasound imaging | Improved ultrasound synthesis quality; FID reduced 20.75→0.19 | Focuses on generation process rather than data utilization |
| Pozzi et al. [10] | DDPM + CRP Explainability Evaluation | Histopathology | Shows visual realism does not guarantee correct diagnostic features | Synthetic quality metrics may not represent clinical usefulness |
| Hosseini and Serag [11] | DDPM + Swin Transformer | X-ray, OCT, Histopathology | Shows synthetic-only training can generalize to real clinical data; AUC/F1 of 0.8–0.99 | Does not investigate optimal real-synthetic integration |
| Fei et al. [12] | UniMIE Diffusion Inversion | 13 imaging modalities | Training-free medical image enhancement using diffusion priors | Not designed for synthetic data augmentation |
| Rehman et al. [13] | LoRA-tuned Stable Diffusion | CheXpert, MIMIC-CXR | Generated demographically balanced synthetic chest X-rays; reduced shortcut learning | Does not optimize synthetic-real data proportion |
| Niemeijer et al. [15] | TSynD Uncertainty-guided Synthesis | PathMNIST (MedMNIST) | Generated informative synthetic samples via classifier uncertainty; accuracy improved 67.4%→73.1% | Does not address optimal synthetic-real sample balancing |
| Azad et al. [16] | Systematic Review | Multi-modality | Comprehensive taxonomy of diffusion-based medical diagnosis methods; identifies gaps in clinical deployment, explainability, future research | Does not investigate synthetic-real data composition or optimization strategies |

## The Gap This Thesis Addresses

Across all of the above, the strategy for **selecting, weighting, and adaptively integrating**
synthetic samples with real clinical data after generation is largely unexplored. Existing
combined-data approaches rely on fixed, manually chosen ratios [7, 8]. The closest prior work —
Rehman et al. [13] (LoRA-tuned SD on CheXpert/MIMIC-CXR, with Grad-CAM verification) and
Niemeijer et al. [15] (TSynD, uncertainty-guided sample generation) — each address one piece of
the problem but neither develops an adaptive, multi-signal selection and mixing strategy. This
thesis's ASISM module (Stage 3) is designed to fill that gap.