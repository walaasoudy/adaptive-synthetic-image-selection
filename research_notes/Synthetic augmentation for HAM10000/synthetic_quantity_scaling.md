# How many synthetic images to add: quantity/scaling curves and real:synthetic ratios

Scope note: compiled 2026-10-02 for the thesis "E4 quantity curve" (varying synthetic images per class added to HAM10000; asking whether an adaptive method should choose a per-class count). Items marked [verified] were read from the paper/abstract page this session. Items marked [not re-fetched] are from prior knowledge of well-known papers. Cite them only after checking the PDF. Numbers pulled from a page summary (not the PDF table directly) are flagged.

## Q1. What scaling behaviour is reported for synthetic data in general vision? Does performance saturate or decline?

### Takeaway
When synthetic images are added to a real training set of fixed size, accuracy goes up at first, peaks at a small multiple of the real set, and then falls. Azizi et al. report a ResNet-50 ImageNet peak at about 1x synthetic (+1.2M images), with gains shrinking at 2x-3x and turning negative beyond that. When models are trained on synthetic data alone, accuracy follows a power law that is weaker than real data's and plateaus at a few million images. Much of the gap comes from a subset of "poor" classes the generator cannot render.

### Cited Findings
- **Azizi, Kornblith, Saharia, Norouzi, Fleet (2023), "Synthetic Data from Diffusion Models Improves ImageNet Classification", TMLR; arXiv:2304.08466.** They augmented ImageNet with 1x to 10x synthetic images (1.2M to 12M added). ResNet-50 results by total training set: 1.2M real only = 76.39%; +1.2M synthetic (2.4M total) = 78.12% (+1.73); 3.6M total = 77.48%; 4.8M total = 76.75%; larger multiples reportedly fall below the real-only baseline. The authors attribute the decline to bias in the generative model. With +1.2M synthetic, ViTs gain less than CNNs (ViT-S/16 +1.11, DeiT-B +1.04, DeiT-L +0.83). [verified via arXiv HTML summary. The per-row numbers came from a page-summary tool, so check them against Table 4 before quoting. The claim that ViT also declines at high multiples is not confirmed.] — [arXiv:2304.08466](https://arxiv.org/abs/2304.08466) / [HTML](https://arxiv.org/html/2304.08466)
- The same paper sets a synthetic-only Classification Accuracy Score (ResNet-50 trained only on synthetic, tested on real ImageNet) of 64.96 top-1 at 256x256 and 69.24 at 1024x1024, against about 76 for real data. [verified] — [arXiv:2304.08466](https://arxiv.org/abs/2304.08466)
- **Fan, Chen, Krishnan, Katabi, Isola, Tian (2024), "Scaling Laws of Synthetic Images for Model Training ... for Now", CVPR 2024; arXiv:2312.04567.**
  - They fit loss ∝ (1/D)^k. For supervised classifiers, synthetic data scales clearly worse than real. For CLIP training the gap is smaller.
  - The power law holds from about 0.125M up to about 4M synthetic images with ViT-B, and up to about 8M with ViT-L, before saturating. A larger model pushes the saturation point out.
  - Per-class analysis splits classes three ways: "easy" (good from the start), "scaling" (poor at first but improve with more data) and "poor" (barely improve, because the text-to-image model cannot generate the concept).
  - Synthetic data helps most when real data is scarce (under about 0.5M images), under distribution shift, and when mixed with real data. Mixing gave up to about +5% in CLIP training at 1M-10M scale.
  - Prompts, CFG scale and choice of generator change the scaling behaviour.
  [verified via arXiv HTML] — [arXiv:2312.04567](https://arxiv.org/abs/2312.04567) / [HTML](https://arxiv.org/html/2312.04567)
- **He, Sun, Yu, Xue, Zhang, Torr, Bai, Qi (2023), "Is Synthetic Data from Generative Models Ready for Image Recognition?", ICLR 2023; arXiv:2210.07574.** They study synthetic data in zero-shot and few-shot settings and for pre-training, and report both strengths and limits. Code: github.com/CVMI-Lab/SyntheticData. [verified: abstract only. From prior knowledge (not re-fetched): zero-shot gains diminish as the amount of synthetic data grows, and the few-shot benefit shrinks as real shots increase. Confirm in Sec. 3 before citing.] — [OpenReview](https://openreview.net/forum?id=nUmCcZ5RKF); [arXiv](https://arxiv.org/abs/2210.07574)
- **Sariyildiz, Alahari, Larlus, Kalantidis (2023), "Fake it till you make it: Learning transferable representations from synthetic ImageNet clones", CVPR 2023; arXiv:2212.08420.** This is training on synthetic data alone, not augmentation. On ImageNet-100 they generated sets 10x, 20x and 50x the real size. At 10x the synthetic-trained model beats the real-trained one, and gains keep rising up to 50x. [verified via search snippet of the paper text. Confirm exact figures in Sec. "Scaling the number of synthetic images".] — [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Sariyildiz_Fake_It_Till_You_Make_It_Learning_Transferable_Representations_From_CVPR_2023_paper.html); [arXiv](https://arxiv.org/abs/2212.08420)
- **Tian et al. (2023), "StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners", NeurIPS 2023; arXiv:2306.00984.** This is self-supervised/contrastive pre-training on synthetic images. With multi-positive contrastive learning on synthetic images, it matches or beats SimCLR/CLIP trained on the same number of real images. [not re-fetched] — [arXiv:2306.00984](https://arxiv.org/abs/2306.00984)
- **Tian et al. (2024), "Learning Vision from Models Rivals Learning Vision from Data" (SynCLR), CVPR 2024; arXiv:2312.17742.** Representations are learned from synthetic images and synthetic captions only, about 600M images. It is a pre-training setting and is not directly comparable to supervised augmentation. [not re-fetched] — [arXiv:2312.17742](https://arxiv.org/abs/2312.17742)
- **Wang, Zhang, Zhang, Chen, Xu, Kwark, Tu (2025), "Exploring the Equivalence of Closed-Set Generative and Real Data Augmentation in Image Classification", arXiv:2508.09550.** The generator is trained on the same training set (closed-set, like a LoRA fit on HAM10000). They estimate how many synthetic images equal one real image. The ratio is not fixed: it changes with the size of the base real set and with how much synthetic data is already added. Real images are generally preferred. Covers natural and medical datasets. [verified: abstract only. Exact ratios not extracted.] — [arXiv:2508.09550](https://arxiv.org/abs/2508.09550)

### Inferences
- For augmentation (real set fixed, synthetic added), the best-documented curve has a peak and then a decline (Azizi), not monotone saturation. Synthetic-only scaling (Fan, Sariyildiz) is a different regime and should not be used to predict augmentation curves.
- Fan's split into easy, scaling and poor classes suggests the useful quantity depends on the class, which is direct prior support for E4's per-class question.
- The 10x-50x gains in synthetic-only settings come from generators that make good images for that domain. HAM10000 LoRA generators, where synthetic mel/bkl are read as nv (see project memory), are closer to Fan's "poor class" case.

### Gaps
- Could not confirm whether Azizi's ViT results also decline at high multiples, or the exact multiple at which ResNet-50 drops below baseline.
- Did not extract He et al.'s quantity-sweep figures.

## Q2. What real:synthetic ratios work best in medical imaging? Any per-class counts or "fill every class to N"?

### Takeaway
Medical results are mixed and depend heavily on setup. In dermatology, Sagers et al. report gains that saturate above about 10:1 synthetic:real in data-limited settings. Class-balancing to the head-class count ("fill to N", N = largest class) is common on ISIC/HAM and helps the smallest classes most, but these papers rarely sweep N. Systematic per-class quantity sweeps with proper statistics are rare.

### Cited Findings
- **Sagers, Diaz, Groh, Rotemberg, Roy, Daneshjou (2023), "Augmenting medical image classifiers with synthetic data from latent diffusion models", arXiv:2308.12453 (dermatology).** 458,920 synthetic images from several generation strategies. Synthetic augmentation helps in data-limited settings, but "performance gains saturate at synthetic-to-real image ratios above 10:1". Adding real images gives much larger gains than adding synthetic ones, and the authors say collecting diverse real data remains the most important step. [verified: abstract] — [arXiv:2308.12453](https://arxiv.org/abs/2308.12453)
- **Akrout et al. (2023), "Diffusion-based Data Augmentation for Skin Disease Classification: Impact Across Original Medical Datasets to Fully Synthetic Images", arXiv:2301.04802.** A classifier trained on a fully synthetic skin-disease dataset keeps similar accuracy to one trained on real data. Fine-grained prompt control is used. [verified: abstract. Ratio-sweep details not extracted.] — [arXiv:2301.04802](https://arxiv.org/abs/2301.04802)
- **Ktena et al. (2024), "Generative models improve fairness of medical classifiers under distribution shifts", Nature Medicine 30:1166-1173; DOI 10.1038/s41591-024-02838-6; arXiv:2304.09218.** Covers histopathology, chest X-ray and dermatology. Adding synthetic samples to real ones improved robustness in all three tasks and improved fairness, mainly out of distribution. In histopathology it beat baselines both with 1,000 labelled samples and with only 100. [verified: abstract/summary. The mixing-ratio ablation could not be parsed from the PDF.] — [Nature Medicine](https://www.nature.com/articles/s41591-024-02838-6); [arXiv](https://arxiv.org/abs/2304.09218)
- **Rehman et al. (2026), "Diffusion-synthesized Chest X-rays improve fairness and diagnostic performance", PLOS Digital Health; DOI 10.1371/journal.pdig.0001277.**
  - Setup: about 84,000 synthetic CheXpert-derived images (about 6,000 per disease, an equal "fill") alongside about 130k real.
  - In a supplementary sweep of synthetic proportion, ResNet-50 AUC rose from 0.911 at 0% to 0.932 at 100%. Demographic AUC gaps shrank (race 0.062 to 0.010). No saturation was observed.
  - Statistics: 250 bootstrap resamples for 95% CIs and paired t-tests. Random seeds are not reported.
  [verified via page summary. The exact meaning of "100% synthetic proportion" should be checked in the supplement.] — [PLOS Digital Health](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001277)
- **Jiang, Subedar, Tickoo (2026), "Synthetic Data Generation for Long-Tail Medical Image Classification: A Case Study in Skin Lesions", arXiv:2605.03221 (ISIC2019, 8 classes).** This is an explicit fill-to-head rule: synthetic count for class j = max(0, |head| - |c_j|), so every class is filled to NV's 12,875. DF (239 real) gets about 12,636 synthetic images. The generator is inpainting diffusion plus OOD filtering, keeping γ = 0.2-0.6 of the samples. Balanced multiclass accuracy goes from 0.757 to 0.802. Per class: DF 0.611 to 0.880, VASC 0.876 to 0.965, AKIEC 0.604 to 0.724. Statistics: 5-fold CV only, with no CIs or significance tests. [verified: arXiv HTML] — [arXiv:2605.03221](https://arxiv.org/html/2605.03221)
- **Li, Lin, Chen, Cheng (2024), "Iterative Online Image Synthesis via Diffusion Model for Imbalanced Classification" (IOIS), MICCAI 2024; arXiv:2403.08407. Evaluated on HAM10000 and APTOS.** Its Accuracy Adaptive Sampling (AAS) module gives more synthetic samples to classes with lower training accuracy, so per-class allocation is driven by classifier feedback rather than a fixed fill-to-N. This is the closest prior work to an "adaptive per-class count" on HAM10000. [verified: abstract] — [arXiv:2403.08407](https://arxiv.org/abs/2403.08407); [MICCAI page](https://papers.miccai.org/miccai-2024/427-Paper0901.html)
- **FedEAS, "WHERE to Generate Matters: Budget-Aware Synthetic Augmentation for Label Skewed Federated Learning", arXiv:2607.06616.** An entropy-adaptive per-class generation budget recovers most of the accuracy gain of full class balancing while using 94.1% less generation budget. This is evidence that filling every class completely is not needed. Federated setting. [verified via search snippet only] — [arXiv:2607.06616](https://arxiv.org/html/2607.06616)
- **DALDA (2024), arXiv:2409.16949.** Few-shot. It notes that synthetic images become less effective as the number of real examples per class grows, and ties its guidance setting to n (σ = 0.05 × n). [verified via search snippet] — [arXiv:2409.16949](https://arxiv.org/abs/2409.16949)
- Histopathology: **Xue et al., "Selective Synthetic Augmentation with HistoGAN", arXiv:2111.06399.** Synthetic patches are filtered by label confidence and feature similarity to real images. The finding is that which synthetic images you keep matters, not just how many. [verified: search snippet] — [arXiv:2111.06399](https://arxiv.org/abs/2111.06399)

### Inferences
- Ratios reported as best range from 1:1 (Azizi, natural images) to more than 10:1 with saturation (Sagers, dermatology, data-limited) to no saturation seen (Rehman, CXR). The best ratio clearly depends on generator quality, real-set size and task, so there is no universal number to borrow for E4.
- Fill-to-head is the common default on ISIC/HAM. Because it gives the largest additions (often 20x-50x real) to the rarest classes, gains on DF/VASC under fill-to-head mix up "more data" with "more data where real data was thinnest".

### Gaps
- A claim that chest X-ray nodule synthesis peaks at about 1:6 real:synthetic and then declines appeared in a search-engine summary attributed to Goyal et al. (MIDL 2026, arXiv:2603.01659). The abstract does not state it, so it is unverified. Do not cite without checking the PDF.
- Found no MRI paper with a clean quantity sweep.
- Found no HAM10000 paper that sweeps a per-class count (e.g., 0/100/300/1000 per class) with multiple seeds.

## Q3. Does the optimal amount differ per class or depend on generator quality for that class?

### Takeaway
Yes, though mostly indirectly. Fan et al. show per-class scaling ability tracks whether the generator can render that class. Adaptive-allocation methods (IOIS on HAM10000, FedEAS) beat or match uniform/fill strategies at lower budgets. Filtering-based work suggests that bad synthetic images for a class hurt, so the optimal count for a poorly generated class may be near zero.

### Cited Findings
- Fan et al.: the easy/scaling/poor class split, with "poor" classes limited by the generator's inability to render the concept. — [arXiv:2312.04567](https://arxiv.org/html/2312.04567)
- Azizi et al. put the decline at high multiples down to generative model bias, which plausibly varies by class (inference by the authors at dataset level). — [arXiv:2304.08466](https://arxiv.org/html/2304.08466)
- IOIS: per-class allocation follows per-class training accuracy, evaluated on HAM10000. — [arXiv:2403.08407](https://arxiv.org/abs/2403.08407)
- FedEAS: an adaptive per-class budget gets close to full-balance accuracy at 5.9% of the budget. — [arXiv:2607.06616](https://arxiv.org/html/2607.06616)
- Jiang et al.: OOD filtering keeps only 20-60% of generated samples, which implies the usable count per class is set by quality, not only by the target. — [arXiv:2605.03221](https://arxiv.org/html/2605.03221)
- Wang et al.: the synthetic-to-real equivalence ratio depends on base set size and on how much synthetic data is already present. Classes with very different real counts (nv about 6.7k vs df about 115 in HAM10000) should therefore have different marginal value per synthetic image. — [arXiv:2508.09550](https://arxiv.org/abs/2508.09550)

### Inferences
- A per-class optimum is well motivated. A dataset-level optimum (Azizi) averages over classes that may peak at very different counts.
- In E4, a per-class quantity curve whose peak location differs between classes, and correlates with a per-class generator-quality signal, would be the direct test. A uniform curve shape across classes would argue against adaptivity.

### Gaps
- Found no paper that directly estimates a per-class optimal synthetic count and correlates it with a per-class generator-fidelity metric. E4 may be filling a real gap.

## Q4. Model collapse / self-consuming loops: relevance to mixing ratios

### Takeaway
The collapse literature concerns generative models retrained on their own outputs over many generations, not a one-shot classifier augmentation like E4. Its main lesson still carries over: keeping, or accumulating, real data with synthetic data prevents collapse, while replacing real with synthetic causes it.

### Cited Findings
- **Shumailov, Shumaylov, Zhao, Papernot, Anderson, Gal (2024), "AI models collapse when trained on recursively generated data", Nature 631:755-759; DOI 10.1038/s41586-024-07566-y.** Training recursively on model-generated data loses the tails of the distribution first, and eventually collapses it. [not re-fetched] — [Nature](https://www.nature.com/articles/s41586-024-07566-y)
- **Alemohammad, Casco-Rodriguez, Luzi, Humayun, Babaei, LeJeune, Siahkoohi, Baraniuk (2024), "Self-Consuming Generative Models Go MAD", ICLR 2024.** Three loop types are tested: fully synthetic, synthetic plus fixed real, and synthetic plus fresh real. Without enough fresh real data in each generation, quality (precision) or diversity (recall) steadily falls. [verified: abstract/summary] — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/ebc042e767de551803ccfcc45e2454f5-Paper-Conference.pdf)
- **Gerstgrasser et al. (2024), "Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data", arXiv:2404.01413.** Adding synthetic data on top of real data keeps test error bounded regardless of iteration count (proved for linear models). Replacing real data makes error grow linearly with iterations. [verified: summary] — [arXiv:2404.01413](https://arxiv.org/abs/2404.01413)

### Inferences
- E4 keeps all real HAM10000 data and adds synthetic on top, which is the "accumulate" regime, so classic collapse is not the expected failure mode.
- The relevant analogue is tail loss. Shumailov finds the tails go first. If the LoRA generator collapses rare-class variation (e.g., synthetic mel read as nv), adding large amounts of it could narrow the decision regions for those classes. That fits a peak-then-decline curve.

### Gaps
- Found no classifier-augmentation paper that frames the decline at high synthetic ratios explicitly in terms of model-collapse theory.

## Q5. Statistical methodology for testing quantity effects

### Takeaway
Most synthetic-augmentation papers report single runs or k-fold means without CIs or multiple-comparison correction. The better medical papers use bootstrap CIs and paired tests. Multi-seed runs with Holm-corrected paired comparisons at each quantity level (as in E4) would be more rigorous than most of the literature.

### Cited Findings
- Rehman et al. 2026 (CXR): 250 bootstrap resamples for 95% CIs and paired t-tests at p < 0.05. Seeds not reported. — [PLOS Digital Health](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001277)
- Jiang et al. 2026 (ISIC2019): 5-fold stratified CV, no CIs or significance tests. — [arXiv:2605.03221](https://arxiv.org/html/2605.03221)
- Azizi et al. and Fan et al. present quantity curves as point estimates per configuration. Seed-level variance was not seen in the sections read (low confidence: not checked thoroughly). — [arXiv:2304.08466](https://arxiv.org/html/2304.08466); [arXiv:2312.04567](https://arxiv.org/html/2312.04567)
- **Bouthillier et al. (2021), "Accounting for Variance in Machine Learning Benchmarks", MLSys 2021; arXiv:2103.03098.** Data sampling, initialization and other sources of randomness cause variance large enough to flip conclusions. They recommend randomizing as many sources of variation as possible and using several runs. [not re-fetched] — [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- **Holm (1979), "A simple sequentially rejective multiple test procedure", Scandinavian Journal of Statistics 6(2):65-70.** The standard step-down correction controlling family-wise error rate (classic reference, before 2021). [not re-fetched] — [JSTOR](https://www.jstor.org/stable/4615733)

### Inferences
- With HAM10000's small rare classes (df about 115, vasc about 142 images), per-class recall on the test split will be very noisy. Multiple training seeds plus a bootstrap over test images, with Holm across quantity levels, is a defensible design, and few published comparators match it.
- Single-run "peak" ratios in the literature (1:1, 10:1, 1:6) may partly reflect noise. E4's multi-seed design can say so explicitly.

### Gaps
- Found no synthetic-augmentation paper that applies Holm or another FWER correction across synthetic quantity levels.
- Did not survey trend tests (e.g., Jonckheere-Terpstra) or monotone-regression approaches for dose-response-style quantity curves in this literature.
