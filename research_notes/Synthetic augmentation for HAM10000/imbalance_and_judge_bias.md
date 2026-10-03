# Class imbalance in HAM10000 and bias of the classifier / feature-extractor "judge" for synthetic samples

Provenance note: sources marked [V] were seen (search result abstract or fetched page) in this session. Sources marked [M] are well-known papers cited from prior knowledge whose bibliographic details were not re-fetched here; the URLs are the canonical arXiv/DOI links, but specific numbers attributed to [M] sources should be checked before going into the thesis.

## A1. HAM10000 class distribution, data issues, and recommended split practice

### Takeaway
HAM10000 is dominated by nevi (6,705 of 10,015 images, about 67%), and it holds several images per lesion (about 7,470 unique lesions). Image-level random splits leak lesion siblings across splits and inflate published numbers. Splits should be grouped by `lesion_id` and stratified by class, and duplicate removal across ISIC releases is recommended.

### Cited Findings
- Dataset paper: Tschandl P, Rosendahl C, Kittler H. "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." Scientific Data 5, 180161 (2018). DOI 10.1038/sdata.2018.161. 10,015 dermatoscopic images in 7 classes: akiec, bcc, bkl, df, mel, nv, vasc — [arXiv 1803.10417](https://arxiv.org/abs/1803.10417) [V for existence; per-class counts below are M]
- Per-class counts (commonly reproduced from the dataset metadata): nv 6,705 (66.9%), mel 1,113 (11.1%), bkl 1,099 (11.0%), bcc 514 (5.1%), akiec 327 (3.3%), vasc 142 (1.4%), df 115 (1.1%). That makes the nv:df ratio about 58:1 — [Tschandl et al. 2018](https://doi.org/10.1038/sdata.2018.161) [M: verify against `HAM10000_metadata.csv`, which can be counted locally]
- 10,015 images but only 7,470 unique lesions, so 2,545 images are repeat photos of the same lesion. The dataset "should be split using lesion_id to prevent this form of data leakage" — [Galaxy Training Network, GLEAM Image Learner HAM10000 tutorial](https://training.galaxyproject.org/training-material/topics/statistics/tutorials/image_learner/tutorial.html) [V; a secondary/tutorial source]
- In that tutorial, keeping a lesion's images in one split lowered accuracy (0.88 vs 0.94 leaky), which shows the size of the inflation — [Galaxy tutorial](https://training.galaxyproject.org/training-material/topics/statistics/tutorials/image_learner/tutorial.html) [V]
- A leakage audit repo reports that a random image-level split puts a sibling of about 40% of test images (about 70% of melanomas) on the training side. Its leakage-free models reach about 0.69 balanced accuracy, "below most published HAM10000 numbers, most of which are leaky" — [jackdoesjava/ham-triage (GitHub)](https://github.com/jackdoesjava/ham-triage) [V; non-peer-reviewed, so treat as indicative]
- Another repo measures lesion overlap between the official ISIC 2018 Task 3 splits and documents a selection bias — [daorre1202/skin-lesion-foundation-models (GitHub)](https://github.com/daorre1202/skin-lesion-foundation-models) [V; non-peer-reviewed]
- Cassidy B, Kendrick C, Brodzicki A, Jaworek-Korjakowska J, Yap MH. "Analysis of the ISIC image datasets: Usage, benchmarks and recommendations." Medical Image Analysis 75:102305 (2022). DOI 10.1016/j.media.2021.102305. Found duplicates within and between ISIC releases, including train/test duplication. Recommends a duplicate-removal strategy (filename matching, ImageHash, MSE, SSIM, cosine similarity) and two curated datasets — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1361841521003509); [Ovid abstract](https://www.ovid.com/journals/meian/abstract/10.1016/j.media.2021.102305~analysis-of-the-isic-image-datasets-usage-benchmarks-and) [V]
- Data-quality problems also carry into derived sets (DermaMNIST, which is built from HAM10000): Abhishek et al., "Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets," Scientific Data (2025) — [Nature](https://www.nature.com/articles/s41597-025-04382-5); [arXiv 2401.14497](https://arxiv.org/abs/2401.14497) [V for existence; I did not fetch the details. I believe it reports leakage from duplicated lesions in DermaMNIST splits, but this is unverified]
- "Copycats: the many lives of a publicly available medical imaging dataset" (2024) discusses dataset duplication/derivatives, including ISIC — [arXiv 2402.06353](https://arxiv.org/abs/2402.06353) [V for existence only]

### Inferences
- With nv at about 67%, a constant-nv classifier gets about 67% accuracy but only 1/7 (about 0.143) balanced accuracy. Accuracy is therefore uninformative, and balanced accuracy (BA, the mean per-class recall) or macro-F1 is the correct headline metric.
- For the thesis, any judge (DenseNet classifier or DINOv2 probe) that was trained or validated on an image-level split would have inflated, sibling-memorising validation numbers. That would make its scores on synthetic images look more trustworthy than they are. The project should state that splits are grouped by `lesion_id`.
- The small classes (df 115, vasc 142 images, and fewer unique lesions) mean that per-class recall on a test split rests on a few dozen lesions. One or two lesions can swing df recall by several points, so bootstrap CIs per class are needed.

### Gaps
- I did not fetch exact unique-lesion counts per class. They can be computed locally from `HAM10000_metadata.csv`.
- I did not verify whether the official ISIC 2018 Task 3 validation/test sets (1,512 test images) overlap HAM10000 training lesions. Only a non-peer-reviewed repo claims to measure this.

## A2. Typical balanced accuracy, macro-F1, and melanoma recall on HAM10000 / ISIC 2018 Task 3

### Takeaway
The official ISIC 2018 Task 3 ranking metric was balanced multiclass accuracy (BMA). Top leaderboard entries reached about 0.85–0.89 BMA, but they used ensembles plus external data and were scored on the official held-out test set. Honest single-model, lesion-grouped HAM10000 results are much lower, roughly 0.6–0.8 BA. The project's BA of 0.58–0.60 sits at the low end, but not implausibly so for a single DenseNet on a leakage-free split at the chosen resolution.

### Cited Findings
- Challenge paper: Codella N, Rotemberg V, Tschandl P, Celebi ME, Dusza S, Gutman D, Helba B, et al. "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)." arXiv:1902.03368 (2019). Task 3 had 7 classes and 159 submissions — [arXiv 1902.03368](https://arxiv.org/abs/1902.03368) [V]
- Task 3 uses balanced multiclass accuracy as the ranking metric — [ISIC Forum, "Metric for the Task 3 / Lesion Diagnosis"](https://forum.isic-archive.com/t/metric-for-the-task-3-lesion-diagnosis/356) [V, title/snippet]
- Legacy leaderboard #1: BMA 0.885 using external data and ensembles, melanoma sensitivity 0.760, average specificity 0.833. A live-leaderboard top value of about 0.895 is reported second-hand, and MSM-CNN (multi-scale multi-network ensemble) got 0.862 BMA — [Mahbod et al., "Transfer learning using a multi-scale and multi-network ensemble for skin lesion classification," Computer Methods and Programs in Biomedicine (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0169260719311460); [arXiv 2102.01284, "Single Model Deep Learning on Imbalanced Small Datasets for Skin Lesion Classification"](https://arxiv.org/abs/2102.01284) [V from search snippets; these numbers came through the search summariser, so confirm against the leaderboard]
- The winning ISIC 2018 entry is reported as MetaOptima — [DermEngine blog](https://www.dermengine.com/blog/metaoptima-isic-2018-skin-disease-classification-artificial-intelligence) [V for existence; a company blog]
- Note that the winner's melanoma sensitivity (0.76) is well below the near-1.0 recall typical for nv, even at the top of the leaderboard. Mel/nv confusion is the dominant residual error in this task — [same sources as above] [V]
- Ensembles of multi-resolution EfficientNets with metadata (Gessert et al., MethodsX 2020) won ISIC 2019 — [ScienceDirect S2215016120300832](https://www.sciencedirect.com/science/article/pii/S2215016120300832) [V for existence]
- Tschandl P, Codella N, et al. "Comparison of the accuracy of human readers versus machine-learning algorithms for pigmented skin lesion classification: an open, web-based, international, diagnostic study." Lancet Oncology 20(7):938–947 (2019). DOI 10.1016/S1470-2045(19)30333-X. Top ISIC 2018 algorithms beat the average human readers on the same test images — [DOI](https://doi.org/10.1016/S1470-2045(19)30333-X) [M]
- A DINOv2-based model is reported at balanced accuracy 0.883 (CI 0.868–0.899) on HAM10000. The search summariser did not make clear which paper or setting (fine-tuned vs probe, split type) this comes from — [search hit around arXiv 2407.14757 / PanDerm](https://arxiv.org/abs/2407.14757) [V-uncertain: do not cite without confirming]
- A synthetic-augmentation benchmark on HAM10000+MILK10K (SkinGenBench) reports melanoma-F1 gains of 8–15 points from synthetic data, with ViT-B/16 F1 about 0.88 — [arXiv 2512.17585](https://arxiv.org/abs/2512.17585) [V from snippet]
- Controllable diffusion synthesis of six minority HAM10000 classes (mel, bcc, akiec, bkl, df, vasc) gave +2.32 F1 on average downstream — [Medical Image Analysis 2026, S1361841526002604](https://www.sciencedirect.com/science/article/pii/S1361841526002604) [V from snippet]

### Inferences
- Many published HAM10000 BA/accuracy figures above 0.85–0.90 probably come from image-level splits (A1). Comparing the project's grouped-split BA to them is not like-for-like and should be stated explicitly.
- The pattern of a mel recall gain paired with an nv recall loss (seen in V3a) matches the general mel/nv tradeoff visible even in challenge winners.

### Gaps
- I could not fetch the full Codella 2019 PDF or the live leaderboard to give per-class recall for top teams or the median submission.
- No source in this session gave typical single-model DenseNet-121 / ResNet-50 / EfficientNet BA on a lesion-grouped HAM10000 split. Only a non-peer-reviewed repo (about 0.69) was found.
- No reliable macro-F1 leaderboard values were found. ISIC 2018 did not rank by macro-F1.

## A3. Imbalance remedies and their tradeoffs (reweighting, focal loss, oversampling, logit adjustment, decoupling; minority vs nv recall)

### Takeaway
Every remedy shifts the decision boundary toward the minority classes. It buys mel/bkl/df recall and pays in nv recall (and nv precision of the minority classes). Post-hoc and decoupled methods (logit adjustment, classifier re-training on a frozen backbone) make this tradeoff explicit and tunable without retraining the features. A classifier trained with plain cross-entropy on HAM10000 carries a prior toward nv.

### Cited Findings
- Logit adjustment: Menon AK, Jayasumana S, Rawat AS, Jain H, Veit A, Kumar S. "Long-tail learning via logit adjustment." ICLR 2021. arXiv:2007.07314. It subtracts tau·log(prior) from the logits post hoc, or adds the prior offset inside the softmax-CE loss. Either way it targets balanced error, i.e. it is consistent for BA — [ICLR 2021 poster](https://iclr.cc/virtual/2021/poster/2675); [Google Research](https://research.google/pubs/long-tail-learning-via-logit-adjustment/); [dblp](https://dblp.org/rec/conf/iclr/MenonJRJVK21.html); [arXiv 2007.07314](https://arxiv.org/abs/2007.07314) [V]
- Decoupling: Kang B, Xie S, Rohrbach M, Yan Z, Gordo A, Feng J, Kalantidis Y. "Decoupling Representation and Classifier for Long-Tailed Recognition." ICLR 2020. arXiv:1910.09217. Instance-balanced sampling learns good features. Re-balancing only the classifier afterwards (cRT, tau-normalisation, LWS) matches or beats joint re-balancing. Much of the long-tail bias sits in the classifier weights' norms, not in the features — [arXiv 1910.09217](https://arxiv.org/abs/1910.09217); [code](https://github.com/facebookresearch/classifier-balancing); [OpenReview PDF](https://openreview.net/pdf?id=r1gRTCVFvB) [V for paper; mechanism summary is M]
- Focal loss: Lin T-Y, Goyal P, Girshick R, He K, Dollár P. "Focal Loss for Dense Object Detection." ICCV 2017. arXiv:1708.02002. It down-weights easy (often majority) examples — [arXiv 1708.02002](https://arxiv.org/abs/1708.02002) [M]
- Class-balanced loss via "effective number of samples": Cui Y, Jia M, Lin T-Y, Song Y, Belongie S. CVPR 2019. arXiv:1901.05555. Inverse-frequency reweighting over-corrects at extreme ratios, and the effective number smooths it — [arXiv 1901.05555](https://arxiv.org/abs/1901.05555) [M]
- Oversampling: Buda M, Maki A, Mazurowski MA. "A systematic study of the class imbalance problem in convolutional neural networks." Neural Networks 106:249–259 (2018). arXiv:1710.05381. Oversampling was generally the best of the methods compared, did not cause CNN overfitting in their setting, and thresholding (prior correction) should be used when overall accuracy matters — [arXiv 1710.05381](https://arxiv.org/abs/1710.05381) [M]
- LDAM-DRW: Cao K, Wei C, Gaidon A, Arechiga N, Ma T. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss." NeurIPS 2019. arXiv:1906.07413. Deferred re-weighting (train plain first, then re-weight) beats re-weighting from the start — [arXiv 1906.07413](https://arxiv.org/abs/1906.07413) [M]
- Calibration: Guo C, Pleiss G, Sun Y, Weinberger KQ. "On Calibration of Modern Neural Networks." ICML 2017. arXiv:1706.04599. Modern deep nets are overconfident, and temperature scaling fixes global miscalibration but not class-prior bias — [arXiv 1706.04599](https://arxiv.org/abs/1706.04599) [M]
- A repo with class-weighted HAM10000 training and threshold analysis is an applied example of the threshold tradeoff — [syed-tahmed/ham10000-skin-lesion-classification](https://github.com/syed-tahmed/ham10000-skin-lesion-classification) [V; non-peer-reviewed]

### Inferences
- Menon et al. imply that a CE-trained classifier's argmax is Bayes-optimal for the training prior (about 67% nv), not for balanced error. On ambiguous inputs, including synthetic mel/bkl that are slightly off-manifold, the nv prior term wins. That is a mechanistic, prior-driven explanation of an "nv sink" that does not require the synthetic images to be bad.
- A cheap diagnostic: apply post-hoc logit adjustment (subtract log class priors) to the DenseNet judge and re-score the synthetic set. If the share of synthetic mel/bkl read as nv drops a lot, the sink is largely prior bias in the judge. Per the project's "results are not errors" rule, run it as a pre-registered diagnostic, not as a way to tune toward a result.
- Kang et al. imply a related test: the nv sink may live in the linear head, not the features. A class-balanced re-trained head (cRT) on the same backbone is a second, low-cost judge variant.
- The V3a pattern (mel recall up, nv recall down, BA roughly unchanged at 0.582 vs 0.600) is the expected signature of moving along the minority/majority tradeoff curve, not of a net gain in discriminative information.

### Gaps
- No HAM10000-specific head-to-head comparison of focal vs reweighting vs logit adjustment on a lesion-grouped split was fetched this session.

## B1. Evidence that classifier-based scoring of generated images is biased by the scoring model

### Takeaway
The literature shows that (i) classifier- and feature-based metrics inherit their backbone's class structure and priors, (ii) self-training and confidence filtering on long-tailed data drift toward head classes, and (iii) using the same model to filter and to judge is confirmation-bias-prone. I found no paper that documents a "majority-class sink" when a dermoscopy classifier scores synthetic dermoscopy, so the thesis would be contributing that observation.

### Cited Findings
- Kynkäänniemi T, Karras T, Aittala M, Aila T, Lehtinen J. "The Role of ImageNet Classes in Fréchet Inception Distance." ICLR 2023. arXiv:2203.06026. The Inception pre-logit features are one affine map from ImageNet logits, so FID is roughly a distance between ImageNet class-probability sets. Matching Top-N class histograms lowers FID substantially without improving quality. On faces, FID is insensitive to the facial region, and classes like "bow tie" or "seat belt" dominate — [OpenReview](https://openreview.net/forum?id=4oXTQ6m_ws8); [arXiv 2203.06026](https://arxiv.org/abs/2203.06026) [V]
- Ravuri S, Vinyals O. "Classification Accuracy Score for Conditional Generative Models." NeurIPS 2019. arXiv:1905.10887. Training on synthetic data and testing on real data exposed failures that FID/IS missed: BigGAN-deep Top-1/Top-5 fell by 27.9%/41.6% versus real data. CAS also reveals per-class failures — [arXiv 1905.10887](https://arxiv.org/abs/1905.10887); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/fcf55a303b71b84d326fb1d06e332a26-Paper.pdf) [V]
- Pseudo-label / self-training bias under long-tailed data: confidence thresholds bias pseudo-labelled data toward dominant classes, causing a "severe distribution mismatch between true and pseudo labels". In imbalanced settings this can produce "an ever increasing bias towards the majority class" — [He et al., "Re-distributing Biased Pseudo Labels for Semi-supervised Semantic Segmentation," ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/He_Re-Distributing_Biased_Pseudo_Labels_for_Semi-Supervised_Semantic_Segmentation_A_Baseline_ICCV_2021_paper.pdf); [Uncertainty-aware Sampling for Long-tailed SSL, arXiv 2401.04435](https://arxiv.org/abs/2401.04435); [ULFine, arXiv 2505.05062](https://arxiv.org/abs/2505.05062) [V from snippets]
- Confirmation bias in pseudo-labelling generally: Arazo E, Ortego D, Albert P, O'Connor NE, McGuinness K. "Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning." IJCNN 2020. arXiv:1908.02983 — [arXiv 1908.02983](https://arxiv.org/abs/1908.02983) [M]
- Synthetic training data from text-to-image models: He R et al. "Is Synthetic Data from Generative Models Ready for Image Recognition?" ICLR 2023. arXiv:2210.07574. Uses CLIP-based filtering of generated samples, an example of an external judge rather than the downstream classifier — [arXiv 2210.07574](https://arxiv.org/abs/2210.07574) [M]
- Label-shift theory: Lipton ZC, Wang Y-X, Smola A. "Detecting and Correcting for Label Shift with Black Box Predictors." ICML 2018. arXiv:1802.03916. A classifier's confusion behaviour under a changed class prior is predictable from its training prior — [arXiv 1802.03916](https://arxiv.org/abs/1802.03916) [M]

### Inferences
- Combining Kynkäänniemi (features encode the backbone's label space) with Menon (CE classifiers encode the training prior) gives a principled basis for the finding: a judge trained on 67% nv labels maps out-of-distribution or ambiguous inputs to nv. A judge with a different training distribution (the DINOv2 probe) has its own sink (df). That is exactly a "classifier-specific" signal.
- That the DINOv2 probe sinks to df (the rarest class, 1.1%) rather than nv is notable. A linear probe trained with class-balanced weights, or one whose df decision region is large and diffuse in a sparsely sampled part of feature space, could absorb off-manifold samples. This is consistent with the probe's head being the source of the sink, not DINOv2's features (Kang et al. logic). It would need checking: if the probe used balanced weighting, df's region may be inflated.
- Filtering synthetic images with the same DenseNet that is later trained or evaluated is the generative-augmentation analogue of pseudo-label confirmation bias. Samples the classifier already "agrees with" survive, which narrows diversity toward its decision regions and gives little new information.

### Gaps
- I found no peer-reviewed paper that specifically shows dermoscopy classifiers assigning synthetic minority-class images to nv. That evidence would be original to the thesis.
- I did not verify how He et al. 2023 used CLIP filtering in detail, or whether they found filtering to help.

## B2. Feature extractors for evaluating medical synthetic images: Inception vs DINOv2 vs CLIP vs domain-specific encoders; dermatology foundation models

### Takeaway
On natural images, DINOv2-ViT-L/14 tracks human judgment better than Inception (Stein et al., NeurIPS 2023), and Inception-FID is driven by ImageNet classes (Kynkäänniemi et al., ICLR 2023). In medical imaging the picture is mixed: a MICCAI 2024 study found ImageNet-trained extractors (SwAV best) beat RadImageNet ones against expert Turing tests. "Domain-specific" therefore does not automatically mean "better judge". Dermatology foundation models (PanDerm, MONET) exist and are plausible independent judges, but they have not been validated as generative-evaluation feature spaces.

### Cited Findings
- Stein G, Cresswell JC, Hosseinzadeh R, Sui Y, Ross BL, Villecroze V, Liu Z, Caterini AL, Taylor JET, Loaiza-Ganem G. "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models." NeurIPS 2023. arXiv:2306.04675. This was the largest human evaluation of generative models to that date. FID under-ranks diffusion models that humans rate as more realistic. Swapping Inception-V3 for DINOv2-ViT-L/14 markedly improves metric–human correlation, and the authors recommend DINOv2. Code: dgm-eval — [arXiv 2306.04675](https://arxiv.org/abs/2306.04675); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/0bc795afae289ed465a65a3b4b1f4eb7-Paper-Conference.pdf); [GitHub dgm-eval](https://github.com/layer6ai-labs/dgm-eval) [V]
- Kynkäänniemi et al. ICLR 2023 (see B1): Inception-FID can be "gamed" by matching ImageNet class histograms — [arXiv 2203.06026](https://arxiv.org/abs/2203.06026) [V]
- Woodland M, Castelo A, Al Taie M, Albuquerque Marques Silva J, Eltaher M, Mohn F, Shieh A, Kundu S, Yung JP, Patel AB, Brock KK. "Feature Extraction for Generative Medical Imaging Evaluation: New Evidence Against an Evolving Trend." MICCAI 2024 (LNCS vol. 15012). arXiv:2311.13717. The study covered 16 StyleGAN2 models, 4 medical modalities, 4 augmentation schemes, and 11 ImageNet- or RadImageNet-trained extractors. ImageNet extractors were more consistent and aligned with human judgment, and the ImageNet-trained SwAV FD correlated significantly with experts. RadImageNet rankings were "volatile and inconsistent". The authors warn against privately trained medical extractors for benchmarking — [arXiv 2311.13717](https://arxiv.org/abs/2311.13717); [MICCAI page](https://papers.miccai.org/miccai-2024/314-Paper2251.html); [PMC12117514](https://pmc.ncbi.nlm.nih.gov/articles/PMC12117514/); [code](https://github.com/mckellwoodland/fid-med-eval) [V]. Note: dermoscopy was not among the modalities as far as the abstract indicates, and I did not confirm whether DINOv2 was among the 11 extractors.
- PanDerm: Yan S, Yu Z, et al. "A multimodal vision foundation model for clinical dermatology." Nature Medicine (2025). DOI 10.1038/s41591-025-03747-y. Self-supervised pretraining on more than 2M images from 11 institutions across 4 modalities (clinical, dermoscopy, total-body photography, dermatopathology). Reports SOTA on many tasks, often with 5–10% of labels, and linear-probe performance comparable to fine-tuning (Extended Data Table 37) — [Nature Medicine](https://www.nature.com/articles/s41591-025-03747-y); [arXiv 2410.15038](https://arxiv.org/abs/2410.15038); [GitHub](https://github.com/SiyuanYan1/PanDerm); [PMC12353815](https://pmc.ncbi.nlm.nih.gov/articles/PMC12353815/) [V]
- MONET: Kim C, Gadgil SU, DeGrave AJ, Omiye JA, Cai ZR, Daneshjou R, Lee S-I. "Transparent medical image AI via an image–text foundation model grounded in medical literature." Nature Medicine (2024). DOI 10.1038/s41591-024-02887-x. A CLIP ViT-L/14 trained on 105,550 dermatology image–text pairs from the literature. It scores concept presence and supports data and model auditing (MA-MONET finds model mistakes) — [Nature Medicine](https://www.nature.com/articles/s41591-024-02887-x); [Hugging Face suinleelab/monet](https://huggingface.co/suinleelab/monet); [medRxiv](https://www.medrxiv.org/content/10.1101/2024.04.17.24305983v1.full.pdf) [V; full author list is M]
- Google "Derm Foundation" (an embedding model for dermatology images, released via Google Health AI Developer Foundations): I found no primary peer-reviewed paper or page this session — [Gap]
- MedImageInsight (Microsoft, 2024) is a general medical embedding model that includes dermatology — [arXiv 2410.06542](https://arxiv.org/abs/2410.06542) [V for existence only]
- A DINOv2-on-radiology study (linear probing across medical benchmarks) found DINOv2 competitive with weakly supervised models on larger datasets — [arXiv 2312.02366](https://arxiv.org/abs/2312.02366) [V from snippet]

### Inferences
- Woodland et al. imply that a dermoscopy-specific judge (PanDerm, MONET) is not automatically better than DINOv2. Each encoder has its own pretraining distribution and its own probe-induced biases. That supports judging with several heterogeneous encoders rather than swapping in one "better" one.
- For the thesis: the DINOv2 df sink argues against simply replacing the DenseNet with DINOv2. Stein et al.'s DINOv2 recommendation concerns distribution-level metrics (FD/precision/recall on natural images), not per-sample class assignment through a linear probe on 7 imbalanced dermoscopy classes. These are different uses.

### Gaps
- I found no paper validating PanDerm, MONET or DINOv2 feature spaces against dermatologist realism judgments of synthetic dermoscopy.
- I could not retrieve the PanDerm Extended Data Table 37 numbers for HAM10000 linear probing (PanDerm vs DINOv2 vs others).

## B3. Linear-probe calibration and per-class bias in foundation-model features for dermoscopy

### Takeaway
I found little direct evidence. The general long-tail literature says a linear head trained on imbalanced features carries the class prior in its weight norms and biases (Kang et al.), and post-hoc prior correction or tau-normalisation removes much of it (Menon et al.). No paper fetched this session reports per-class calibration of DINOv2/PanDerm linear probes on HAM10000.

### Cited Findings
- Classifier weight norms correlate with class frequency, and tau-normalising them or re-training the classifier with class-balanced sampling (cRT) corrects long-tail bias without touching the features — [Kang et al. ICLR 2020, arXiv 1910.09217](https://arxiv.org/abs/1910.09217) [V for paper; mechanism M]
- Post-hoc logit adjustment is a principled correction for a probe trained under a skewed prior — [Menon et al. ICLR 2021](https://arxiv.org/abs/2007.07314) [V]
- PanDerm reports that linear probing is roughly on par with fine-tuning, which implies strong frozen dermoscopy features — [PanDerm arXiv 2410.15038 v3](https://arxiv.org/html/2410.15038v3) [V; no per-class data]
- An applied repo on HAM10000 does lesion-level leakage auditing, calibrated posteriors, and conformal prediction sets, which is a practical template for per-class calibration checks — [jackdoesjava/ham-triage](https://github.com/jackdoesjava/ham-triage) [V; non-peer-reviewed]

### Inferences
- A probe's sink class reflects three things together: where real training samples sit in feature space, the probe's loss weighting, and how off-manifold samples project. A rare class like df, with only 115 images and diffuse support, can claim a large region if class weights are inverse-frequency. The thesis should report the probe's training weighting and per-class reliability diagrams on real held-out data before reading its synthetic-sample assignments.
- A non-parametric judge (k-NN in DINOv2 or PanDerm space against real training images, with class-balanced reference sets) avoids learned-head prior bias. It is a natural complement to a linear probe.

### Gaps
- No peer-reviewed per-class calibration (ECE per class, reliability diagrams) of foundation-model linear probes on HAM10000 was found.

## B4. Recommendations for independent or ensemble judges

### Takeaway
I found no consensus protocol specific to medical synthetic data. Best practice assembled from the sources: (1) use a judge that is independent of the downstream classifier being trained, (2) use several heterogeneous feature spaces (DINOv2 plus a domain model such as PanDerm/MONET, possibly SwAV per Woodland), (3) prefer downstream real-test utility (CAS/TSTR) over judge agreement, (4) correct judges for class priors, and (5) validate against human/expert judgment on a subset.

### Cited Findings
- Stein et al. recommend DINOv2-ViT-L/14 over Inception for FD-type metrics, and argue that no single metric suffices and human evaluation remains the reference — [arXiv 2306.04675](https://arxiv.org/abs/2306.04675) [V]
- Woodland et al. validate extractor choice against expert visual Turing tests and find the rankings can flip by extractor family. They recommend checking extractor–human agreement in the medical domain — [arXiv 2311.13717](https://arxiv.org/abs/2311.13717) [V]
- CAS (train on synthetic, test on real) measures downstream utility and exposes per-class failures that judges miss — [Ravuri & Vinyals, arXiv 1905.10887](https://arxiv.org/abs/1905.10887) [V]
- MONET's model-auditing mode (MA-MONET) offers concept-level, text-grounded checks, e.g. whether synthetic mel images show melanoma dermoscopic concepts. That makes it an independent semantic judge — [Kim et al., Nature Medicine 2024](https://www.nature.com/articles/s41591-024-02887-x) [V]
- Kynkäänniemi et al. caution that any feature-space metric can be moved by matching the backbone's class histogram, so a judge whose label space overlaps the training target is easy to "satisfy" — [arXiv 2203.06026](https://arxiv.org/abs/2203.06026) [V]

### Inferences
- If two independent judges with different training priors (DenseNet trained on HAM10000 vs a DINOv2 probe) disagree, and each has its own sink, then agreement between them is a stronger, less biased signal than either alone. Disagreement localises judge bias rather than generator failure. This directly supports reading the signals as "classifier-specific".
- The project's planned judge J1 (DINOv2 on classifier_train+gen_train, per memory) is consistent with these recommendations only if its head is prior-corrected and its sink behaviour is reported. A second, domain-specific encoder (PanDerm or MONET) would turn J1 into an ensemble and test whether the df sink is DINOv2-specific.
- Human/dermatologist spot checks on a small stratified subset (e.g. synthetic mel read as nv) would be the decisive tiebreaker, per Stein and Woodland.

### Gaps
- No paper found proposes a formal ensemble-judge protocol for medical synthetic data.
- No primary source was found for Google Derm Foundation.
- No peer-reviewed study was found that compares judge disagreement against human ratings for synthetic dermoscopy.
