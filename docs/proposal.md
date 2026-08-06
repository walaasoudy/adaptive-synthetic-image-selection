# Thesis Project Context

## Who I am
I am a Master's student in AI at Zewail City University. This is my master's thesis project.
I am implementing this framework myself, step by step, in order to generate results. After I
finish getting results, I will write the paper based on this implementation and the results.

## Thesis Title
An Adaptive Quality-Aware Synthetic Data Selection Framework for Medical Image Classification
Using Diffusion Foundation Models

## Dataset
CheXpert-v1.0-small — 224,316 chest radiographs from 65,240 patients, frontal and lateral views,
with 14 pathology labels (including uncertainty labels: -1, 0, 1). Located at `data/chexpert/`.

## Framework Overview
The framework consists of five stages. Implement them **one stage at a time, in order** —
do not jump ahead to a later stage until the current one is working and I've confirmed it.

### Stage 1 — Generative Model Setup
- Stable Diffusion XL (SDXL)
- LoRA Fine-Tuning on CheXpert chest X-rays

### Stage 2 — Synthetic Data Generation
- Generate synthetic medical images (chest X-rays) using the fine-tuned SDXL+LoRA model

### Stage 3 — Adaptive Synthetic Image Selection Module (ASISM) [Novel Contribution]
This is the core novel contribution of the thesis. It combines multiple algorithms to decide
which synthetic images are good enough to keep:
- CLIP / DINOv2 Similarity Scoring
- Image Quality Assessment (IQA)
- Uncertainty Estimation (Monte Carlo Dropout or Deep Ensembles)
- Grad-CAM / Score-CAM Explainability Verification
- Multi-Objective Ranking Network (Novel)
- Adaptive Threshold Learning (Novel)

### Stage 4 — Classifier Training
- Train the classifier using only the synthetic images selected by ASISM

### Stage 5 — Evaluation
Compare classifier performance across three training conditions:
- Real images only
- Real + all synthetic images (unfiltered)
- Real + selected synthetic images (ASISM-filtered)

## What I need from you
- Implement actual, runnable code for each stage — not just descriptions or pseudocode.
- Read `docs/literature_review.md` before starting, and ground design choices (e.g. what
  the ASISM module should look for, what baselines to compare against) in the papers
  summarized there — especially Rehman et al. (LoRA-tuned SD on CheXpert/MIMIC-CXR) and
  Niemeijer et al. (TSynD, uncertainty-guided synthesis), since my thesis directly extends
  the gap they leave open: neither optimizes how synthetic and real data should be
  selected and mixed after generation.
- If something in current published literature (post your training data) would change an
  implementation choice, flag it — but don't assume you have live search; tell me what to
  look up myself if you're not sure your information is current.
- Work stage by stage. After each stage, stop and let me review results before continuing.
- The end goal is a full set of experimental results (Stage 5 comparisons) that I will use
  to write a research paper for publication.