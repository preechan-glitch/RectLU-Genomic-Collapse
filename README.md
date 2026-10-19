# RectLU-Genomic-Collapse
Disentangling Signal Propagation from Class Imbalance in CNN-Based Genomic Variant Detection (Smooth RectLU &amp; Hybrid Architecture)
# Smooth Activation Functions Can Silently Collapse to the Majority Class: Disentangling Signal Propagation from Class Imbalance in CNN-Based Genomic Variant Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Benchmark: GIAB](https://img.shields.io/badge/Benchmark-GIAB%20GRCh38%20v4.2.1-blue.svg)](https://www.nist.gov/programs-projects/genome-bottle)

Official PyTorch implementation, chromosome-held-out split manifests, and mechanistic diagnostic suite for our paper:
> **"Smooth Activation Functions Can Silently Collapse to the Majority Class: Disentangling Signal Propagation from Class Imbalance in CNN-Based Genomic Variant Detection"**  
> *Preecha Noiumkar and Thawatchai Chomsiri*  
> Research Center of Information Technology for the Future, Department of Information Technology, Faculty of Informatics, Mahasarakham University, Thailand.

---

## 📌 Overview & Key Findings
Deep learning models for germline genomic variant calling operate under extreme class imbalance ($>99:1$, majority reference rate $= 0.9944$). This repository provides the theoretical and empirical framework to diagnose and prevent **silent majority-class collapse** ($\text{Accuracy} = 0.9944, \text{MCC} = 0.0000$) in piecewise-smooth quadratic transition activations (`Smooth RectLU` and `AS-RectLU`):

1. **Second-Moment Variance Contraction (Proposition 1):** Under He initialization ($\sigma_w^2 = 2$), `Smooth RectLU` ($L=0, W=2$) transmits a forward second-moment gain of only $\mathbb{E}[f^2]/\mathbb{E}[x^2] = 0.088$, compared to $\ge 0.355$ for functional baselines (`ReLU`, `Leaky ReLU`, `PReLU`, `ELU`, `SELU`, `GELU`, `SiLU`, `Mish`, `Softplus`).
2. **Synergistic Interaction (H1 $\times$ H2):** `Smooth RectLU` converges normally on balanced $50:50$ data ($\text{MCC} = 0.9692$, $12.50\%$ dead units), but suffers a **$52.15\%$ dense-layer dead-unit cascade** under $99:1$ imbalance.
3. **Hybrid Activation Architecture (E5 Remedy):** While global weight scaling ($s > 1$) triggers linear-tail gradient explosion ($>8000$), pairing `Smooth RectLU` in convolutional feature extractors with a kinked activation (`ReLU`/`Leaky ReLU`) in the dense classification head suppresses dead units to $14.84\%$ and restores detection.

---

## 📂 Repository Structure
```text
├── evaluate_12_activations.py    # E1: 12-Activation benchmark on 1D-CNN (Pathway A) & 2D-CNN (Pathway B)
├── mechanistic_diagnostics.py    # E2, E4, E5: Class-ratio sweep, init scale (s), dead-unit & gradient norm tracker
├── requirements.txt              # Python package dependencies
└── README.md                     # Documentation and reproduction protocol
