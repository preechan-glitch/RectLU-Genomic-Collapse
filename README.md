# Smooth Activation Functions Can Silently Collapse to the Majority Class: Disentangling Signal Propagation from Class Imbalance in CNN-Based Genomic Variant Detection

**Authors:** Preecha Noiumkar and Thawatchai Chomsiri
**Affiliation:** Research Center of Information Technology for the Future, Department of Information Technology, Faculty of Informatics, Mahasarakham University, Mahasarakham 44150, Thailand
**Contact:** preecha.n@msu.ac.th

## Overview
Official PyTorch implementation, GIAB (GRCh38 v4.2.1) chromosome-held-out split manifests, and raw test confusion matrices for our study on activation signal propagation and class imbalance in CNN-based genomic variant calling.

## Repository Contents
- models/activations.py: Implementations of RectLU, Smooth RectLU, and AS-RectLU.
- models/cnn_pathways.py: 1D-CNN (k-mer vectors), 2D-CNN (BAM pileup tensors), and Hybrid Activation Architecture.
- splits/giab_grch38_v4.2.1_splits.csv: Chromosome-held-out dataset split counts and baseline majority accuracy.
- esults/: Complete 12-activation test confusion matrices and performance metrics (Pathway A and Pathway B).
- erify_variance_propagation.py: Monte-Carlo script reproducing Table 1 second-moment forward/backward gains.
