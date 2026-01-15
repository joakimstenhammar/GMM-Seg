# Gaussian Mixture Model–Based Segmentation of Diffusive Trajectories

## Overview

This repository implements a **Gaussian Mixture Model (GMM)–based framework for the segmentation of particle trajectories** using stepwise displacement statistics. The method performs a **statistical segmentation of displacement time series into distinct regimes**.

The framework is fully unsupervised and does **not** rely on labeled training data or predefined segmentation thresholds. Any physical interpretation of the resulting segments (e.g., as diffusive states) must be performed a posteriori.

The project supports:

1. **Synthetic trajectory generation with known ground truth**, enabling quantitative benchmarking.
2. **Segmentation of experimental trajectory data**, where no ground truth is assumed.

A central feature of the approach is the **automatic optimization of a Gaussian temporal filter**, selected by minimizing overlap between mixture components.

---

## Scientific Motivation

Many physical and biological systems exhibit **state-dependent diffusion**, where particles alternate between regimes characterized by distinct diffusion coefficients. Accurately identifying these states from noisy trajectory data is a challenging inference problem.

This framework addresses the problem by:

- Representing trajectories through displacement statistics
- Modeling these statistics using Gaussian mixture models
- Optimizing temporal smoothing to maximize state separability
- Producing a **discrete state barcode** for each trajectory

---

## Repository Structure

- Synthetic two-state diffusion simulator: data_gen.py

- GMM segmentation and accuracy evaluation on synthetic data: GMM_Seg_synthetic.py

- GMM segmentation of experimental trajectory data: GMM_seg_exp.py

- raw-data/: Generated synthetic datasets

---

## Scripts Description

### Synthetic Data Generation (data_gen.py)

Simulates **two-state diffusive trajectories** with exponentially distributed state dwell times.

Key features:

- Alternating diffusion coefficients
- Optional temporal blurring (segment averaging)
- Optional localization noise

---

### Segmentation of Synthetic Data (GMM_Seg_synthetic.py)

Applies the GMM-based segmentation pipeline to synthetic trajectories and evaluates accuracy against ground truth.

---

### Segmentation of Experimental Data (GMM_seg_exp.py)

Applies the same segmentation framework to **experimental particle tracking data**.

Expected input columns:

- TRACK_ID (Particle label)
- FRAME
- POSITION_X
- POSITION_Y

---

## Usage

Synthetic data generation:
python data_gen.py

Segmentation of synthetic data:
python GMM_Seg_synthetic.py

Segmentation of experimental data:
python GMM_seg_exp.py

---

## Dependencies

- Python ≥ 3.8
- NumPy
- SciPy
- scikit-learn
- pandas

---
