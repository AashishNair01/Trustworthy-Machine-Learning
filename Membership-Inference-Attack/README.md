# Membership Inference Attack (MIA) Project Report

## Overview
This project implements Membership Inference Attacks (MIA) to determine whether a given sample was part of a model’s training dataset. The goal is to evaluate privacy leakage in machine learning models by analyzing how easily membership information can be inferred from model outputs.

The target model is assumed to be a pretrained ResNet-18 classifier. The attack is evaluated using a public dataset for development and a private dataset for inference, with performance measured using the TPR@5%FPR metric.

---

## Dataset Description

- **Public Dataset (14,000 samples):**
  - `id`: Sample identifier  
  - `image`: Input image  
  - `label`: Ground-truth class  
  - `membership`: Binary label (1 = member, 0 = non-member)

- **Private Dataset:**
  - Used only for inference
  - No membership labels provided

---

## Approach

We explore multiple membership inference strategies:

### 1. Threshold-Based Attack
Uses softmax confidence, loss, and entropy scores to distinguish members from non-members.


### 3. LiRA (Likelihood Ratio Attack)
Approximates membership likelihood by comparing loss distributions using shadow models.

### 4. Ensemble Method
Combines confidence, loss, and LiRA scores using weighted averaging to improve robustness and stability.

---

## Model Assumption
- Target model: ResNet-18 (pretrained and fixed)
- Access: Black-box (only outputs/logits available)

---

## Evaluation Metric
- **TPR@5%FPR**
  - True Positive Rate at 5% False Positive Rate
  - Focuses on strong privacy-relevant detection performance

---

## Key Findings
- Loss and LiRA-based methods outperform simple confidence scoring.
- Ensemble methods provide the most stable and robust results.
- Membership leakage is clearly observable in model outputs, confirming privacy risks in standard training setups.

---

## Practical Implications
Successful MIA demonstrates that machine learning models can leak sensitive information about their training data. This raises concerns in privacy-critical domains such as healthcare, finance, and biometric systems, where even membership disclosure can be sensitive.

---

## Security Concerns
- Overfitted models are highly vulnerable to MIA
- Confidence scores and logits can leak training information
- Deployment via APIs increases the attack surface
- Highlights need for privacy-preserving techniques (e.g., differential privacy)

---

## References
- Shokri et al., 2017 – Membership Inference Attacks against ML Models  
- Carlini et al., 2021 – Membership Inference Attacks from First Principles

---

## Authors
Team XLVI  
Aashish Nair  
Sumedh Joshi
