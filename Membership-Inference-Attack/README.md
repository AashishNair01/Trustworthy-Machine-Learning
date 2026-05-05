# Membership Inference Attack (MIA) Project

## Overview
This project implements Membership Inference Attacks (MIA) to determine whether a given sample was part of a model’s training dataset. The goal is to evaluate privacy leakage in machine learning models by analyzing how easily membership information can be inferred from model outputs.

The target model is assumed to be a pretrained ResNet-18 classifier. The attack is evaluated using a public dataset for development and a private dataset for inference, with performance measured using the TPR@5%FPR metric.

---

## References
- Shokri et al., 2017 – Membership Inference Attacks against ML Models  
- Carlini et al., 2021 – Membership Inference Attacks from First Principles

---

## Authors
- Team XLVI: Aashish Nair, Sumedh Joshi
---

## Instructions

To reproduce the best results, navigate to the project directory:
Go to the project directory and run the script:

```bash id="run_cmd_one"

cd Membership-Inference-Attack && python task_template.py



