# CTCLAI: Constrained Token-Consistent Inference for AMPLIFY
Take-Home Submission — Zahid Hasan Pranto (2026)

---

## Overview

This repository extends **AMPLIFY** (Learning Actionless Motion Priors through Video) with an inference-time improvement:

> **CTCLAI — Constrained Token-Consistent, Loss-Aware Inference**

AMPLIFY decomposes policy learning into:

1. Motion Tokenization  
2. Forward Dynamics in Token Space  
3. Inverse Dynamics (Tokens → Action Chunks)

While modularization improves data efficiency, it introduces a failure mode:

**Plan–Act Mismatch:**  
The inverse dynamics output may not realize the predicted motion-token plan.

CTCLAI modifies only the inference procedure to reduce this mismatch.

---

## Repository Structure

```

AMPLIFY-main/
├── train_motion_tokenizer.py
├── train_forward_dynamics.py
├── train_inverse_dynamics.py
├── train_ctclai.py
├── eval_libero.py
│
├── amplify/
│   ├── amplify.py
│   └── bundle_amplify.py
│
├── cfg/
│   ├── train_motion_tokenizer.yaml
│   ├── train_forward_dynamics.yaml
│   ├── train_inverse_dynamics.yaml
│   ├── train_ctclai.yaml
│   └── eval_libero.yaml
│
├── preprocessing/
│   └── preprocess_libero.py

````

---

## Installation

### 1. Create environment

```bash
conda create -n amplify_ctclai python=3.10
conda activate amplify_ctclai
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Dataset Setup (LIBERO)

1. Download LIBERO dataset (see official LIBERO repo).
2. Update dataset path inside:

```
cfg/train_*.yaml
cfg/eval_libero.yaml
```

---

## Training Pipeline (AMPLIFY Baseline)

### 1. Train Motion Tokenizer

```bash
python train_motion_tokenizer.py \
    --config-name=train_motion_tokenizer
```

### 2. Train Forward Dynamics

```bash
python train_forward_dynamics.py \
    --config-name=train_forward_dynamics
```

### 3. Train Inverse Dynamics

```bash
python train_inverse_dynamics.py \
    --config-name=train_inverse_dynamics
```

---

## Baseline Evaluation (LIBERO)

```bash
python eval_libero.py \
    --config-name=eval_libero
```

This performs:

* Token rollout (T=16)
* Gaussian inverse decoding
* ACT-style temporal ensembling
* Success rate evaluation

---

## CTCLAI Training

CTCLAI adds constrained inference scoring.

Train CTCLAI components:

```bash
python train_ctclai.py \
    --config-name=train_ctclai
```

This trains:

* Token-consistency scorer
* Future-loss predictor

No retraining of tokenizer or forward model is required.

---

## CTCLAI Evaluation

Run evaluation with CTCLAI enabled:

```bash
python eval_libero.py \
    --config-name=eval_libero \
    ctclai.enable=true \
    ctclai.N=8 \
    ctclai.lambda_tok=1.0 \
    ctclai.lambda_loss=1.0 \
    ctclai.lambda_prior=0.1
```

---

## Evaluation Protocol

Evaluate:

* Baseline vs CTCLAI
* Success vs candidate count (N = 1, 4, 8, 16)
* Plan–act mismatch reduction
* Compute overhead

All experiments:

* Horizon T = 16
* Same tokenizer & forward checkpoints
* Same seeds

---

## Contribution

CTCLAI introduces a constrained inference operator for modular latent world models, improving robustness without modifying AMPLIFY’s training pipeline.

---
