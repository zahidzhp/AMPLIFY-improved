
# CTCLAI: Constrained Token-Consistent Inference  
### Take-Home Submission – Modular World Models (AMPLIFY)

Author: Zahid Hasan Pranto  

---

## 1. Objective

This repository implements **CTCLAI**, an inference-time improvement for the modular world model **AMPLIFY**.

AMPLIFY decomposes policy learning into:
1. Motion Tokenization (video → discrete tokens)  
2. Forward Dynamics in token space  
3. Inverse Dynamics (tokens → action chunk)

A key failure mode in this decomposition is **plan–act mismatch**:
the decoded action chunk may not realize the predicted token rollout.

CTCLAI addresses this by introducing constrained candidate selection at inference time.

---

## 2. Method Summary

At each timestep:

1. Predict token rollout using the forward model.
2. Sample **N** candidate action chunks from the inverse policy.
3. Score each candidate using:
   - **Token consistency** (does the action realize the planned tokens?)
   - **Predicted future loss**
   - **Policy likelihood prior**
4. Execute the best candidate.

No retraining of tokenizer or forward model is required.

---

## 3. Repository Structure

```
.
├── models/
│   ├── token_predictor.py
│   └── loss_predictor.py
├── inference/
│   └── ctclai_wrapper.py
├── experiments/
│   ├── train_token_predictor.py
│   ├── train_loss_predictor.py
│   ├── run_baseline.py
│   └── run_ctclai.py
├── configs/
│   └── libero10.yaml
└── main.tex

````

---

## 4. Setup

### Environment

```bash
conda create -n ctclai python=3.10
conda activate ctclai
pip install torch torchvision numpy tqdm hydra-core
````

Ensure LIBERO dataset path is set in:

```
configs/libero10.yaml
```

---

## 5. Baseline Reproduction

Run standard AMPLIFY inference:

```bash
python experiments/run_baseline.py \
    --config configs/libero10.yaml \
    --seed 0
```

This evaluates:

* Token rollout (T = 16)
* Gaussian inverse decoding
* ACT-style temporal ensembling
* LIBERO-10 success rate

---

## 6. Training Auxiliary Predictors

### Token Consistency Model

```bash
python experiments/train_token_predictor.py \
    --config configs/libero10.yaml
```

### Future-Loss Predictor

```bash
python experiments/train_loss_predictor.py \
    --config configs/libero10.yaml
```

---

## 7. Running CTCLAI

```bash
python experiments/run_ctclai.py \
    --config configs/libero10.yaml \
    --N 8 \
    --lambda_tok 1.0 \
    --lambda_loss 1.0 \
    --lambda_prior 0.1
```

---

## 8. Core Equation

For candidate action (a^{(i)}):

[
a^* =
\arg\min_i
\lambda_{tok} C_{tok}
+
\lambda_{loss} C_{loss}
+
\lambda_{prior} C_{prior}
]

where:

* (C_{tok}): token-plan consistency
* (C_{loss}): predicted future risk
* (C_{prior}): policy likelihood

---

## 9. Evaluation

Evaluate:

* Baseline vs CTCLAI
* Success vs candidate count (N = 1, 4, 8, 16)
* Token mismatch reduction
* Compute overhead

---

## 10. Contribution

CTCLAI introduces a constrained inference operator for modular latent world models, improving robustness without modifying AMPLIFY’s training pipeline.
