# Forget-What-Matters

# Summary of Multilingual Unlearning Experiment

This document summarizes the methodology, parameters, and data used in the unlearning experiment conducted on the XGLM language model.

---

## 1. What Was Done (Methodology)

The experiment attempted **Machine Unlearning** on a multilingual language model using the **LingTea (Negative Gradient)** method.

* **Model:** `facebook/xglm-564M`
* **Target:** Reduce the model's knowledge of the **Telugu (te), Bengali (bn), and Hindi (hi)** languages.
* **Training Objective:** The model was trained for **1 epoch** by applying a negative gradient step. This is achieved by **maximizing the loss** ($\text{Loss} = \text{Loss} \times -1$) on the retained dataset, which pushes the model's weights away from that specific knowledge space.
* **Stabilization Measures:** Due to the inherent instability of the negative gradient method (which often causes exploding gradients), the following measures were necessary:
    * **Frozen Embeddings:** The input embedding layers were frozen to prevent catastrophic divergence at the token level.
    * **Low Learning Rate:** The learning rate was set to a very conservative value.
    * **Gradient Clipping:** Used to cap the magnitude of gradients during backpropagation.

---

## 2. Key Parameters Used

| Parameter | Value | Role in Experiment |
| :--- | :--- | :--- |
| **Model** | `facebook/xglm-564M` | The base multilingual transformer model. |
| **Target Languages** | `te`, `bn`, `hi` | The specific languages being targeted for memory removal. |
| **Learning Rate** | $1 \times 10^{-5}$ | Controlled the aggressiveness of the negative gradient step. |
| **Accumulation Steps** | 4 | Used to gather gradients over multiple batches, providing a more stable average gradient. |
| **Epochs** | 1 | The duration of the unlearning training. |
| **Max Sequence Length** | 256 | The maximum length for input tokenization. |

---

## 3. Data Used

The experiment used small **dummy placeholder data** generated to strictly adhere to the expected **FLORES multilingual dataset structure** (JSONL format with language codes as top-level keys).

| Data Split | Purpose in Experiment |
| :--- | :--- |
| **Retain Set** | Used as the training data for the **negative gradient** (unlearning) step. |
| **Forget Set** | Used for **evaluation** to measure the reduction in model memorization (target: low MA). |
| **Test Set** | Used for **evaluation** to measure the model's general performance and fluency (target: stable MA, low PPL). |
