# Alignment between Reinforcement Pre-Training and Reinforcement Learning with Internal Feedback in Mathematical Tasks

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![CNPq PIBITI](https://img.shields.io/badge/CNPq-PIBITI%202025--2026-yellow)](https://www.gov.br/cnpq/pt-br)

This repository contains the code and experimental configurations for the Scientific Initiation project (IT/PIBITI CNPq, UFPR):
**"Analysis of the Alignment between Reinforcement Pre-Training and Reinforcement Learning with Internal Feedback in Mathematics and Programming Tasks"**

*   **Researcher:** Angelo Man de Oliveira
*   **Advisor:** Prof. Rodrigo Clemente Thom de Souza
*   **Institution:** Federal University of Paraná (UFPR) — Campus Jandaia do Sul
*   **Program:** PIBITI CNPq 2025–2026

---

## Overview

The primary objective of this project is to investigate whether **Reinforcement Pre-Training (RPT)** creates a pre-trained distribution that is more compatible and aligned with **Reinforcement Learning with Internal Feedback (RLIF)** compared to conventional **Supervised Pre-Training (SPT)**. We focus our analysis on complex mathematical reasoning tasks.

The experiment evaluates a base language model, **Qwen2.5-1.5B** (base, non-instruct version), across three pre-training states:
1.  **BASE**: The raw base model with no additional pre-training.
2.  **SPT**: The base model pre-trained on a supervised dataset of mathematical problems.
3.  **RPT**: The base model pre-trained using reinforcement learning objectives.

Each state is fine-tuned using two distinct post-training paradigms: **GRPO** (using external reward verifiers) and **Intuitor** (an RLIF method utilizing the model's internal self-certainty feedback). Pre-training is conducted on the **OmniMATH** corpus (4,428 Olympiad-level problems), and fine-tuning evaluation is benchmarked on **GSM8K** (pass@1 accuracy over 200 test examples, seed=42, with greedy decoding).

---

## Results

### Pre-Training Metrics (OmniMATH)
Evaluated on 200 mathematical problems at high-entropy token positions:

| Pre-Training State | Next-Token Prediction (NTP) Accuracy | Perplexity (ppl) |
| :--- | :---: | :---: |
| **BASE** (Qwen2.5-1.5B) | 23.5% | 2.65 |
| **SPT** | 26.5% | 2.47 |
| **RPT** | **35.5%** | 2.82 |

### Fine-Tuning Performance (GSM8K pass@1)
Results of post-training Qwen2.5-1.5B (greedy decoding, $N_{\text{test}}=200$):

| Pre-Training | Zero-Shot Accuracy | → GRPO (External RL) | → Intuitor (Internal RLIF) |
| :--- | :---: | :---: | :---: |
| **BASE** | 1.5% | **71.0%** | **67.5%** |
| **SPT** | 2.0% | 69.0% | 56.0% |
| **RPT** | 1.5% | 70.5% | 49.0% |

---

## Repository Structure

The repository consists of 6 Jupyter notebooks, each representing a specific combination of pre-trained state and fine-tuning algorithm:

*   [`grpo.ipynb`](./grpo.ipynb): Fine-tuning BASE (Qwen2.5-1.5B) using GRPO.
*   [`intuitor.ipynb`](./intuitor.ipynb): Fine-tuning BASE (Qwen2.5-1.5B) using Intuitor (RLIF).
*   [`spt_grpo.ipynb`](./spt_grpo.ipynb): Loading and merging the SPT checkpoint, followed by GRPO fine-tuning.
*   [`spt_intuitor.ipynb`](./spt_intuitor.ipynb): Loading and merging the SPT checkpoint, followed by Intuitor (RLIF) fine-tuning.
*   [`rpt_grpo.ipynb`](./rpt_grpo.ipynb): Loading and merging the RPT checkpoint, followed by GRPO fine-tuning.
*   [`rpt_intuitor.ipynb`](./rpt_intuitor.ipynb): Loading and merging the RPT checkpoint, followed by Intuitor (RLIF) fine-tuning.

---

## Setup

All notebooks are designed to run in Google Colab using an **NVIDIA L4 24GB** GPU environment. 

### Dependencies and Repository Clone
To clone the required Intuitor framework and install the pinned dependencies, run the following setup commands (replicated in Célula 1 of all notebooks):

```bash
# 1. Clone the official codebase
git clone --depth 1 https://github.com/sunblaze-ucb/Intuitor.git

# 2. Add src to Python path (in Python environment)
# import sys; sys.path.insert(0, "/content/Intuitor/open-r1-intuitor/src")

# 3. Install core dependencies
pip install -q --upgrade pip
pip install -q \
    transformers==4.51.0 \
    trl==0.17.0 \
    accelerate==1.4.0 \
    peft==0.15.1 \
    datasets==3.2.0

# 4. Install specific open-r1-intuitor requirements
pip install -q \
    math-verify \
    latex2sympy2_extended \
    liger-kernel \
    bitsandbytes \
    einops \
    wandb \
    sentencepiece \
    async-lru
```

---

## Usage

### Run Order Recommendation
To reproduce or analyze the experiments, run the notebooks in the following order:
1.  **Baseline establishing:** Run `grpo.ipynb` and `intuitor.ipynb` first, since they do not require external checkpoints.
2.  **SPT-initialized runs:** Place the SPT checkpoints in Google Drive, then execute `spt_grpo.ipynb` and `spt_intuitor.ipynb`.
3.  **RPT-initialized runs:** Place the RPT checkpoints in Google Drive, then execute `rpt_grpo.ipynb` and `rpt_intuitor.ipynb`.

### Execution and Configuration
-   Ensure that **L4 GPU** is selected as the runtime type in Google Colab.
-   Each notebook is pre-configured with a fixed `MODE` variable (`"grpo"` or `"intuitor"`) representing its target trainer algorithm.
-   Ensure Google Drive is mounted to load checkpoint weights (see [Pre-training Checkpoints](#pre-training-checkpoints)).

---

## Pre-Training Checkpoints

The pre-trained checkpoints are stored in Google Drive under your mount workspace. The notebooks expect them at:
-   **SPT Checkpoint:** `/content/drive/MyDrive/intuitor-ic/qwen-1.5b-spt-omnimath`
-   **RPT Checkpoint:** `/content/drive/MyDrive/intuitor-ic/qwen-1.5b-rpt-v3-omnimath`

### Loading and Merging in Notebooks
During execution, the notebooks automatically perform the following steps:
1.  Mount Google Drive: `drive.mount("/content/drive")`.
2.  Instantiate the base `Qwen2.5-1.5B` model.
3.  Load the LoRA adapter checkpoints from Drive using `PeftModel.from_pretrained(model, CKPT_PATH)`.
4.  Merge the weights and unload the PEFT container to establish a standalone backbone before starting fine-tuning:
    ```python
    model = model.merge_and_unload()
    ```

---

## Hyperparameters

### Pre-Training Phase (OmniMATH)

| Parameter | Supervised Pre-Training (SPT) | Reinforcement Pre-Training (RPT) |
| :--- | :--- | :--- |
| **Learning Rate ($lr$)** | `3e-4` | `1e-6` |
| **LoRA Config** | $r = 16$, $\alpha = 32$ | $r = 16$, $\alpha = 32$ |
| **Batch Configuration** | Batch size = 4, Grad Accum = 4 | Generations ($G$) = 8 |
| **Temperature ($T$)** | - | 0.8 |
| **KL Penalty ($\beta$)**| - | 0 |
| **Max Token Length** | 512 | - |
| **Training Duration** | 1 Epoch | 750 Steps |
| **Corpus Split** | 1,000 OmniMATH problems | OmniMATH P70 subset |
| **Reward Objective** | N/A | Dense Reward (Apêndice A) |

### Fine-Tuning Phase (All 6 Notebooks)

| Parameter | Value |
| :--- | :--- |
| **Learning Rate ($lr$)** | `3e-6` |
| **LoRA Config** | $r = 16$, $\alpha = 32$ |
| **Generations ($G$)** | 8 |
| **Temperature ($T$)** | 0.9 |
| **KL Penalty ($\beta$)** | 0.007 |
| **Training Steps** | 1,000 steps |
| **Train Samples ($N_{\text{train}}$)** | 1,000 (GSM8K) |
| **Dataset** | GSM8K |

---

## Key Findings

1.  **Robustness of GRPO:** GRPO demonstrates high stability and robustness to initial pre-training weights, with all initializations (BASE: 71.0%, SPT: 69.0%, RPT: 70.5%) converging to a performance plateau of approximately ~70% on GSM8K.
2.  **Sensitivity of Intuitor (RLIF):** Intuitor's policy optimization relies heavily on internal self-certainty signals. Starting from the pure BASE model yields the highest task performance (67.5%). Pre-training on OmniMATH using either supervised (SPT) or reinforcement (RPT) loss significantly degrades final performance (56.0% and 49.0% respectively).
3.  **Self-Certainty Degradation:** The results suggest that pre-training on complex, Olympiad-level tasks (OmniMATH) degrades the entropy-related confidence signal that Intuitor utilizes as internal reward feedback.
4.  **Refutation of Primary Hypothesis:** The experimental results **refute** the initial hypothesis: Reinforcement Pre-Training (RPT) did *not* lead to better alignment with RLIF (Intuitor) in mathematical reasoning tasks.

---

## References

*   **Intuitor / RLIF Paper:** Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, Dawn Song. *Learning to Reason without External Rewards*. ICLR 2026. [arXiv:2505.19590](https://arxiv.org/abs/2505.19590).
*   **RPT Paper:** *Reinforcement Pre-Training*. [arXiv:2506.08007](https://arxiv.org/abs/2506.08007).
*   **Qwen2.5 Base Model:** Qwen Team. *Qwen2.5: A New Family of Frontier Models*. [Hugging Face Repository](https://huggingface.co/Qwen).
*   **OmniMATH:** *Omni-MATH: A Universal Olympiad Level Mathematic Benchmark For Large Language Models*. [arXiv:2410.07985](https://arxiv.org/abs/2410.07985).
*   **GSM8K Dataset:** Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman. *Training Verifiers to Solve Math Word Problems*. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168).
