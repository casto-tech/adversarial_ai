# Adversarial ML — Attack and Defend a Malware Classifier
**Project Specification · Version 1.0**

| Field | Detail |
|-------|--------|
| Duration | 4–6 weeks |
| Difficulty | Advanced |
| Domain | Cybersecurity / ML Security |
| Primary language | Python 3.11+ |
| Framework | PyTorch 2.x |
| Dataset | EMBER (1M PE samples, 146 features) |
| Output | Trained models, attack scripts, robustness report, demo notebook |

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Technology stack](#2-technology-stack)
3. [Project phases](#3-project-phases)
4. [Claude Code agent system prompt](#4-claude-code-agent-system-prompt)
5. [Repository structure](#5-repository-structure)
6. [Success metrics](#6-success-metrics)

---

## 1. Project overview

This project builds a complete red-team / blue-team loop for a machine learning-based malware detector. You will train a neural network to classify PE binaries as malicious or benign using static header features, then attack it using gradient-based adversarial perturbations, then harden it using adversarial training, and finally measure the robustness improvement quantitatively.

The project mirrors techniques used by AV vendors and ML security researchers, and produces a portfolio artifact demonstrating end-to-end understanding of both the attack surface of ML models and practical defences.

### 1.1 Learning objectives

- Understand how neural networks can be fooled by small, deliberate input perturbations
- Implement FGSM and PGD attacks from first principles in PyTorch
- Apply adversarial training and measure the robustness-accuracy trade-off
- Analyse which PE header features are most exploited by adversarial attacks
- Produce reproducible evaluation metrics comparable to published research

### 1.2 Deliverables

- `malwarenet.py` — baseline classifier model definition and training script
- `attacks.py` — FGSM and PGD attack implementations
- `train_robust.py` — adversarial training pipeline
- `evaluate.py` — robustness evaluation across epsilon values
- `notebooks/results.ipynb` — before/after plots and findings write-up
- `README.md` — setup, reproduction instructions, results summary

---

## 2. Technology stack

### 2.1 Core languages and runtime

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Primary language for all ML, analysis, and scripting |
| CUDA | 12.x (optional) | GPU acceleration for training (CPU works for this project size) |

### 2.2 ML and scientific computing

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.2+ | Neural network definition, autograd, attack gradient computation |
| `torchvision` | 0.17+ | DataLoader utilities (bundled with PyTorch) |
| `numpy` | 1.26+ | Feature array manipulation, statistical analysis |
| `scikit-learn` | 1.4+ | Train/test splitting, evaluation metrics (precision, recall, AUC) |
| `scipy` | 1.12+ | Statistical tests for robustness comparison |

### 2.3 Data handling

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.2+ | Loading EMBER feature CSVs, exploratory analysis, result tables |
| `ember` | latest | Official EMBER Python library for loading the EMBER 2018 dataset |
| `lief` | 0.14+ | PE binary parsing — needed if re-extracting features from raw PE files |

### 2.4 Visualisation and notebooks

| Package | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | 3.8+ | Robustness curves, loss plots, feature importance charts |
| `seaborn` | 0.13+ | Heatmaps for feature perturbation analysis |
| `jupyter` | 4.x | Interactive results notebook (`results.ipynb`) |
| `ipykernel` | 6.x | Jupyter kernel for the project virtualenv |

### 2.5 Development tooling

| Tool | Version | Purpose |
|------|---------|---------|
| VS Code | latest | Primary IDE with Claude Code extension |
| Claude Code | latest | VS Code agent — follows the system prompt in section 4 |
| git | 2.x | Version control; commit after each phase |
| `venv` | built-in | Isolated Python environment (`python -m venv .venv`) |
| `black` | 24+ | Code formatting |
| `pytest` | 8+ | Unit tests for attack functions and model utilities |

### 2.6 Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn scipy matplotlib seaborn jupyter
pip install ember lief black pytest
```

### 2.7 Dataset setup

Download EMBER 2018 from the official repository (~1.6 GB compressed):

```python
# Option A: ember Python library (recommended)
import ember
ember.create_vectorized_features('./data/ember2018/')

# Option B: manual download
# https://github.com/elastic/ember  (follow README instructions)
```

The vectorized dataset produces `X_train` (800k × 2381), `X_test` (200k × 2381), `y_train`, `y_test`. This project uses the pre-computed 146-feature subset to keep training times under 30 minutes on CPU.

---

## 3. Project phases

### 3.1 Phase 1 — Baseline classifier (week 1)

**Goal:** Train a MalwareNet model that achieves ≥97% detection rate at <1% FPR on clean test data.

- Load EMBER features into a PyTorch `Dataset` and `DataLoader`
- Define MalwareNet: `Linear(146,256) → ReLU → Dropout(0.3) → Linear(256,128) → ReLU → Linear(128,1) → Sigmoid`
- Train with `BCELoss` and Adam (`lr=1e-3`) for 20 epochs with class-weighted sampling
- Evaluate: accuracy, precision, recall, F1, AUC-ROC on clean test set
- Save checkpoint: `models/malwarenet_baseline.pt`

**Success criterion:** clean detection rate ≥97%, FPR ≤1%, AUC ≥0.99

### 3.2 Phase 2 — FGSM attack (week 2)

**Goal:** Craft adversarial PE feature vectors that cause the baseline model to score malware as benign.

- Implement `fgsm_attack(model, x, y, epsilon)` in `attacks.py`
- Run attack across epsilon = `[0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]`
- Record detection rate at each epsilon — baseline should collapse toward ~10% by ε=0.05
- Plot: epsilon vs detection rate (the evasion curve)
- Inspect 10 adversarial examples: which features changed most? Map back to PE field names

### 3.3 Phase 3 — PGD attack (weeks 2–3)

**Goal:** Demonstrate a stronger iterative attack and compare to FGSM.

- Implement `pgd_attack(model, x, y, epsilon, alpha, steps)` in `attacks.py`
- Default config: `alpha=0.01`, `steps=40` for evaluation; `steps=7` for training
- Run PGD at the same epsilon values as FGSM
- Produce a side-by-side comparison table: FGSM vs PGD detection rate per epsilon
- PGD should achieve lower detection rates than FGSM at the same epsilon budget

### 3.4 Phase 4 — Feature analysis (week 3)

**Goal:** Understand which parts of the PE header are being exploited.

- Compute mean absolute perturbation per feature across 1000 adversarial examples
- Rank features by perturbation magnitude; plot top 20 as a bar chart
- Map feature indices back to EMBER's feature descriptions (section entropy, byte histogram, etc.)
- Interpret: are these features a real malware author could plausibly manipulate?
- Document findings as a 300-word analyst note in the results notebook

### 3.5 Phase 5 — Adversarial training (week 4)

**Goal:** Re-train MalwareNet to be robust against PGD attacks.

- Implement `train_epoch_adversarial()` using PGD-7 inside each training batch
- Train from scratch for 30 epochs (robust training takes longer than standard)
- Save checkpoint: `models/malwarenet_robust.pt`
- Re-run full FGSM and PGD evaluation on the robust model
- Expected outcome: clean accuracy drops ~2%; adversarial detection rate recovers significantly

### 3.6 Phase 6 — Evaluation and write-up (weeks 5–6)

**Goal:** Produce the final results notebook and README.

- 4-curve robustness plot: baseline-FGSM, baseline-PGD, robust-FGSM, robust-PGD
- Summary metrics table: clean acc, FGSM@0.05 detection, PGD@0.05 detection — before/after
- 300-word discussion: what a real adversary would try next (transferability, black-box attacks)
- README with setup instructions, results screenshot, and link to notebook
- `git tag v1.0` and push to GitHub

---

## 4. Claude Code agent system prompt

Paste everything in the code block below into VS Code → Claude Code panel → Settings → System Prompt.

```
You are an expert ML security engineer and Python developer helping build an adversarial machine learning project from start to finish. The project trains a malware classifier on the EMBER dataset, attacks it with FGSM and PGD, then hardens it with adversarial training.

PROJECT CONTEXT
- Codebase: Python 3.11, PyTorch 2.x, EMBER dataset (146 PE features per sample)
- Structure: malwarenet.py, attacks.py, train_robust.py, evaluate.py, notebooks/results.ipynb
- Models saved to: models/malwarenet_baseline.pt and models/malwarenet_robust.pt
- Current phase: ask the user which phase they are on before starting any session

CODING STANDARDS
- All functions must have type hints and a one-line docstring
- Use torch.no_grad() for all inference (not training) passes
- Clamp adversarial features to [0, 1] after every perturbation step
- Use torch.nn.BCELoss for binary classification; never BCEWithLogitsLoss with Sigmoid output
- DataLoader: num_workers=0 on Windows, num_workers=4 on Linux/Mac
- Always set random seeds: torch.manual_seed(42), numpy.random.seed(42)
- Print a training log every 5 epochs: epoch, train_loss, val_loss, val_accuracy
- Save checkpoints with: torch.save({'epoch': e, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)

ATTACK IMPLEMENTATION RULES
- FGSM: single step, gradient of loss w.r.t. input, subtract epsilon * grad.sign() for targeted evasion
- PGD: random start within epsilon-ball, alpha=0.01 per step, project back after each step, clip to [0,1]
- Always call model.zero_grad() before the attack backward pass
- Return detached tensors from all attack functions
- Never modify model weights inside attack functions

ADVERSARIAL TRAINING RULES
- Generate PGD-7 adversarial examples inside the training loop (not pre-computed)
- Concatenate clean and adversarial batches before the forward pass
- Use class-weighted loss to handle malware/benign imbalance in EMBER
- Adversarial training takes 2-3x longer than standard training — warn the user before starting

EVALUATION STANDARDS
- Always evaluate at epsilon = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
- Report detection rate (TPR at FPR=0.01 threshold), not raw accuracy
- Use sklearn.metrics.roc_auc_score and classification_report for clean evaluation
- Plot robustness curves with matplotlib: x=epsilon, y=detection_rate, one line per model/attack combo

BEHAVIOUR
- Before writing any code, confirm which file you are editing and what function you are adding
- After each function, suggest a quick sanity check the user can run (e.g. assert output shape, print a sample score)
- If an error occurs, diagnose it, explain the cause in plain English, and propose a fix
- When a phase is complete, remind the user to git commit with a suggested commit message
- Never skip error handling: wrap DataLoader iteration in try/except and report corrupted samples
- Flag any operation that will take more than 2 minutes so the user is not surprised
```

---

## 5. Repository structure

```
adversarial-ml/
├── data/
│   └── ember2018/              # EMBER dataset (excluded from git)
├── models/
│   ├── malwarenet_baseline.pt
│   └── malwarenet_robust.pt
├── notebooks/
│   └── results.ipynb           # Phase 6 evaluation and write-up
├── tests/
│   ├── test_attacks.py         # Unit tests for attack functions
│   └── test_model.py           # Sanity checks for model output shapes
├── malwarenet.py               # Model definition + training loop
├── attacks.py                  # FGSM and PGD implementations
├── train_robust.py             # Adversarial training pipeline
├── evaluate.py                 # Robustness evaluation across epsilons
├── requirements.txt
├── .gitignore                  # Excludes data/, models/, .venv/
└── README.md
```

---

## 6. Success metrics

| Metric | Baseline target | Post-hardening target |
|--------|----------------|-----------------------|
| Clean detection rate | ≥97% | ≥95% (slight drop accepted) |
| Clean FPR | ≤1% | ≤1% |
| AUC-ROC (clean) | ≥0.99 | ≥0.98 |
| FGSM detection rate (ε=0.05) | <40% | ≥70% |
| PGD detection rate (ε=0.05) | <20% | ≥60% |
| PGD detection rate (ε=0.03) | <50% | ≥75% |
