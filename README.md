# Adversarial ML — Attack and Defend a Malware Classifier

Train a neural network to classify PE binaries, attack it with FGSM and PGD, then harden it with adversarial training.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download EMBER 2018 (~1.6 GB) and place it under `data/ember2018/`:

```bash
# Using the ember library after pip install ember
python -c "import ember; ember.create_vectorized_features('./data/ember2018/')"
```

## Project phases

| Phase | Goal | Command |
|-------|------|---------|
| 1 | Train baseline (≥97% detection) | `python malwarenet.py` |
| 2–3 | FGSM + PGD attacks | Run cells in `notebooks/results.ipynb` |
| 5 | Adversarial training | `python train_robust.py` |
| 6 | Full evaluation + plots | `python evaluate.py` |

## Tests

```bash
pytest tests/
```

## Results

*(Add screenshot of `notebooks/robustness_curves.png` here after Phase 6)*

| Metric | Baseline | Robust |
|--------|----------|--------|
| Clean detection rate | — | — |
| FGSM @ ε=0.05 | — | — |
| PGD @ ε=0.05 | — | — |
| PGD @ ε=0.03 | — | — |

## Repository structure

```
adversarial_ai/
├── data/ember2018/          # EMBER dataset (gitignored)
├── models/                  # Saved checkpoints (gitignored)
├── notebooks/results.ipynb  # Evaluation and write-up
├── tests/
│   ├── test_attacks.py
│   └── test_model.py
├── malwarenet.py            # Model definition + training
├── attacks.py               # FGSM and PGD
├── train_robust.py          # Adversarial training
├── evaluate.py              # Robustness evaluation
└── requirements.txt
```
