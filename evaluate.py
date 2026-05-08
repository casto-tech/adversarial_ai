"""Robustness evaluation across epsilon values for baseline and robust MalwareNet models."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report
from torch.utils.data import DataLoader

from malwarenet import MalwareNet, EMBERDataset, load_ember_data
from attacks import fgsm_attack, pgd_attack

EPSILONS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]


def load_model(checkpoint_path: str, device: torch.device) -> MalwareNet:
    """Load a MalwareNet from a saved checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    input_dim = ckpt.get("input_dim", 2381)
    model = MalwareNet(input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def detection_rate_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Compute TPR at the threshold where FPR <= target_fpr."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    # Find highest TPR where FPR is still within target
    valid = fpr <= target_fpr
    return float(tpr[valid].max()) if valid.any() else 0.0


def evaluate_at_epsilon(
    model: MalwareNet,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilon: float,
    attack: str,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Return detection rate (TPR@FPR=0.01) for a given epsilon and attack type."""
    dataset = EMBERDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        X_out = X_batch.clone()
        # Perturb only malware samples; benign samples stay clean (for FPR anchor)
        if epsilon > 0:
            mal_mask = y_batch == 1
            if mal_mask.sum() > 0:
                X_mal = X_batch[mal_mask]
                y_mal = y_batch[mal_mask]
                if attack == "fgsm":
                    X_out[mal_mask] = fgsm_attack(model, X_mal, y_mal, epsilon)
                elif attack == "pgd":
                    alpha = max(epsilon / 10, 1e-4)
                    X_out[mal_mask] = pgd_attack(model, X_mal, y_mal, epsilon, alpha=alpha, steps=40)
                else:
                    raise ValueError(f"Unknown attack: {attack}")
        with torch.no_grad():
            preds = model(X_out)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    return detection_rate_at_fpr(np.array(all_labels), np.array(all_preds))


def run_full_evaluation(
    baseline_path: str = "./models/malwarenet_baseline.pt",
    robust_path: str = "./models/malwarenet_robust.pt",
    data_dir: str = "./data/ember2018",
    output_plot: str = "./notebooks/robustness_curves.png",
) -> dict:
    """Run full evaluation across epsilons for both models and both attacks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, X_test, y_test = load_ember_data(data_dir)

    results: dict = {
        "baseline_fgsm": [],
        "baseline_pgd": [],
        "robust_fgsm": [],
        "robust_pgd": [],
    }

    baseline = load_model(baseline_path, device)
    robust = load_model(robust_path, device)

    for eps in EPSILONS:
        print(f"Evaluating epsilon={eps:.3f}...")
        results["baseline_fgsm"].append(evaluate_at_epsilon(baseline, X_test, y_test, eps, "fgsm", device))
        results["baseline_pgd"].append(evaluate_at_epsilon(baseline, X_test, y_test, eps, "pgd", device))
        results["robust_fgsm"].append(evaluate_at_epsilon(robust, X_test, y_test, eps, "fgsm", device))
        results["robust_pgd"].append(evaluate_at_epsilon(robust, X_test, y_test, eps, "pgd", device))

    # Plot robustness curves
    plt.figure(figsize=(8, 5))
    plt.plot(EPSILONS, results["baseline_fgsm"], "b-o", label="Baseline — FGSM")
    plt.plot(EPSILONS, results["baseline_pgd"], "b--s", label="Baseline — PGD")
    plt.plot(EPSILONS, results["robust_fgsm"], "r-o", label="Robust — FGSM")
    plt.plot(EPSILONS, results["robust_pgd"], "r--s", label="Robust — PGD")
    plt.xlabel("Epsilon")
    plt.ylabel("Detection Rate (TPR @ FPR=1%)")
    plt.title("MalwareNet Robustness Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.show()
    print(f"Plot saved to {output_plot}")

    # Summary table
    eps_idx_005 = EPSILONS.index(0.05)
    eps_idx_003 = EPSILONS.index(0.03)
    print("\n=== Summary Table ===")
    print(f"{'Metric':<35} {'Baseline':>10} {'Robust':>10}")
    print("-" * 57)
    print(f"{'Clean detection rate'::<35} {results['baseline_fgsm'][0]:>10.3f} {results['robust_fgsm'][0]:>10.3f}")
    print(f"{'FGSM @ ε=0.05'::<35} {results['baseline_fgsm'][eps_idx_005]:>10.3f} {results['robust_fgsm'][eps_idx_005]:>10.3f}")
    print(f"{'PGD @ ε=0.05'::<35} {results['baseline_pgd'][eps_idx_005]:>10.3f} {results['robust_pgd'][eps_idx_005]:>10.3f}")
    print(f"{'PGD @ ε=0.03'::<35} {results['baseline_pgd'][eps_idx_003]:>10.3f} {results['robust_pgd'][eps_idx_003]:>10.3f}")

    return results


def feature_perturbation_analysis(
    model: MalwareNet,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    n_samples: int = 1000,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Compute mean absolute perturbation per feature over n_samples adversarial examples."""
    mal_idx = np.where(y_test == 1)[0][:n_samples]
    X_mal = X_test[mal_idx]
    y_mal = y_test[mal_idx]

    X_t = torch.tensor(X_mal, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_mal, dtype=torch.float32).to(device)
    X_adv = pgd_attack(model, X_t, y_t, epsilon=epsilon, alpha=0.01, steps=40)

    perturbations = (X_adv.cpu().numpy() - X_mal)
    mean_abs_pert = np.abs(perturbations).mean(axis=0)
    return mean_abs_pert


if __name__ == "__main__":
    run_full_evaluation()
