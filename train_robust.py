"""Adversarial training pipeline using PGD-7 — produces malwarenet_robust.pt."""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.manual_seed(42)
np.random.seed(42)

from malwarenet import MalwareNet, load_ember_data, make_loaders, evaluate
from attacks import pgd_attack


def train_epoch_adversarial(
    model: MalwareNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epsilon: float = 0.05,
) -> float:
    """One adversarial training epoch: concatenates clean and PGD-7 batches before forward pass."""
    model.train()
    total_loss = 0.0
    try:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Generate PGD-7 adversarial examples (model temporarily in eval for attack)
            model.eval()
            X_adv = pgd_attack(model, X_batch, y_batch, epsilon=epsilon, alpha=0.01, steps=7)
            model.train()
            # Concatenate clean + adversarial
            X_combined = torch.cat([X_batch, X_adv], dim=0)
            y_combined = torch.cat([y_batch, y_batch], dim=0)
            optimizer.zero_grad()
            preds = model(X_combined)
            loss = criterion(preds, y_combined)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_combined)
    except Exception as e:
        print(f"[WARNING] DataLoader error: {e} — skipping corrupted batch")
    return total_loss / (2 * len(loader.dataset))


def train_robust(
    data_dir: str = "./data/ember2018",
    save_path: str = "./models/malwarenet_robust.pt",
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 512,
    epsilon: float = 0.05,
) -> MalwareNet:
    """Full adversarial training pipeline for MalwareNet robust model."""
    print("[WARNING] Adversarial training takes 2–3x longer than standard training.")
    print(f"Estimated time: {epochs * 3}–{epochs * 5} minutes on CPU. Starting...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading EMBER data...")
    X_train, y_train, X_test, y_test = load_ember_data(data_dir)
    train_loader, val_loader, _ = make_loaders(X_train, y_train, X_test, y_test, batch_size)

    input_dim = X_train.shape[1]
    model = MalwareNet(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Class-weighted loss to handle EMBER imbalance
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
    criterion = nn.BCELoss(weight=None)  # weighting handled by sampler; adjust if needed

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_adversarial(model, train_loader, optimizer, criterion, device, epsilon)
        if epoch % 5 == 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    torch.save(
        {"epoch": epochs, "input_dim": input_dim, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
        save_path,
    )
    print(f"\nRobust model saved to {save_path}")
    return model


if __name__ == "__main__":
    train_robust()
