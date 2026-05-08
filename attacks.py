"""FGSM and PGD adversarial attack implementations for MalwareNet."""

import torch
import torch.nn as nn
from malwarenet import MalwareNet


def fgsm_attack(
    model: MalwareNet,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Single-step FGSM attack targeting evasion (malware scored as benign)."""
    x_adv = x.clone().detach().requires_grad_(True)
    model.zero_grad()
    criterion = nn.BCELoss()
    preds = model(x_adv)
    loss = criterion(preds, y)
    loss.backward()
    # Add gradient sign to maximize loss → minimize malware score (evasion toward benign)
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()


def pgd_attack(
    model: MalwareNet,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float = 0.01,
    steps: int = 40,
) -> torch.Tensor:
    """PGD attack with random start; targets evasion toward benign class."""
    criterion = nn.BCELoss()
    # Random start within epsilon-ball
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = (x + delta).clamp(0.0, 1.0).detach()

    for _ in range(steps):
        x_adv = x_adv.requires_grad_(True)
        model.zero_grad()
        preds = model(x_adv)
        loss = criterion(preds, y)
        loss.backward()
        # Step to maximize loss → minimize malware score (evasion toward benign)
        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv.detach() + alpha * grad_sign
        # Project back into epsilon-ball around original x
        delta = (x_adv - x).clamp(-epsilon, epsilon)
        x_adv = (x + delta).clamp(0.0, 1.0)

    return x_adv.detach()
