"""Unit tests for FGSM and PGD attack functions."""

import torch
import pytest
from malwarenet import MalwareNet
from attacks import fgsm_attack, pgd_attack


@pytest.fixture
def model() -> MalwareNet:
    m = MalwareNet().eval()
    return m


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.rand(16, 146)
    y = torch.ones(16)  # all malware
    return x, y


def test_fgsm_output_shape(model: MalwareNet, batch: tuple) -> None:
    """FGSM output must match input shape."""
    x, y = batch
    x_adv = fgsm_attack(model, x, y, epsilon=0.05)
    assert x_adv.shape == x.shape


def test_fgsm_clamped(model: MalwareNet, batch: tuple) -> None:
    """FGSM output must be clamped to [0, 1]."""
    x, y = batch
    x_adv = fgsm_attack(model, x, y, epsilon=0.10)
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0


def test_fgsm_detached(model: MalwareNet, batch: tuple) -> None:
    """FGSM must return a detached tensor."""
    x, y = batch
    x_adv = fgsm_attack(model, x, y, epsilon=0.05)
    assert not x_adv.requires_grad


def test_fgsm_perturbation_bounded(model: MalwareNet, batch: tuple) -> None:
    """FGSM perturbation should not exceed epsilon (before clamping at boundaries)."""
    x, y = batch
    epsilon = 0.05
    x_adv = fgsm_attack(model, x, y, epsilon=epsilon)
    # After clamping, effective perturbation may be smaller at boundaries
    delta = (x_adv - x).abs()
    assert delta.max() <= epsilon + 1e-6, f"Max perturbation {delta.max():.4f} > epsilon {epsilon}"


def test_pgd_output_shape(model: MalwareNet, batch: tuple) -> None:
    """PGD output must match input shape."""
    x, y = batch
    x_adv = pgd_attack(model, x, y, epsilon=0.05, steps=5)
    assert x_adv.shape == x.shape


def test_pgd_clamped(model: MalwareNet, batch: tuple) -> None:
    """PGD output must be clamped to [0, 1]."""
    x, y = batch
    x_adv = pgd_attack(model, x, y, epsilon=0.10, steps=5)
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0


def test_pgd_detached(model: MalwareNet, batch: tuple) -> None:
    """PGD must return a detached tensor."""
    x, y = batch
    x_adv = pgd_attack(model, x, y, epsilon=0.05, steps=5)
    assert not x_adv.requires_grad


def test_pgd_stronger_than_fgsm(model: MalwareNet, batch: tuple) -> None:
    """PGD with many steps should achieve lower model confidence than FGSM."""
    x, y = batch
    epsilon = 0.05
    x_fgsm = fgsm_attack(model, x, y, epsilon=epsilon)
    x_pgd = pgd_attack(model, x, y, epsilon=epsilon, steps=20)
    with torch.no_grad():
        score_fgsm = model(x_fgsm).mean().item()
        score_pgd = model(x_pgd).mean().item()
    # PGD should drive scores lower (closer to benign)
    assert score_pgd <= score_fgsm, f"PGD score {score_pgd:.3f} not < FGSM score {score_fgsm:.3f}"


def test_model_weights_unchanged(model: MalwareNet, batch: tuple) -> None:
    """Attack functions must not modify model weights."""
    x, y = batch
    params_before = [p.clone() for p in model.parameters()]
    fgsm_attack(model, x, y, epsilon=0.05)
    pgd_attack(model, x, y, epsilon=0.05, steps=5)
    for p_before, p_after in zip(params_before, model.parameters()):
        assert torch.allclose(p_before, p_after), "Model weights were modified by attack"
