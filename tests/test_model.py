"""Sanity checks for MalwareNet model output shapes and value ranges."""

import torch
import pytest
from malwarenet import MalwareNet, EMBERDataset
import numpy as np


@pytest.fixture
def model() -> MalwareNet:
    return MalwareNet().eval()


def test_output_shape_single(model: MalwareNet) -> None:
    """Model should return a scalar per sample."""
    x = torch.rand(1, 146)
    out = model(x)
    assert out.shape == (1,), f"Expected (1,), got {out.shape}"


def test_output_shape_batch(model: MalwareNet) -> None:
    """Model should handle a batch of 64 samples."""
    x = torch.rand(64, 146)
    out = model(x)
    assert out.shape == (64,), f"Expected (64,), got {out.shape}"


def test_output_range(model: MalwareNet) -> None:
    """All outputs must be in [0, 1] (Sigmoid)."""
    x = torch.rand(100, 146)
    with torch.no_grad():
        out = model(x)
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output outside [0, 1]"


def test_ember_dataset_shapes() -> None:
    """EMBERDataset should return correct feature and label shapes."""
    X = np.random.rand(50, 146).astype(np.float32)
    y = np.random.randint(0, 2, 50).astype(np.float32)
    ds = EMBERDataset(X, y)
    feat, label = ds[0]
    assert feat.shape == (146,)
    assert label.shape == ()
