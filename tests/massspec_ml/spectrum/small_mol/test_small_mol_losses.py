import pytest
from pytest import approx
import numpy as np
import torch
from massspec_ml.pytorch.base_objects import ModelInput, ModelOutput
from massspec_ml.pytorch.spectrum.small_mol.small_mol_losses import *


@pytest.fixture()
def batch():
    y = torch.tensor([[1.0, 1.0, 0.5], [-1.0, -1.0, -1.0], [1.0, 0.5, -1.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    return ModelInput(x=None, y=y, index=None)

@pytest.fixture()
def output():
    y_prime = torch.tensor([[1.1, 0.1, 0.8], [1.0, 1.0, 1.0], [0.2, 0.5, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    return ModelOutput(y_prime=y_prime, score=None, var=None)

def test_SearchScore(output, batch):
    score = SearchLoss()
    result = score(output, batch)
    assert torch.eq(result, 0.7958865)
