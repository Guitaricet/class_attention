import pytest
import torch
import torch.utils.data

import class_attention as cat

torch.manual_seed(40)


def test_make_mlp():
    mlp = cat.modelling_utils.make_mlp(n_layers=1, input_size=3, hidden_size=5, output_size=7)

    assert isinstance(mlp, torch.nn.Linear)
    assert mlp.weight.shape == (7, 3)

    mlp = cat.modelling_utils.make_mlp(n_layers=2, input_size=3, hidden_size=5, output_size=7)
    assert isinstance(mlp, torch.nn.Sequential)
    assert mlp[0].weight.shape == (5, 3)
    assert mlp[1].weight.shape == (7, 5)

    mlp = cat.modelling_utils.make_mlp(n_layers=2, input_size=3, hidden_size=5)
    assert isinstance(mlp, torch.nn.Sequential)
    assert mlp[0].weight.shape == (5, 3)
    assert mlp[1].weight.shape == (5, 5)
