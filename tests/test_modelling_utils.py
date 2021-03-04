import pytest
import torch
import torch.utils.data

import class_attention as cat


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


def test_bahdanau_fn():
    batch_size = 11
    n_classes = 3
    hidden1 = 5
    hidden2 = 7

    ht = torch.randn(batch_size, hidden1)
    hc = torch.randn(n_classes, hidden2)

    scoring_fn_weights = torch.randn(hidden1 + hidden2, 1)
    scoring_fn = lambda x: x @ scoring_fn_weights

    scores = cat.modelling_utils.bahdanau_fn(key=ht, query=hc, scoring_fn=scoring_fn)

    assert scores.shape == (batch_size, n_classes)
    assert torch.allclose(scores[0, 0], torch.cat([ht[0], hc[0]], dim=-1) @ scoring_fn_weights)
    assert torch.allclose(scores[0, 1], torch.cat([ht[0], hc[1]], dim=-1) @ scoring_fn_weights)
    assert torch.allclose(scores[0, 2], torch.cat([ht[0], hc[2]], dim=-1) @ scoring_fn_weights)
    assert torch.allclose(scores[1, 0], torch.cat([ht[1], hc[0]], dim=-1) @ scoring_fn_weights)
    assert torch.allclose(scores[1, 1], torch.cat([ht[1], hc[1]], dim=-1) @ scoring_fn_weights)
