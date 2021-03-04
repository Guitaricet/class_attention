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


def test_bahdanau_fn_numeric():
    # fmt: off
    ht = torch.FloatTensor([
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
    ])
    hc = torch.FloatTensor([
        [-1,  0, -1,  0, -1],
        [-1, -1, -1, -1, -1],
        [ 0,  0, -1,  0, -1],
    ])
    scoring_fn = lambda x: torch.sum(x[:, :5] * x[:, 5:], dim=-1)
    assert scoring_fn(torch.cat([ht[0], hc[0]], dim=-1).unsqueeze(0)) == ht[0] @ hc[0].T

    expected_scores = torch.FloatTensor([
        [-3, -5, -2],
        [-6, -10, -4],
    ])
    # fmt: on

    scores = cat.modelling_utils.bahdanau_fn(key=ht, query=hc, scoring_fn=scoring_fn)
    assert torch.allclose(scores, expected_scores)
    assert not torch.allclose(torch.softmax(scores, dim=-1)[0], torch.softmax(scores, dim=-1)[1])


def test_bahdanau_fn_random():
    batch_size = 11
    n_classes = 3
    hidden1 = 97
    hidden2 = 89

    ht = 0.01 * torch.randn(batch_size, hidden1)
    hc = torch.randn(n_classes, hidden2) * torch.arange(n_classes).float().unsqueeze(-1)

    scoring_fn_weights = torch.randn(hidden1 + hidden2, 1)
    scoring_fn = lambda x: x @ scoring_fn_weights

    scores = cat.modelling_utils.bahdanau_fn(key=ht, query=hc, scoring_fn=scoring_fn)

    assert scores.shape == (batch_size, n_classes)
    assert torch.allclose(scores[0, 0], torch.cat([ht[0], hc[0]], dim=-1) @ scoring_fn_weights)
    assert torch.allclose(scores[0, 1], torch.cat([ht[0], hc[1]], dim=-1) @ scoring_fn_weights)
    assert torch.allclose(scores[0, 2], torch.cat([ht[0], hc[2]], dim=-1) @ scoring_fn_weights)
    assert torch.allclose(scores[1, 0], torch.cat([ht[1], hc[0]], dim=-1) @ scoring_fn_weights)
    assert torch.allclose(scores[1, 1], torch.cat([ht[1], hc[1]], dim=-1) @ scoring_fn_weights)

    # the numbers are very close due to a very similar distributions
    assert torch.any(torch.softmax(scores, dim=-1)[0] != torch.softmax(scores, dim=-1)[1])


def test_bahdanau_fn_scalar_prod():
    batch_size = 11
    n_classes = 3
    hidden1 = 5
    hidden2 = hidden1  # required for scalar product

    ht = torch.randn(batch_size, hidden1)
    hc = torch.randn(n_classes, hidden2)

    scoring_fn = lambda x: torch.sum(x[:, :hidden1] * x[:, hidden1:], dim=-1)

    scores = cat.modelling_utils.bahdanau_fn(key=ht, query=hc, scoring_fn=scoring_fn)

    assert torch.allclose(scores, ht @ hc.T)
