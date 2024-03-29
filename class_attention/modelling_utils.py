import torch
import torch.nn as nn

import transformers


def validate_inputs(text_input_dict, labels_input_dict):
    if not isinstance(text_input_dict, dict):
        raise ValueError("text input should be a dict")
    if not isinstance(labels_input_dict, dict):
        raise ValueError("classes input should be a dict")

    if labels_input_dict["input_ids"].shape[0] == 1:
        raise RuntimeError(
            "batch dimension of classes tensor is the number of possible classes and cannot be equal to one"
        )

    # check that labels_input does not have duplicated classes
    unique_classes = torch.unique(labels_input_dict["input_ids"], dim=0)
    if unique_classes.shape[0] != labels_input_dict["input_ids"].shape[0]:
        raise ValueError("labels_input should only contain unique classes")


def maybe_format_inputs(text_input, labels_input):
    if isinstance(text_input, torch.Tensor):
        text_input = {"input_ids": text_input}

    if isinstance(labels_input, torch.Tensor):
        labels_input = {"input_ids": labels_input}

    return text_input, labels_input


def normalize_embeds(embeds):
    """Returns a batch of vectors, each of L2 norm=1"""
    return embeds / torch.sqrt(torch.sum(embeds.pow(2), dim=1, keepdim=True))


def cos2(embeds):
    e_normed = normalize_embeds(embeds)
    sim = torch.pow(e_normed @ e_normed.T, 2)  # cos^2
    sim.sub_(
        torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
    )  # remove the diagonal which is always 1
    sim = sim.sum() / 2.0  # divide by two, because the similarity matrix is symmetric
    return sim


def make_mlp(
    n_layers,
    input_size,
    hidden_size,
    output_size=None,
    use_bias=True,
    activation_fn=None,
    dropout=0,
    spectral_normalization=False,
):
    if not isinstance(n_layers, int):
        raise ValueError(f"n_layers should be int, got {type(n_layers)} instead")
    if n_layers < 1:
        raise ValueError(n_layers)
    if activation_fn is None:
        activation_fn = nn.ReLU

    linear = nn.Linear
    if spectral_normalization:

        def linear(*args, **kwargs):
            return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

    output_size = output_size or hidden_size

    if n_layers == 1:
        return linear(input_size, output_size, bias=use_bias)

    layers = [
        linear(input_size, hidden_size),
    ]
    for _ in range(n_layers - 2):
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(activation_fn())
        layers.append(linear(hidden_size, hidden_size, bias=use_bias))

    layers.append(linear(hidden_size, output_size, bias=use_bias))

    model = nn.Sequential(*layers)
    return model


def remove_smallest_princpial_component(vecs, remove_n=1):
    """Leaves the dimensionality the same,
    but zeroes the direction corresponding
    to the smallest singular value eigenvector.

    U, S, V = PCA(A)
    A = U @ diag(S) @ V.T

    S is a list of singular values, sorted descending.
    """
    U, S, V = torch.pca_lowrank(vecs)
    S_r = S[: -remove_n - 1]
    U_r = U[:, : -remove_n - 1]
    V_r = V[:, : -remove_n - 1]

    A_r = U_r @ torch.diag(S_r) @ V_r.T
    return A_r


def get_small_transformer(hidden_size=8):
    return transformers.AutoModel.from_config(
        transformers.BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=2,
            intermediate_size=16,
            num_attention_heads=2,
        )
    )
