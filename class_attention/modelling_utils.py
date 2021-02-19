import torch
import torch.nn as nn

import transformers


def get_output_dim(model: transformers.PreTrainedModel):
    # it looks like Transformers changed this in some version
    # config = model.config
    # if isinstance(config, transformers.DistilBertConfig):
    #     return config.hidden_size
    return model.config.hidden_size


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
    return embeds / torch.sqrt(torch.sum(embeds * embeds, dim=1, keepdim=True))


def make_mlp(n_layers, input_size, hidden_size):
    if not isinstance(n_layers, int):
        raise ValueError(f"n_layers should be int, got {type(n_layers)} instead")

    if n_layers == 1:
        return nn.Linear(input_size, hidden_size)

    layers = [nn.Linear(input_size, hidden_size),]
    for _ in range(n_layers - 1):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, hidden_size))

    model = nn.Sequential(*layers)
    return model


def remove_smallest_principial_component(vecs, remove_n=1):
    """Leaves the dimensionality the same,
    but zeroes the direction corresponding
    to the smallest singular value eigenvector.

    U, S, V = PCA(A)
    A = U @ diag(S) @ V.T

    S is a list of singular values, sorted descending.
    """
    U, S, V = torch.pca_lowrank(vecs)
    S_r = S[:-remove_n - 1]
    U_r = U[:, :-remove_n - 1]
    V_r = V[:, :-remove_n - 1]

    A_r = U_r @ torch.diag(S_r) @ V_r.T
    return A_r


def get_small_transformer():
    return transformers.AutoModel.from_config(
        transformers.BertConfig(
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
        )
    )
