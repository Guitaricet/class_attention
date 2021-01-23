import pytest

import torch
import transformers

import class_attention as cat


@pytest.fixture()
def model():
    text_encoder = transformers.AutoModel.from_config(
        transformers.BertConfig(
            hidden_size=32, num_hidden_layers=2, intermediate_size=64, num_attention_heads=4
        )
    )
    label_encoder = transformers.AutoModel.from_config(
        transformers.BertConfig(
            vocab_size=50,
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
        )
    )

    model = cat.ClassAttentionModel(text_encoder, label_encoder, hidden_size=7)
    return model


def test_forward_random_input(model):
    x = torch.randint(0, 100, size=[3, 5])
    c = torch.randint(0, 50, size=[7, 1])

    x_dict = {'input_ids': x}
    c_dict = {'input_ids': c}

    out = model(text_input_dict=x_dict, classes_input_dict=c_dict)

    assert out.shape == (3, 7)


def test_forward(model):
    # fmt: off
    x_dict = {"input_ids": torch.LongTensor([
        [101, 9387, 6148, 17633, 2007, 2543, 1999, 1996, 7579, 102],
        [101, 6148, 17633, 2543, 1999, 1996, 7579, 102,     0,   0],
    ])}

    # c_dict is not labels of the x_dict, these are all possible labels
    c_dict = {"input_ids": torch.LongTensor([[41], [13], [12]])}
    # fmt: on

    out = model(x_dict, c_dict)

    assert out.shape == (2, 3)  # [batch_size, n_possible_classes]
