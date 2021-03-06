import pytest

import torch
import transformers

import class_attention as cat
import tests.utils


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


@pytest.fixture()
def bahdanau_model():
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

    model = cat.ClassAttentionModel(
        text_encoder, label_encoder, hidden_size=7, attention_type="bahdanau"
    )
    return model


def test_forward_random_input(model):
    x = torch.randint(0, 100, size=[3, 5])
    c = torch.unique(torch.randint(0, 50, size=[7, 1])).unsqueeze(1)

    x_dict = {"input_ids": x}
    c_dict = {"input_ids": c}

    out = model(x_dict, c_dict)

    assert out.shape == (3, c.shape[0])


def test_forward(model):
    # fmt: off
    x_dict = {"input_ids": torch.LongTensor([
        [101, 9387, 6148, 17633, 2007, 2543, 1999, 1996, 7579, 102],
        [101, 6148, 17633, 2543, 1999, 1996, 7579, 102,     0,   0],
    ])}

    # c_dict is not classes of the x_dict, these are all possible classes
    c_dict = {"input_ids": torch.LongTensor([[41], [13], [12]])}
    # fmt: on

    out = model(x_dict, c_dict)

    assert out.shape == (2, 3)  # [batch_size, n_possible_classes]


def test_forward_bahdanau(bahdanau_model):
    # fmt: off
    x_dict = {"input_ids": torch.LongTensor([
        [101, 9387, 6148, 17633, 2007, 2543, 1999, 1996, 7579, 102],
        [101, 6148, 17633, 2543, 1999, 1996, 7579, 102,     0,   0],
    ])}

    # c_dict is not classes of the x_dict, these are all possible classes
    c_dict = {"input_ids": torch.LongTensor([[41], [13], [12]])}
    # fmt: on

    out = bahdanau_model(x_dict, c_dict)

    assert out.shape == (2, 3)  # [batch_size, n_possible_classes]


def test_glove_embedder():
    tests.utils.make_glove_file()

    emb_matrix, word2id = cat.utils.load_glove_from_file(tests.utils.GLOVE_TMP_PATH)
    embedder = cat.modelling.PreTrainedEmbeddingEncoder(
        embedding_matrix=emb_matrix, word2id=word2id
    )
    tests.utils.delete_glove_file()

    test_input = torch.LongTensor(
        [
            [3, 12, 0],
            [1, 0, 0],
            [3, 45, 3],
            [7, 0, 0],
        ]
    )
    out = embedder(input_ids=test_input)

    assert out is not None
    assert isinstance(out, tuple)
    assert len(out) == 1
    assert out[0].shape == (4, 1, 3)
