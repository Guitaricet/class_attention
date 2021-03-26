import pytest
import os

import torch

import class_attention as cat
import tests.utils


@pytest.fixture()
def model():
    return tests.utils.model_factory()


@pytest.fixture()
def cross_attention_model():
    return tests.utils.model_factory(
        model_kwargs={
            "cross_attention_layers": 1,
            "cross_attention_heads": 1,
            "n_projection_layers": 1,
        }
    )


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


def test_forward_cross_attention(cross_attention_model):
    # fmt: off
    x_dict = {"input_ids": torch.LongTensor([
        [101, 9387, 6148, 17633, 2007, 2543, 1999, 1996, 7579, 102],
        [101, 6148, 17633, 2543, 1999, 1996, 7579, 102,     0,   0],
    ])}

    # c_dict is not classes of the x_dict, these are all possible classes
    c_dict = {"input_ids": torch.LongTensor([[41], [13], [12]])}
    # fmt: on

    out = cross_attention_model(x_dict, c_dict)

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


def test_save_load():
    # fmt: off
    x_dict = {"input_ids": torch.LongTensor([
        [101, 9387, 6148, 17633, 2007, 2543, 1999, 1996, 7579, 102],
        [101, 6148, 17633, 2543, 1999, 1996, 7579, 102,     0,   0],
    ])}

    # c_dict is not classes of the x_dict, these are all possible classes
    c_dict = {"input_ids": torch.LongTensor([[41], [13], [12]])}
    # fmt: on

    model1 = tests.utils.model_factory()
    model2 = tests.utils.model_factory()

    out1 = model1(x_dict, c_dict)
    out2 = model2(x_dict, c_dict)

    assert not torch.allclose(out1, out2), "initial models are the same"

    checkpoint_path = "tmp_model.pt"
    model1.save(checkpoint_path)
    model2.load_state_dict_from_checkpoint(checkpoint_path)
    os.remove(checkpoint_path)

    out2_loaded = model2(x_dict, c_dict)
    assert torch.allclose(out1, out2_loaded), "loaded model is different from saved"
