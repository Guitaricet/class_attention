import re
import pytest

import torch
import torch.utils.data
import transformers
import datasets

import class_attention as cat


@pytest.fixture()
def random_model():
    random_text_encoder = transformers.BertModel(
        transformers.BertConfig(num_hidden_layers=2, intermediate_size=256)
    )
    random_label_encoder = transformers.BertModel(
        transformers.BertConfig(num_hidden_layers=2, intermediate_size=256)
    )
    random_model = cat.ClassAttentionModel(
        random_text_encoder, random_label_encoder, hidden_size=768
    )

    return random_model


@pytest.fixture()
def dataloader():
    # fmt: off
    train_texts = [
        "This is a news article",
        "Good news, everyone!",
        "It is cloudy with a chance of meatballs",
        "This is a sport article",
    ]
    train_labels = ["News", "News", "Weather", "Sport"]
    # fmt: on

    text_tokenizer = cat.utils.make_whitespace_tokenizer(train_texts)
    label_tokenizer = cat.utils.make_whitespace_tokenizer(train_labels, unk_token=None)

    dataset = cat.CatDataset(train_texts, text_tokenizer, train_labels, label_tokenizer)

    possible_labels_str = ["News", "Weather", "Sport"]
    possible_labels_ids = torch.LongTensor(
        [label_tokenizer.encode(l).ids for l in possible_labels_str]
    )
    test_collator = cat.CatTestCollator(possible_labels_ids=possible_labels_ids, pad_token_id=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=test_collator)
    return dataloader


def test_validate_model_on_dataloader(random_model, dataloader):
    _acc = cat.utils.evaluate_model(random_model, dataloader, device="cpu")
    assert 0 <= _acc <= 1


def test_valiadte_model_per_class_on_dataloader(random_model, dataloader):
    labels = ["My News", "Weather", "Sport"]
    metrics = _acc = cat.utils.evaluate_model_per_class(
        random_model, dataloader, device="cpu", labels_str=labels, zeroshot_labels=["My News"]
    )

    at_least_one = False
    assert 0 <= metrics["acc"] <= 1
    for metric in ["P", "R"]:
        for label in labels:
            m = metrics[f"{metric}/{label}"]
            assert 0 <= m <= 1
            if m > 0 and m < 1:
                at_least_one = True

    assert at_least_one, "all metrics are either 0 or 1"
