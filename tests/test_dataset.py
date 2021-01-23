import pytest
import class_attention as cat


def test_getitem():
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
    x, y = dataset[0]

    assert len(x.shape) == 1
    assert x.shape[0] > 1
    assert len(y.shape) == 1
