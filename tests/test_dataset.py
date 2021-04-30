import pytest
import datasets
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


def test_getitem_wcrop():
    # fmt: off
    train_texts = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 100],
        [1, 2, 3, 4, 5, 6, 7, 100],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
        [1, 2, 3, 100],
    ]
    train_labels = [[1,], [3,], [2, 3], [1,]]
    # fmt: on

    arrow_dataset = datasets.Dataset.from_dict(
        {"text_ids": train_texts, "label_ids": train_labels}
    )

    dataset = cat.PreprocessedCatDatasetWCropAug(
        dataset=arrow_dataset,
        text_field="text_ids",
        class_field="label_ids",
        tokenizer=None,
        max_text_len=5,
    )
    x, y = dataset[0]

    assert len(x.shape) == 1
    assert x.shape[0] > 1
    assert len(y.shape) == 1
    assert x[0] == 1  # first token should remain
    assert x[-1] == 100  # last token should remain
