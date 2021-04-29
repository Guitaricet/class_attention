import pytest

import datasets
import torch
import transformers

import class_attention as cat

import tests.utils


def test_make_test_classes_only_dataloader():
    dataset_str = datasets.load_dataset("Fraser/news-category-dataset")["validation"]
    test_classes_str = [
        "POLITICS",
        "WORLD NEWS",
        "EDUCATION",
        "TRAVEL",
        "SPORTS",
        "RELIGION",
        "ARTS",
        "COLLEGE",
    ]
    text_tokenizer = label_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    tco_dataloader = cat.evaluation_utils.make_test_classes_only_dataloader(
        dataset=dataset_str,
        test_classes_str=test_classes_str,
        text_tokenizer=text_tokenizer,
        label_tokenizer=label_tokenizer,
        text_field="headline",
        class_field="category",
    )

    assert True


def test_make_test_classes_only_dataloader_integration():
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
        _,
    ) = tests.utils.default_prepare_dataloaders(
        dataset_frac=1.0,
    )

    tco_dataloader = cat.evaluation_utils.make_test_classes_only_dataloader(
        dataset=data["test"],
        test_classes_str=test_classes_str,
        text_tokenizer=train_dataloader.dataset.text_tokenizer,
        label_tokenizer=train_dataloader.dataset.label_tokenizer,
        text_field="headline",
        class_field="category",
    )

    assert True


def test_precision_at_k():
    batch_size, hidden = 128, 3
    x = cat.modelling_utils.normalize_embeds(torch.randn(batch_size, hidden))
    y = cat.modelling_utils.normalize_embeds(torch.randn(batch_size, hidden))

    p_full = cat.evaluation_utils.precision_at_k(x, x, k=1)
    p_full_at5 = cat.evaluation_utils.precision_at_k(x, x, k=5)

    assert p_full == 1.0
    assert p_full_at5 == 1.0

    p_random = cat.evaluation_utils.precision_at_k(x, y, k=1)
    p_random_at5 = cat.evaluation_utils.precision_at_k(x, y, k=5)
    assert p_random > 0
    assert p_random_at5 > p_random


def test_get_all_embeddings():
    (
        train_dataloader,
        _,
        _,
        _,
        _,
        _,
    ) = tests.utils.default_prepare_dataloaders()
    model = tests.utils.model_factory()

    text_embs, class_embs = cat.evaluation_utils.get_all_embeddings(model, train_dataloader)
    assert text_embs.shape == class_embs.shape
