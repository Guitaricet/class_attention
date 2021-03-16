import pytest

import datasets
import transformers

import class_attention as cat

import tests.utils


def test_make_test_classes_only_dataloader():
    dataset_str = datasets.load_dataset("Fraser/news-category-dataset")["validation"]
    test_classes_str = ['POLITICS', 'WORLD NEWS', 'EDUCATION', 'TRAVEL', 'SPORTS', 'RELIGION', 'ARTS', 'COLLEGE']
    text_tokenizer = label_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    tco_dataloader = cat.evaluation_utils.make_test_classes_only_dataloader(
        dataset=dataset_str,
        test_classes_str=test_classes_str,
        text_tokenizer=text_tokenizer,
        label_tokenizer=label_tokenizer,
    )

    assert True


def test_make_test_classes_only_dataloader_integration():
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
    ) = tests.utils.default_prepare_dataloaders(
        dataset_frac=1.0,
        p_training_classes=0.0,
        glove_path=None,
    )

    tco_dataloader = cat.evaluation_utils.make_test_classes_only_dataloader(
        dataset=data["test"],
        test_classes_str=test_classes_str,
        text_tokenizer=train_dataloader.dataset.text_tokenizer,
        label_tokenizer=train_dataloader.dataset.label_tokenizer,
    )

    assert True
