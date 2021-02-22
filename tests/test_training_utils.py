import pytest
import torch
import torch.utils.data

import class_attention as cat


def test_prepare_dataset():
    reduced_train_set, test_set, all_classes, test_classes = cat.training_utils.prepare_dataset(
        test_class_frac=0.1, dataset_frac=1.0
    )

    assert set(test_classes).issubset(set(all_classes))
    assert set(test_classes) != set(all_classes)
    assert set(test_classes).issubset(set(test_set["category"]))
    assert set(test_set["category"]) == set(all_classes)


def test_prepare_dataloaders():
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
    ) = cat.training_utils.prepare_dataloaders(
        test_class_frac=0.2, batch_size=32, model_name="distilbert-base-uncased", dataset_frac=0.1
    )

    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(test_dataloader, torch.utils.data.DataLoader)
    assert isinstance(train_dataloader.collate_fn, cat.CatCollator)
    assert isinstance(test_dataloader.collate_fn, cat.CatTestCollator)
    assert isinstance(all_classes_str, list)
    assert isinstance(test_classes_str, list)
    assert isinstance(all_classes_str[0], str)
    assert isinstance(test_classes_str[0], str)
    assert set(all_classes_str).issuperset(set(test_classes_str))
