import pytest
import torch
import torch.utils.data

import class_attention as cat


DATASET = "Fraser/news-category-dataset"


def test_prepare_dataset():
    reduced_train_set, test_set, all_classes, test_classes = cat.training_utils.prepare_dataset(
        dataset_name_or_path=DATASET,
        test_class_frac=0.1,
        dataset_frac=0.001,
    )

    assert set(test_classes).issubset(set(all_classes))
    assert set(test_classes) != set(all_classes)
    assert set(test_classes).issubset(set(test_set["category"]))
    assert (
        len(set(test_classes).intersection(set(reduced_train_set["category"]))) == 0
    ), "train dataset should not contain test classes"


def test_prepare_dataloaders():
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
    ) = cat.training_utils.prepare_dataloaders(
        dataset_name_or_path=DATASET,
        test_class_frac=0.2,
        batch_size=32,
        model_name="distilbert-base-uncased",
        dataset_frac=0.1,
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


def test_training_pipeline():
    # this test becomes VERY slow if num_workers > 0
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
    ) = cat.training_utils.prepare_dataloaders(
        dataset_name_or_path=DATASET,
        test_class_frac=0.2,
        batch_size=8,
        model_name="distilbert-base-uncased",
        dataset_frac=0.001,
        num_workers=0,
    )

    text_encoder = cat.modelling_utils.get_small_transformer()
    label_encoder = cat.modelling_utils.get_small_transformer()
    model = cat.ClassAttentionModel(
        text_encoder,
        label_encoder,
    )
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-4)

    cat.training_utils.train_cat_model(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        all_classes_str=all_classes_str,
        test_classes_str=test_classes_str,
        max_epochs=3,
        device="cpu",
    )
