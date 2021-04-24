import pytest
import torch
import torch.utils.data
import numpy as np

from accelerate import Accelerator

import class_attention as cat
import tests.utils

np.random.seed(8)
torch.manual_seed(41)
DATASET = "Fraser/news-category-dataset"


@pytest.fixture
def accelerator():
    return Accelerator()


def test_prepare_dataset():
    reduced_train_set, test_set, all_classes, test_classes = cat.training_utils.prepare_dataset(
        dataset_name_or_path=DATASET,
        test_class_frac=0.1,
        dataset_frac=0.001,
        class_field="category",
        test_set_name="validation",
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
        data,
    ) = tests.utils.default_prepare_dataloaders()

    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(test_dataloader, torch.utils.data.DataLoader)
    assert isinstance(train_dataloader.collate_fn, cat.CatCollator)
    assert isinstance(test_dataloader.collate_fn, cat.CatTestCollator)
    assert isinstance(all_classes_str, list)
    assert isinstance(test_classes_str, list)
    assert isinstance(all_classes_str[0], str)
    assert isinstance(test_classes_str[0], str)
    assert set(all_classes_str).issuperset(set(test_classes_str))
    assert train_dataloader.dataset != test_dataloader.dataset[0]
    assert isinstance(data, dict)
    assert "train" in data
    assert "test" in data


def test_train_cat_model(accelerator):
    # this test becomes VERY slow if num_workers > 0
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
    ) = tests.utils.default_prepare_dataloaders()

    text_encoder = cat.modelling_utils.get_small_transformer()
    label_encoder = cat.modelling_utils.get_small_transformer()
    model = cat.ClassAttentionModel(
        text_encoder,
        label_encoder,
    )
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-4)

    cat.training_utils.train_cat_model(
        model=model,
        model_optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        all_classes_str=all_classes_str,
        test_classes_str=test_classes_str,
        max_epochs=3,
        accelerator=accelerator,
    )


def test_train_cat_model_discriminator(accelerator):
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
    ) = tests.utils.default_prepare_dataloaders()

    text_encoder = cat.modelling_utils.get_small_transformer()
    label_encoder = cat.modelling_utils.get_small_transformer()
    model = cat.ClassAttentionModel(
        text_encoder,
        label_encoder,
        n_projection_layers=1,
        hidden_size=13,
    )

    discriminator = cat.modelling_utils.make_mlp(
        n_layers=2,
        input_size=model.final_hidden_size,
        hidden_size=16,
        output_size=1,
        spectral_normalization=True,
    )

    model_optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-4)
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    cat.training_utils.train_cat_model(
        model=model,
        model_optimizer=model_optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        all_classes_str=all_classes_str,
        test_classes_str=test_classes_str,
        max_epochs=1,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_opt,
        discriminator_update_freq=2,
        accelerator=accelerator,
    )
