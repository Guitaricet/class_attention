import torch
import torch.utils.data
import numpy as np

import class_attention as cat
import tests.utils

np.random.seed(8)
torch.manual_seed(41)
DATASET = "Fraser/news-category-dataset"


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


def test_prepare_dataloaders_glove():
    tests.utils.make_glove_file()

    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
    ) = tests.utils.default_prepare_dataloaders(
        glove_path=tests.utils.GLOVE_TMP_PATH,
    )

    tests.utils.delete_glove_file()

    assert isinstance(train_dataloader.dataset.label_tokenizer, cat.utils.GloVeTokenizer)
    assert isinstance(test_dataloader.dataset.label_tokenizer, cat.utils.GloVeTokenizer)

    # we check glove tokenizer here too
    glove_tokenizer = test_dataloader.dataset.label_tokenizer
    out = glove_tokenizer.encode("arts")
    assert isinstance(out, np.ndarray)

    out = glove_tokenizer.encode("arts & culture")
    assert out.shape == (3,)

    out = glove_tokenizer.batch_encode_plus(["arts", "arts & culture"])
    assert "input_ids" in out
    assert out["input_ids"][0, 1] == 0, "padding failed or pad_id is not 0"

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


def test_train_cat_model():
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
        device="cpu",
    )


def test_train_cat_model_nolabel():
    # this test becomes VERY slow if num_workers > 0
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
    ) = tests.utils.default_prepare_dataloaders(
        p_no_class=0.5,
        p_extra_classes=0.5,
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
        model_optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        all_classes_str=all_classes_str,
        test_classes_str=test_classes_str,
        max_epochs=3,
        device="cpu",
    )


def test_train_cat_model_discriminator():
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
        device="cpu",
        discriminator=discriminator,
        discriminator_optimizer=discriminator_opt,
        discriminator_update_freq=2,
    )


def test_train_cat_model_extra_classes():
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
    )

    model_optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=1e-4)
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    tests.utils.make_extra_classes_file()
    extra_classes_dataloader = cat.training_utils.make_extra_classes_dataloader_from_file(
        file_path=tests.utils.EXTRA_CLASSES_TMP_PATH,
        tokenizer=train_dataloader.dataset.label_tokenizer,
        batch_size=4,
    )
    tests.utils.delete_extra_classes_file()

    cat.training_utils.train_cat_model(
        model=model,
        model_optimizer=model_optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        all_classes_str=all_classes_str,
        test_classes_str=test_classes_str,
        max_epochs=1,
        device="cpu",
        discriminator=discriminator,
        discriminator_optimizer=discriminator_opt,
        discriminator_update_freq=2,
        extra_classes_dataloader=extra_classes_dataloader,
    )


def test_make_extra_classes_dataloader_from_glove():
    tests.utils.make_glove_file()

    dataloader = cat.training_utils.make_extra_classes_dataloader_from_glove(
        tests.utils.GLOVE_TMP_PATH, batch_size=7
    )
    tests.utils.delete_glove_file()

    batch = next(iter(dataloader))

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (7, 1)  # figure out the numbers
