import torch

import datasets
import transformers

import class_attention as cat


def prepare_dataset(test_class_frac, dataset_frac=1.0):
    news_dataset = datasets.load_dataset("Fraser/news-category-dataset")
    train_set = news_dataset["train"]
    test_set = news_dataset["validation"]
    all_classes = list(set(news_dataset["train"]["category"]))

    if dataset_frac < 1:
        # some magic is happening here to make a toy dataset that is consistent, read carefully
        train_set = cat.utils.sample_dataset(news_dataset["train"], p=dataset_frac)

        classes_left = list(set(train_set["category"]))

        test_set = news_dataset["validation"]
        if len(all_classes) > len(classes_left):
            _, test_set = cat.utils.split_classes(test_set)

        test_set = cat.utils.sample_dataset(test_set, p=dataset_frac)

    reduced_train_set, _train_set_remainder = cat.utils.split_classes(
        train_set, p_test_classes=test_class_frac, verbose=True
    )
    test_classes = list(set(_train_set_remainder["category"]))

    return reduced_train_set, test_set, all_classes, test_classes


def prepare_dataloaders(test_class_frac, batch_size, model_name, dataset_frac=1.0, num_workers=8):
    """Loads dataset with zero-shot classes, creates collators and dataloaders

    Args:
        test_class_frac: 0 < float < 1, a proportion of classes that are made zero-shot
        dataset_frac: 0 < float < 1, a fraction of dataset to use
        batch_size: batch size for the dataloadres
        model_name: str, used as AutoTokenizer.from_pretrained(model_name)
        num_workers: number of workers in each dataloader

    Returns:
        tuple (train_dataloader, test_dataloader, all_classes_str, test_classes_str)
        where
            train_dataloader is a dataloader with CatCollator
            test_dataloader is a dataloader with CatTestCollator
            all_classes_str is a full list of class string representations (e.g., ["science", "sport", "politics"]
            test_classes_str has the same format as all_classes_str, but only contains zero-shot classes
    """
    (
        reduced_train_set,
        test_set,
        all_classes_str,
        test_classes_str,
    ) = cat.training_utils.prepare_dataset(test_class_frac, dataset_frac)

    text_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, fast=True)
    label_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, fast=True)

    # Datasets
    reduced_train_dataset = cat.CatDataset(
        reduced_train_set["headline"],
        text_tokenizer,
        reduced_train_set["category"],
        label_tokenizer,
    )
    test_dataset = cat.CatDataset(
        reduced_train_set["headline"],
        text_tokenizer,
        reduced_train_set["category"],
        label_tokenizer,
    )

    # Dataloaders
    all_classes_ids = label_tokenizer.batch_encode_plus(
        all_classes_str,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )["input_ids"]

    train_collator = cat.CatCollator(pad_token_id=label_tokenizer.pad_token_id)
    test_collator = cat.CatTestCollator(
        possible_labels_ids=all_classes_ids, pad_token_id=label_tokenizer.pad_token_id
    )

    train_dataloader = torch.utils.data.DataLoader(
        reduced_train_dataset,
        batch_size=batch_size,
        collate_fn=train_collator,
        num_workers=num_workers,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_collator,
        num_workers=num_workers,
    )
    return train_dataloader, test_dataloader, all_classes_str, test_classes_str
