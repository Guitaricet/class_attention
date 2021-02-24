"""
Functions used in the training script
"""

import logging
import os
import sys

import torch
import torch.nn.functional as F

import datasets
import transformers
import wandb
from tqdm.auto import tqdm

import class_attention as cat


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


def prepare_dataset(dataset_name_or_path, test_class_frac, dataset_frac=1.0):
    news_dataset = get_dataset_by_name_or_path(dataset_name_or_path)
    train_set = news_dataset["train"]
    test_set = news_dataset["validation"]

    if dataset_frac < 1:
        # sample from the train set and the test set
        train_set = cat.utils.sample_dataset(train_set, p=dataset_frac)
        test_set = cat.utils.sample_dataset(test_set, p=dataset_frac)

        _train_classes = set(train_set["category"])
        _test_classes = set(test_set["category"])

        # check that there are still multiple training classes left
        if len(_train_classes) < 2:
            raise ValueError(
                f"Sampling size is too small for the *training* set, only {len(_train_classes)} left"
            )

        if len(_test_classes) < 2:
            raise ValueError(
                f"Sampling size is too small for the *test* set, only {len(_test_classes)} left"
            )

    if test_class_frac > 0.0:
        train_set, _ = cat.utils.split_classes(
            train_set, p_test_classes=test_class_frac, verbose=True
        )

    train_classes = set(train_set["category"])
    test_classes = set(test_set["category"])

    # do not move this code above the IF statement as it may change all_classes
    all_classes = train_classes | test_classes
    zero_shot_classes = test_classes.difference(train_classes)

    if len(zero_shot_classes) < 2:
        logger.warning(f"Less than two zero-shot classes in the split: {zero_shot_classes}")

    return train_set, test_set, list(all_classes), list(zero_shot_classes)


def prepare_dataloaders(
    dataset_name_or_path, test_class_frac, batch_size, model_name, dataset_frac=1.0, num_workers=8
):
    """Loads dataset with zero-shot classes, creates collators and dataloaders

    Args:
        dataset_name_or_path: path to a Datasets file
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
    ) = prepare_dataset(dataset_name_or_path, test_class_frac, dataset_frac)

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


def train_cat_model(
    model,
    optimizer,
    train_dataloader,
    test_dataloader,
    all_classes_str,
    test_classes_str,
    max_epochs,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    global_step = -1

    for epoch in tqdm(range(max_epochs), desc="Epochs"):
        for x, c, y in train_dataloader:
            if c.shape[0] == 1:
                logger.warning("Number of possible classes is 1, skipping this batch")
                continue

            global_step += 1
            optimizer.zero_grad()

            x = x.to(device)
            c = c.to(device)
            y = y.to(device)

            x_dict = {"input_ids": x}
            c_dict = {"input_ids": c}
            logits = model(x_dict, c_dict)  # [batch_size, n_classes]

            loss = F.cross_entropy(logits, y)
            # if args.double_loss:
            # similar to CLIP cross_entropy_loss(..., axis=0)
            # TODO: average text vectors with the same class
            # then compute the transposed cross entropy

            _, preds = logits.max(-1)
            acc = torch.sum(preds == y).float() / x.shape[0]

            # fmt: off
            if wandb.run is not None:
                wandb.log({
                    "train/acc": acc,
                    "train/loss": loss,
                    "train/epoch": epoch,
                    "global_step": global_step,
                })
            # fmt: on

            loss.backward()
            optimizer.step()

        # validation
        metrics = cat.utils.evaluate_model_per_class(
            model,
            test_dataloader,
            device=device,
            labels_str=all_classes_str,
            zeroshot_labels=test_classes_str,
        )
        if wandb.run is not None:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

    return model


def get_dataset_by_name_or_path(name_or_path):
    try:
        dataset = datasets.load_from_disk(name_or_path)
    except FileNotFoundError:
        try:
            dataset = datasets.load_dataset(name_or_path)
        except FileNotFoundError:
            raise ValueError(f"The dataset {name_or_path} wasn't found locally or downloaded")

    return dataset
