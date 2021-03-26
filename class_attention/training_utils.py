"""
Functions used in the training script
"""

import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def prepare_dataset(
    dataset_name_or_path,
    test_class_frac=0.0,
    dataset_frac=1.0,
    return_zero_shot_examples=False,
    class_field=None,
    test_set_name=None,
):
    """

    Args:
        dataset_name_or_path: name for the Datasets.load_dataset or path for the Datasets.load_from_disk
            we expect the dataset to be of the form
            "train":
                class_field: [],
                ...
             "validation":
                class_field: [],
                ...

        test_class_frac:
        dataset_frac:
        return_zero_shot_examples: if yes, returns an extra dataset containing zero shot examples,
            NOTE: returns none instead of a dataset if the original dataset does not contain this key
        class_field: name of the class key in the dataset

    Returns:

    """
    if class_field is None:
        raise ValueError("class_field is required")
    if test_set_name is None:
        raise ValueError("test_set_name is required")

    news_dataset = cat.utils.get_dataset_by_name_or_path(dataset_name_or_path)
    train_set = news_dataset["train"]
    test_set = news_dataset[test_set_name]

    if dataset_frac < 1:
        # sample from the train set and the test set
        train_set = cat.utils.sample_dataset(train_set, p=dataset_frac)
        test_set = cat.utils.sample_dataset(test_set, p=dataset_frac)

        _train_classes = set(train_set[class_field])
        _test_classes = set(test_set[class_field])

        # check that there are still multiple training classes left
        if len(_train_classes) < 2:
            raise ValueError(
                f"Sampling size is too small for the *training* set, only {len(_train_classes)} left"
            )

        if len(_test_classes) < 2:
            raise ValueError(
                f"Sampling size is too small for the *test* set, only {len(_test_classes)} left"
            )

    zero_shot_examples_set = None
    if test_class_frac > 0.0:
        train_set, zero_shot_examples_set = cat.utils.split_classes(
            train_set, class_field=class_field, p_test_classes=test_class_frac, verbose=True
        )

    if return_zero_shot_examples and test_class_frac == 0:
        zero_shot_examples_set = news_dataset.get("zero_shot_examples", None)

    train_classes = set(train_set[class_field])
    test_classes = set(test_set[class_field])

    # do not move this code above the IF statement as it may change all_classes
    all_classes = train_classes | test_classes
    zero_shot_classes = test_classes.difference(train_classes)

    if len(zero_shot_classes) < 2:
        logger.warning(f"Less than two zero-shot classes in the split: {zero_shot_classes}")

    if return_zero_shot_examples:
        return (
            train_set,
            test_set,
            list(all_classes),
            list(zero_shot_classes),
            zero_shot_examples_set,
        )

    return train_set, test_set, list(all_classes), list(zero_shot_classes)


def prepare_dataloaders(
    dataset_name_or_path,
    test_class_frac,
    batch_size,
    model_name,
    dataset_frac=1.0,
    num_workers=8,
    glove_path=None,
    p_extra_classes=0,
    return_zero_shot_examples=False,
    text_field=None,
    class_field=None,
    test_set_name=None,
    p_no_class=0,
) -> (DataLoader, DataLoader, list, list, dict):
    """Loads dataset with zero-shot classes, creates collators and dataloaders

    Args:
        dataset_name_or_path: path to a Datasets file
        test_class_frac: 0 < float < 1, a proportion of classes that are made zero-shot
        dataset_frac: 0 < float < 1, a fraction of dataset to use
        batch_size: batch size for the dataloadres
        model_name: str, used as AutoTokenizer.from_pretrained(model_name)
        num_workers: number of workers in each dataloader
        glove_path: path to a GloVe file, these embeddings will be used as a label tokenizer
        return_zero_shot_examples: if True, returns an unlabeled dataset with examples of zero-shot classes

    Returns:
        tuple (train_dataloader, test_dataloader, all_classes_str, test_classes_str)
        where
            train_dataloader is a dataloader with CatCollator
            test_dataloader is a dataloader with CatTestCollator
            all_classes_str is a full list of class string representations (e.g., ["science", "sport", "politics"]
            test_classes_str has the same format as all_classes_str, but only contains zero-shot classes
    """
    if text_field is None or class_field is None:
        raise ValueError("text_field and class_field are required")

    if test_set_name is None:
        raise ValueError("test_set_name is required")

    (
        reduced_train_set,
        test_set,
        all_classes_str,
        test_classes_str,
        zero_shot_examples,
    ) = prepare_dataset(
        dataset_name_or_path=dataset_name_or_path,
        test_class_frac=test_class_frac,
        dataset_frac=dataset_frac,
        return_zero_shot_examples=True,
        class_field=class_field,
        test_set_name=test_set_name,
    )

    text_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, fast=True)

    if glove_path is not None:
        _, word2id = cat.utils.load_glove_from_file(glove_path)
        label_tokenizer = cat.utils.GloVeTokenizer(word2id)
    else:
        label_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, fast=True)

    # Datasets
    reduced_train_dataset = cat.CatDataset(
        texts=reduced_train_set[text_field],
        text_tokenizer=text_tokenizer,
        labels=reduced_train_set[class_field],
        label_tokenizer=label_tokenizer,
    )
    test_dataset = cat.CatDataset(
        texts=test_set[text_field],
        text_tokenizer=text_tokenizer,
        labels=test_set[class_field],
        label_tokenizer=label_tokenizer,
    )

    # Dataloaders
    all_classes_ids = label_tokenizer.batch_encode_plus(
        all_classes_str,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )["input_ids"]

    train_collator = cat.CatCollator(
        pad_token_id=label_tokenizer.pad_token_id,
        possible_label_ids=all_classes_ids,
        p_extra_classes=p_extra_classes,
        p_no_class=p_no_class,
    )
    test_collator = cat.CatTestCollator(
        possible_labels_ids=all_classes_ids, pad_token_id=label_tokenizer.pad_token_id
    )

    train_dataloader = DataLoader(
        reduced_train_dataset,
        batch_size=batch_size,
        collate_fn=train_collator,
        num_workers=num_workers,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_collator,
        num_workers=num_workers,
        shuffle=False,
    )

    data = {"train": reduced_train_set, "test": test_set}

    # TODO: figure out a better way to solve this
    # or maybe just remove this feature
    if return_zero_shot_examples:
        if zero_shot_examples is None:
            return (
                train_dataloader,
                test_dataloader,
                all_classes_str,
                test_classes_str,
                data,
                None,
            )

        zero_shot_examples_dataset = cat.CatDataset(
            zero_shot_examples[class_field],
            text_tokenizer,
        )
        zero_shot_examples_dataloader = DataLoader(
            zero_shot_examples_dataset,
            batch_size=batch_size,
            collate_fn=train_collator,
            num_workers=num_workers,
            shuffle=True,
        )

        return (
            train_dataloader,
            test_dataloader,
            all_classes_str,
            test_classes_str,
            data,
            zero_shot_examples_dataloader,
        )

    return train_dataloader, test_dataloader, all_classes_str, test_classes_str, data


def make_extra_classes_dataloader_from_glove(
    glove_path,
    batch_size,
    class_names=None,
):
    _, word2id = cat.utils.load_glove_from_file(glove_path)
    label_tokenizer = cat.utils.GloVeTokenizer(word2id)
    if class_names is None:
        words = cat.utils.filter_words(word2id.keys(), extra_filter=lambda x: x != "[PAD]")
        words_dataset = cat.CatDataset(words, label_tokenizer)
    else:
        # NOTE: some of the class names can have length > 1
        # DataLoader won't be able to collate them with the default collator
        class_names = [c.lower() for c in class_names]
        words_dataset = cat.CatDataset(class_names, label_tokenizer)

    dataloader = DataLoader(words_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_cat_model(
    model: cat.ClassAttentionModel,
    optimizer,
    train_dataloader,
    test_dataloader,
    all_classes_str,
    test_classes_str,
    max_epochs,
    device=None,
    predict_into_file=None,
    early_stopping=None,
    save_path=None,
    extra_examples_dataloader=None,
    examples_entropy_reg=None,
    extra_classes_dataloader=None,
    classes_entropy_reg=None,
    eval_every_steps=None,
    label_smoothing=None,
):
    patience = 0
    monitor_name = "eval/F1_macro"
    best_monitor = 0
    train_classes_str = sorted(set(all_classes_str).difference(set(test_classes_str)))

    if save_path is None and early_stopping is not None:
        logger.warning(
            "No save path is provided, early stopping will not load the best model after training"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if extra_examples_dataloader is not None:
        assert examples_entropy_reg is not None
        extra_examples_dataloader = cat.utils.infinite_iterator(extra_examples_dataloader)

    if extra_classes_dataloader is not None:
        assert classes_entropy_reg is not None
        extra_classes_dataloader = cat.utils.infinite_iterator(extra_classes_dataloader)

    if label_smoothing is not None and label_smoothing > 0:
        loss_fn = cat.loss.LabelSmoothingLoss(label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss()

    global_step = -1

    metrics = cat.evaluation_utils.evaluate_model(
        model,
        test_dataloader,
        device=device,
        labels_str=all_classes_str,
        zeroshot_labels=test_classes_str,
    )
    metrics["global_step"] = global_step

    if wandb.run is not None:
        wandb.log(metrics)

    for epoch in tqdm(range(max_epochs), desc="Epochs"):
        for x, c, y in train_dataloader:
            if c.shape[0] < 2:
                logger.warning(f"Number of possible classes is {c.shape[0]}, skipping this batch")
                continue

            global_step += 1
            optimizer.zero_grad()

            x = x.to(device)
            c = c.to(device)
            y = y.to(device)
            verify_shapes(x, y, c)

            # --- Loss computation STARTS here ---

            logits, h_c = model(
                text_input=x, labels_input=c, return_class_embeddings=True
            )  # [batch_size, n_classes]

            # maximize the entropy of class distribution for the examples without the true label
            # compute cross entropy on the rest
            has_label_mask = y != -1  # [batch_size,]

            total_loss = torch.tensor(0, dtype=torch.float32, device=device)

            ce_loss = None
            acc = None
            if torch.sum(has_label_mask) > torch.tensor(1, device=has_label_mask.device):
                # only use examples with the true label to compute cross-entropy loss
                ce_logits = logits[has_label_mask]
                y = y[has_label_mask]

                if not (torch.all(0 <= y) and torch.all(y < ce_logits.size(1))):
                    import ipdb; ipdb.set_trace()

                _ce_loss = loss_fn(ce_logits, y)
                total_loss += _ce_loss

                ce_loss = _ce_loss.detach().clone()  # used for logging

                _, preds = ce_logits.max(-1)
                acc = torch.sum(preds == y).float() / x.shape[0]

            no_label_entropy = None
            if not torch.all(has_label_mask):
                no_label_logits = logits[~has_label_mask]
                _no_label_entropy = get_entropy(no_label_logits)
                # minus sign, because we want to maximize the entropy
                total_loss -= _no_label_entropy

                no_label_entropy = _no_label_entropy.detach().clone()

            # if args.double_loss:
            # similar to CLIP cross_entropy_loss(..., axis=0)
            # TODO: average text vectors with the same class
            # then compute the transposed cross entropy

            # Regularization
            extra_wandb_logs = dict()
            if extra_examples_dataloader is not None:
                neg_entropy = get_extra_examples_neg_entropy(
                    extra_examples_dataloader, model, device
                )

                total_loss += examples_entropy_reg * neg_entropy
                extra_wandb_logs["train/extra_examples_entropy"] = -neg_entropy

            if extra_classes_dataloader:
                neg_entropy = get_extra_classes_neg_entropy(
                    x, extra_classes_dataloader, model, device
                )
                total_loss += classes_entropy_reg * neg_entropy
                extra_wandb_logs["train/extra_classes_entropy"] = -neg_entropy

            # --- Loss computation ENDS here ---

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/acc": acc,
                        "train/loss": total_loss,
                        "train/cross_entropy": ce_loss,
                        "train/no_label_entropy": no_label_entropy,
                        "train/epoch": epoch,
                        "global_step": global_step,
                        **extra_wandb_logs,
                    }
                )

            total_loss.backward()
            optimizer.step()

            if (eval_every_steps is not None and global_step % eval_every_steps == 0) or (global_step < 5):
                metrics = cat.evaluation_utils.evaluate_model(
                    model,
                    test_dataloader,
                    device=device,
                    labels_str=all_classes_str,
                    zeroshot_labels=test_classes_str,
                    predict_into_file=predict_into_file if (epoch == max_epochs - 1) else None,
                )
                extra_metrics = get_extra_metrics(
                    model,
                    train_classes_str,
                    test_classes_str,
                    train_dataloader.dataset.label_tokenizer,
                    device,
                )
                metrics = {**metrics, **extra_metrics}

                if wandb.run is not None:
                    wandb.log(metrics)

        # validation
        metrics = cat.evaluation_utils.evaluate_model(
            model,
            test_dataloader,
            device=device,
            labels_str=all_classes_str,
            zeroshot_labels=test_classes_str,
            predict_into_file=predict_into_file if (epoch == max_epochs - 1) else None,
        )
        extra_metrics = get_extra_metrics(
            model,
            train_classes_str,
            test_classes_str,
            train_dataloader.dataset.label_tokenizer,
            device,
        )
        metrics = {**metrics, **extra_metrics}

        if wandb.run is not None:
            wandb.log(metrics)

        # Early stopping
        if early_stopping is not None:
            monitor = metrics[monitor_name]
            if monitor > best_monitor:
                patience = 0
                best_monitor = monitor
                if save_path is not None:
                    model.save(
                        file_path=save_path,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        train_classes_str=train_classes_str,
                        test_classes_str=test_classes_str,
                    )
            else:
                patience += 1
                if patience > early_stopping:
                    logger.info(
                        f"The target metric did not improve over {patience} iterations. Stopping early."
                    )
                    break

    if early_stopping is not None and save_path is not None:
        logger.info(f"Loading the best model, expecting {monitor_name} to be {best_monitor}")
        state_dict = torch.load(save_path)["model_state_dict"]
        model.load_state_dict(state_dict)

    return model


def validate_dataloader(dataloader: DataLoader, test_classes, is_test=False):
    if is_test:
        assert isinstance(dataloader.sampler, torch.utils.data.sampler.SequentialSampler)
    else:
        assert isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler)

    dataset: cat.CatDataset = dataloader.dataset

    if is_test:
        assert set(dataset.labels).issuperset(set(test_classes))
    else:
        assert set(dataset.labels).isdisjoint(set(test_classes))


def make_label_encoder(model_name_or_path, glove=None):
    if glove is not None:
        emb_matrix, word2id = cat.utils.load_glove_from_file(glove)
        return cat.modelling.PreTrainedEmbeddingEncoder(emb_matrix, word2id)

    return transformers.AutoModel.from_pretrained(model_name_or_path)


def get_extra_examples_neg_entropy(extra_examples_dataloader, model, device):
    extra_x, all_c = next(extra_examples_dataloader)

    extra_x = extra_x.to(device)
    all_c = all_c.to(device)
    examples_batch_size = extra_x.shape[0]

    logits = model(extra_x, all_c)

    neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    neg_entropy = torch.sum(neg_entropy) / examples_batch_size

    return neg_entropy


def get_extra_classes_neg_entropy(x, extra_classes_dataloader, model, device):
    # NOTE: can we concat extra classes to the original `c`
    # so you don't need to forward the model one more time?
    #
    # Use real data, but extra classes.
    # Question: do we really need to sample real data?
    # Answer: yes, because:
    #     1. Look at the math
    #     2. Intuition of this regularization is that
    #     the model should not prefer any class for the example from a set of wrong classes

    extra_c = next(extra_classes_dataloader)
    if extra_c.shape[0] == 1:
        logger.warning(
            "Number of possible classes is 1, sampling from extra_classes_dataloader again"
        )
        # TODO: awful hack, what if the sampling is bad again?
        extra_c = next(extra_classes_dataloader)

    extra_c = extra_c.to(device)
    n_classes = extra_c.shape[0]

    logits = model(x, extra_c)

    neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    neg_entropy = torch.sum(neg_entropy) / n_classes

    return neg_entropy


def get_entropy(logits, normalize_by_dim=0):
    norm = logits.size(normalize_by_dim)
    entropy = -F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    return torch.sum(entropy) / norm


def get_extra_metrics(model, train_classes_str, test_classes_str, label_tokenizer, device):
    # central class embedding distance
    cced = cat.utils.get_cced(
        model,
        train_classes_str,
        test_classes_str,
        label_tokenizer,
        device,
    )

    # root mean absolute scalar product
    rmasp = cat.utils.get_rmasp(
        model,
        train_classes_str,
        test_classes_str,
        label_tokenizer,
        device,
    )

    res = {"cced": cced, "rmasp": rmasp}

    return res


def verify_shapes(x, y, c):
    batch_size_1, text_seq_len = x.shape
    n_classes, class_seq_len = c.shape
    batch_size_2, = y.shape

    assert n_classes > 0

    assert batch_size_1 == batch_size_2
    assert batch_size_1 > 0
