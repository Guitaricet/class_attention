"""
Functions used in the training script
"""
import logging
import os
import sys
from random import random

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


def make_extra_classes_dataloader_from_file(
    file_path,
    tokenizer,
    batch_size,
):
    """Creates a dataloader that will

    Args:
        file_path: path to a text file containing class name on every line
        tokenizer: transformers.tokenizer
        batch_size: int

    Returns:
        torch.DataLoader(cat.CatDataset, batch_size=batch_size)
    """
    with open(file_path) as f:
        class_names = f.read().splitlines()

    words_dataset = cat.CatDataset(class_names, tokenizer)
    collator = cat.CatCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(words_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True)
    return dataloader


def train_cat_model(
    model: cat.ClassAttentionModel,
    model_optimizer,
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
    eval_every_steps=None,
    label_smoothing=None,
    discriminator=None,
    discriminator_optimizer=None,
    discriminator_update_freq=None,
    class_cos2_reg=None,
    adv_reg_weight=1.0,
    use_wasserstein_loss=False,
):
    _check_discriminator_triplet(discriminator, discriminator_optimizer, discriminator_update_freq)

    if discriminator_update_freq is None:
        discriminator_update_freq = 1  # only to safely use in with the %-operation

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
            model_optimizer.zero_grad()

            x = x.to(device)
            c = c.to(device)
            y = y.to(device)
            verify_shapes(x, y, c)

            model_loss, h_x, h_c, training_step_metrics = get_model_loss(
                model=model,
                x=x,
                c=c,
                y=y,
                loss_fn=loss_fn,
                device=device,
                global_step=global_step,
                epoch=epoch,
                extra_examples_dataloader=extra_examples_dataloader,
                examples_entropy_reg=examples_entropy_reg,
                class_vec_cos_reg=class_cos2_reg,
            )

            is_discriminator_update_step = global_step % discriminator_update_freq == 0

            if discriminator is not None and not is_discriminator_update_step:
                # use adversarial loss to update the network
                _h_c = maybe_compute_new_hc(x, h_c, model, extra_classes_dataloader)

                anti_discriminator_loss, _ = get_discriminator_loss_from_h(
                    discriminator,
                    h_x,
                    _h_c,
                    invert_targets=True,
                    wasserstein_loss=use_wasserstein_loss,
                )
                model_loss += adv_reg_weight * anti_discriminator_loss
                training_step_metrics["train/loss"] = model_loss  # overrides the existing value
                training_step_metrics["train/adversarial_reg"] = anti_discriminator_loss

            if wandb.run is not None:
                wandb.log(training_step_metrics)

            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            model_optimizer.step()

            if discriminator is not None and is_discriminator_update_step:
                # train discriminator
                discriminator_optimizer.zero_grad()

                _h_c = maybe_compute_new_hc(x, h_c, model, extra_classes_dataloader)

                discriminator_loss, discriminator_metrics = get_discriminator_loss_from_h(
                    discriminator,
                    h_x.detach(),
                    _h_c.detach(),
                    wasserstein_loss=use_wasserstein_loss,
                )
                if wandb.run is not None:
                    wandb.log(discriminator_metrics)

                discriminator_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
                discriminator_optimizer.step()

            is_eval_step = eval_every_steps is not None and global_step % eval_every_steps == 0
            if is_eval_step or global_step < 5:
                metrics = cat.evaluation_utils.evaluate_model(
                    model=model,
                    dataloader=test_dataloader,
                    device=device,
                    labels_str=all_classes_str,
                    zeroshot_labels=test_classes_str,
                )
                extra_metrics = get_extra_metrics(
                    model=model,
                    train_classes_str=train_classes_str,
                    test_classes_str=test_classes_str,
                    label_tokenizer=train_dataloader.dataset.label_tokenizer,
                    device=device,
                )
                metrics = {**metrics, **extra_metrics}

                if wandb.run is not None:
                    wandb.log(metrics)

        # validation
        metrics = cat.evaluation_utils.evaluate_model(
            model=model,
            dataloader=test_dataloader,
            device=device,
            labels_str=all_classes_str,
            zeroshot_labels=test_classes_str,
            predict_into_file=predict_into_file if (epoch == max_epochs - 1) else None,
        )
        extra_metrics = get_extra_metrics(
            model=model,
            train_classes_str=train_classes_str,
            test_classes_str=test_classes_str,
            label_tokenizer=train_dataloader.dataset.label_tokenizer,
            device=device,
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
                        optimizer=model_optimizer,
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

    # Training loop ends

    if early_stopping is not None and save_path is not None:
        logger.info(f"Loading the best model, expecting {monitor_name} to be {best_monitor}")
        state_dict = torch.load(save_path)["model_state_dict"]
        model.load_state_dict(state_dict)

    return model


def maybe_compute_new_hc(x, h_c, model, extra_classes_dataloader, real_hc_prob=0.1):
    # NOTE: `x` is a dirty and lazy hack so we don't have to figure out some text input
    # (remember that the model requires both text and class input and we can't just forward classes
    # and just getting the h_c is not implemented ¯\_(ツ)_/¯)
    if extra_classes_dataloader is None or random() < real_hc_prob:
        return h_c

    x = x[0].unsqueeze(0)
    c, _ = next(extra_classes_dataloader)
    if c.shape[0] == 1:
        c, _ = next(extra_classes_dataloader)

    if c.shape[0] == 1:
        logger.warning("Only one class is returned two times in a row, falling back to the original h_c")
        return h_c

    c = c.to(x.device)

    _, _, h_c = model(
        text_input=x, labels_input=c, return_embeddings=True
    )  # [batch_size, n_classes]

    return h_c


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
    (batch_size_2,) = y.shape

    assert n_classes > 0

    assert batch_size_1 == batch_size_2
    assert batch_size_1 > 0


def get_model_loss(
    model: cat.ClassAttentionModel,
    x: torch.LongTensor,
    c: torch.LongTensor,
    y: torch.LongTensor,
    loss_fn,
    device,
    global_step,
    epoch,
    extra_examples_dataloader=None,
    examples_entropy_reg=None,
    class_vec_cos_reg=None,
):
    """
    Computes a single forward step for the model (not the discriminator).

    Args:
        model: cat.ClassAttentionModel
        x: LongTensor[batch_size, text_seq_len], text representation
        c: LongTensor[n_classes, class_seq_len], class representation
        y: LongTensor[batch_size,], targets
        loss_fn: function that accepts logits and targets and returns a scalar, e.g. cat.loss.LabelSmoothingLoss
        device: torch.device
        global_step: int
        epoch: int
        extra_examples_dataloader: a dataloader with extra examples without labels (used to maximize entropy on them)
        examples_entropy_reg: extra examples entropy regularization coefficient

    Returns:
        loss, metrics

        where
            loss: torch scalar
            metrics: dict(str -> scalar)
    """
    logits, h_x, h_c = model(
        text_input=x, labels_input=c, return_embeddings=True
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
            raise ValueError(y)

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

    # Entropy regularization
    extra_metrics = dict()
    if extra_examples_dataloader is not None:
        neg_entropy = get_extra_examples_neg_entropy(extra_examples_dataloader, model, device)

        total_loss += examples_entropy_reg * neg_entropy
        extra_metrics["train/extra_examples_entropy"] = -neg_entropy

    # Similarity regularization
    # minimizing the cosine similarity should make the vectors more orthogonal
    cos2 = cat.modelling_utils.cos2(h_c)
    extra_metrics["train/class_embed_similarity"] = cos2

    if class_vec_cos_reg is not None:
        total_loss += class_vec_cos_reg * cos2

    metrics = {
        "train/acc": acc,
        "train/loss": total_loss,
        "train/cross_entropy": ce_loss,
        "train/no_label_entropy": no_label_entropy,
        "train/epoch": epoch,
        "global_step": global_step,
        **extra_metrics,
    }

    return total_loss, h_x, h_c, metrics


def _check_discriminator_triplet(
    discriminator, discriminator_optimizer, discriminator_update_freq
):
    if (
        discriminator is not None
        or discriminator_optimizer is not None
        or discriminator_update_freq is not None
    ):
        if discriminator is None:
            raise ValueError("Provide discriminator")
        if discriminator_optimizer is None:
            raise ValueError("Provide discriminator optimizer")
        if discriminator_update_freq is None:
            raise ValueError("Provide discriminator_update_freq")

        if discriminator_update_freq < 2:
            raise ValueError(
                "Discriminator update freq should be above 1, because the model gets adversarial loss on the steps "
                "when we do not update the discriminator. If update freq = 1 then the model does not get adversarial "
                "loss at all (only the discriminator does)"
            )


def get_discriminator_loss_from_h(discriminator, h_x, h_c, invert_targets=False, wasserstein_loss=False):
    """Computes the loss to train the discriminator (invert_targets=False) or
    to train the adversary (invert_targets=True).

    Args:
        discriminator: a model that accepts [h_x:h_c] as input
        h_x: torch.FloatTensor[batch_size, hidden]
        h_c: torch.FloatTensor[n_classes, hidden]

    Returns:
        loss, metrics

        `metrics` is None if invert_targets=True, because they do not make sense in this case
    """
    n_texts = h_x.shape[0]
    n_classes = h_c.shape[0]
    device = h_x.device

    targets = torch.cat(
        [
            torch.zeros(n_texts, dtype=torch.float32, device=device),
            torch.ones(n_classes, dtype=torch.float32, device=device),
        ]
    )

    if invert_targets:
        targets = torch.ones_like(targets) - targets

    h_all = torch.cat([h_x, h_c], dim=0)

    logits = discriminator(h_all).squeeze(-1)  # [batch_size + n_classes,]

    if not wasserstein_loss:
        # usual GAN-like loss
        loss = F.binary_cross_entropy_with_logits(logits, targets)
    else:
        # Wasserstein GAN-like loss
        text_logits, class_logits = torch.split(logits, [n_texts, n_classes])
        # loss = (d_fake - d_real).mean()
        loss = text_logits.mean() - class_logits.mean()

    # preds are not entirely correct in case of
    preds = (torch.sigmoid(logits) > 0.5).long()
    acc = torch.sum(preds == targets).float() / preds.shape[0]

    metrics = {"discr/loss": loss, "discr/acc": acc}
    if invert_targets:
        metrics = None

    return loss, metrics
