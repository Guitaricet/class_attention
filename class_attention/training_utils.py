"""
Functions used in the training script
"""
import logging
import os
import sys
from random import random
from functools import partial

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
    class_field=None,
    test_set_name=None,
    build_class_sets=True,
    verbose=False,
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
        class_field: name of the class key in the dataset
        build_class_sets: compute a set of unique test classes and all classes, return them as lists,
            if False, return None instead

    Returns:
        train_dataset, test_dataset, list_all_classes, list_test_classes, (optionaly zero_shot_examples_set)
    """
    if class_field is None:
        raise ValueError("class_field is required")
    if test_set_name is None:
        raise ValueError("test_set_name is required")

    if verbose:
        logger.info(f"Loading dataset {dataset_name_or_path} into memory")

    dataset_dict = cat.utils.get_dataset_by_name_or_path(dataset_name_or_path)

    if verbose:
        logger.info("The dataset is loaded")

    train_set = dataset_dict["train"]
    test_set = dataset_dict[test_set_name]

    if dataset_frac < 1:
        if verbose:
            logger.info("Sampling from the dataset")

        # sample from the train set and the test set
        train_set = cat.utils.sample_dataset(train_set, p=dataset_frac)
        test_set = cat.utils.sample_dataset(test_set, p=dataset_frac)

        if "wiki" in dataset_name_or_path:
            logger.warning(
                "Sampling from a Wikipedia dataset may take a long time (~10+ minutes). "
                "Consider pre-sampling and saving it as a different file to speed up the experiments."
            )

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

    if test_class_frac > 0.0:
        if verbose:
            logger.info("Creating test classes")

        train_set, _ = cat.utils.split_classes(
            train_set, class_field=class_field, p_test_classes=test_class_frac, verbose=True
        )

    all_classes, zero_shot_classes = [], []
    if build_class_sets:
        if verbose:
            logger.info("Building class sets")

        train_classes = set(train_set[class_field])
        test_classes = set(test_set[class_field])

        # do not move this code above the IF statement as it may change all_classes
        all_classes = train_classes | test_classes
        zero_shot_classes = test_classes.difference(train_classes)

        if len(zero_shot_classes) < 2:
            logger.warning(f"Less than two zero-shot classes in the split: {zero_shot_classes}")

    return train_set, test_set, list(all_classes), list(zero_shot_classes)


def prepare_dataloaders(
    dataset_name_or_path,
    batch_size,
    model_name,
    test_class_frac=0,
    dataset_frac=1.0,
    num_workers=8,
    text_field=None,
    class_field=None,
    test_set_name=None,
    test_dataset_name_or_path=None,
    test_text_field=None,
    test_class_field=None,
    verbose=False,
    max_text_length=512,
    faiss_index_path=None,
    index_field=None,
    percent_hard=None,
    experience_replay_dataset_name_or_path=None,
    replay_text_field=None,
    replay_class_field=None,
    percent_replay_data=None,
) -> (DataLoader, DataLoader, list, list, dict):
    """Loads dataset with zero-shot classes, creates collators and dataloaders

    Args:
        dataset_name_or_path: path to a Datasets file
        test_class_frac: 0 < float < 1, a proportion of classes that are made zero-shot
        dataset_frac: 0 < float < 1, a fraction of dataset to use
        batch_size: batch size for the dataloadres
        model_name: str, used as AutoTokenizer.from_pretrained(model_name)
        num_workers: number of workers in each dataloader
        test_dataset_name_or_path: load a different dataset to evaluate on ("validation" field is used)
        faiss_index_path: path to a faiss index for the dataset, use scripts/index_wikipedia.py to produce it
        index_field: dataset field name that is associated with the faiss index

    Returns:
        tuple (train_dataloader, test_dataloader, all_classes_str, test_classes_str)
        where
            train_dataloader is a dataloader with CatCollator
            test_dataloader is a dataloader with CatTestCollator
            all_classes_str is a full list of class string representations (e.g., ["science", "sport", "politics"]
            test_classes_str has the same format as all_classes_str, but only contains zero-shot classes
    """
    # input validation
    if experience_replay_dataset_name_or_path is not None:
        if replay_text_field is None:
            raise ValueError()
        if replay_class_field is None:
            raise ValueError()
        if percent_replay_data is None:
            raise ValueError()

    if faiss_index_path is not None and percent_hard is None:
        raise ValueError()

    if text_field is None or class_field is None:
        raise ValueError("text_field and class_field are required")

    if test_set_name is None:
        raise ValueError("test_set_name is required")

    if "wiki" in dataset_name_or_path and test_dataset_name_or_path is None:
        raise NotImplementedError()

    if faiss_index_path is not None and dataset_frac < 1.0:
        logger.warning("faiss index should not be used with dataset_frac < 1.0")

    test_text_field = test_text_field or text_field
    test_class_field = test_class_field or class_field

    # end of input validation

    if verbose:
        logger.info("Preparing training set")

    reduced_train_set, test_set, all_classes_str, test_classes_str = prepare_dataset(
        dataset_name_or_path=dataset_name_or_path,
        test_class_frac=test_class_frac,
        dataset_frac=dataset_frac,
        class_field=class_field,
        test_set_name=test_set_name,
        build_class_sets=bool(test_dataset_name_or_path is None),  # only
        verbose=True,
    )

    if test_dataset_name_or_path:
        ranking_test_set = test_set

        if verbose:
            logger.info("Preparing test set")

        _, test_set, all_classes_str, test_classes_str = prepare_dataset(
            dataset_name_or_path=test_dataset_name_or_path,
            test_class_frac=test_class_frac,
            dataset_frac=dataset_frac,
            class_field=test_class_field,
            test_set_name=test_set_name,
            build_class_sets=True,
            verbose=True,
        )

    text_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, fast=True)
    label_tokenizer = text_tokenizer  # legacy stuff

    # torch.Dataset objects
    if "ids" in text_field:  # means the dataset is preprocessed
        if verbose:
            logger.info(
                "The training dataset is preprocessed, creating PreprocessedCatDatasetWCropAug object"
            )

        if faiss_index_path is None:
            # regular case
            reduced_train_dataset = cat.PreprocessedCatDatasetWCropAug(
                dataset=reduced_train_set,
                text_field=text_field,
                class_field=class_field,
                tokenizer=text_tokenizer,
                max_text_len=max_text_length,
            )

        else:
            # hard negatives using FAISS case
            logger.info(
                "Using hard negatives sampling. One epoch will take much longer to compute."
            )
            reduced_train_set.load_faiss_index(index_name=index_field, file=faiss_index_path)

            reduced_train_dataset = cat.hard_negatives.HardNegativeDatasetWAug(
                dataset=reduced_train_set,
                batch_size=batch_size,
                text_ids_field=text_field,
                label_ids_field=class_field,
                tokenizer=text_tokenizer,
                index_field=index_field,
                collator=cat.CatCollator(pad_token_id=text_tokenizer.pad_token_id),
                max_text_len=max_text_length,
                percent_hard=percent_hard,
            )

        ranking_test_dataset = cat.PreprocessedCatDatasetWCropAug(
            dataset=ranking_test_set,
            text_field=text_field,
            class_field=class_field,
            tokenizer=text_tokenizer,
            max_text_len=max_text_length,
            no_augmentations=True,
        )

    else:
        if faiss_index_path is not None:
            raise NotImplementedError("FAISS indexing is only supported in preprocessed datasets")

        if verbose:
            logger.info("The training dataset is raw text, creating CatDataset object")

        reduced_train_dataset = cat.CatDataset(
            texts=reduced_train_set[text_field],
            text_tokenizer=text_tokenizer,
            labels=reduced_train_set[class_field],
            label_tokenizer=label_tokenizer,
            max_text_len=max_text_length,
        )

    if "ids" in test_text_field:
        raise NotImplementedError()

    if experience_replay_dataset_name_or_path is not None:
        logger.info("Loading experience replay dataset")
        # replace train_dataset with a new one that contains extra examples
        replay_dataset_dict = cat.utils.get_dataset_by_name_or_path(experience_replay_dataset_name_or_path)
        replay_dataset_str = replay_dataset_dict["train"]

        # TODO: cleanup dirty code
        if "ids" not in replay_text_field or "ids" not in replay_class_field:
            raise ValueError("replay dataset should be numericalized and its fields should have ids in their names")

        replay_dataset = cat.PreprocessedCatDatasetWCropAug(
            dataset=replay_dataset_str,
            text_field=replay_text_field,
            class_field=replay_class_field,
            tokenizer=text_tokenizer,
            max_text_len=max_text_length,
        )

        reduced_train_dataset = cat.dataset.SampleConcatSubset(
            concat_dataset=reduced_train_dataset,
            sample_dataset=replay_dataset,
            sample_probability=percent_replay_data,
        )

    if verbose:
        logger.info("Creating CatDataset object for the test set")

    test_dataset = cat.CatDataset(
        texts=test_set[test_text_field],
        text_tokenizer=text_tokenizer,
        labels=test_set[test_class_field],
        label_tokenizer=label_tokenizer,
        max_text_len=max_text_length,
    )

    # Dataloaders
    if len(all_classes_str) > 100:
        logger.warning(10 * "More than 100 test classes. Continuing this run is not recommended\n")

    if verbose:
        logger.info("Encoding test classes")

    all_classes_ids = label_tokenizer.batch_encode_plus(
        all_classes_str,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )["input_ids"]

    if verbose:
        logger.info("Building dataloaders")

    train_collator = cat.CatCollator(
        pad_token_id=label_tokenizer.pad_token_id,
    )
    test_collator = cat.CatTestCollator(
        possible_labels_ids=all_classes_ids, pad_token_id=label_tokenizer.pad_token_id
    )

    if faiss_index_path is None:
        train_dataloader = DataLoader(
            reduced_train_dataset,
            batch_size=batch_size,
            collate_fn=train_collator,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
        )
    else:
        assert index_field is not None

        train_dataloader = DataLoader(
            reduced_train_dataset,
            batch_size=None,
            batch_sampler=None,
            num_workers=0,  # TODO: can we increase this?
            collate_fn=lambda x: x[0],  # DataLoader it creates a list of one element
        )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=test_collator,
        num_workers=num_workers,
        shuffle=False,
    )

    ranking_test_dataloader = None
    if test_dataset_name_or_path:
        # we use train collator here, because CatTest is classification-oriented
        # and requires to know all possible classes in advance
        # which is not realistic in ranking setting
        ranking_test_dataloader = DataLoader(
            ranking_test_dataset,
            batch_size=batch_size,
            collate_fn=train_collator,
            num_workers=num_workers,
            shuffle=False,
        )

    data = {"train": reduced_train_set, "test": test_set}

    return (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
        ranking_test_dataloader,
    )


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
    dataloader = DataLoader(
        words_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True
    )
    return dataloader


def train_cat_model(
    model: cat.ClassAttentionModel,
    model_optimizer,
    train_dataloader,
    test_dataloader,
    all_classes_str,
    test_classes_str,
    max_epochs,
    accelerator,
    predict_into_file=None,
    early_stopping=None,
    early_stopping_metric="eval/F1_macro",
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
    ranking_test_dataloader=None,
):
    _check_discriminator_triplet(discriminator, discriminator_optimizer, discriminator_update_freq)

    if discriminator_update_freq is None:
        discriminator_update_freq = 1  # only to safely use in with the %-operation

    patience = 0
    monitor_name = early_stopping_metric
    best_monitor = 0
    train_classes_str = sorted(set(all_classes_str).difference(set(test_classes_str)))

    if save_path is None and early_stopping is not None:
        logger.warning(
            "No save path is provided, early stopping will not load the best model after training"
        )

    device = accelerator.device

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
        labels_str=all_classes_str,
        zeroshot_labels=test_classes_str,
        ranking_dataloader=ranking_test_dataloader,
    )
    metrics["global_step"] = global_step

    if wandb.run is not None:
        wandb.log(metrics)

    should_stop_early = False

    maybe_progress_bar = lambda x: x
    if len(train_dataloader) > 10_000:
        maybe_progress_bar = partial(tqdm, total=len(train_dataloader))

    for epoch in tqdm(range(max_epochs), desc="Epochs"):
        if should_stop_early:
            break

        for x, c, y in maybe_progress_bar(train_dataloader):
            if global_step == -1 and wandb.run is not None:
                # fmt: off
                batch_html = cat.utils.batch_to_html(
                    text_ids=x,
                    label_ids=c,
                    targets=y,
                    text_tokenizer=train_dataloader.dataset.text_tokenizer,
                    label_tokenizer=train_dataloader.dataset.label_tokenizer,
                )
                logger.info("The first batch text is uploaded to wandb under `first_batch` key, look it up")
                wandb.log({"first_batch": wandb.Html(batch_html)})
                # fmt: on

            if c.shape[0] < 2:
                logger.warning(f"Number of possible classes is {c.shape[0]}, skipping this batch")
                continue

            global_step += 1
            model_optimizer.zero_grad()

            verify_shapes(x, y, c)

            model_loss, h_x, h_c, training_step_metrics = get_model_loss(
                model=model,
                x=x,
                c=c,
                y=y,
                loss_fn=loss_fn,
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

            accelerator.backward(model_loss)
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

                accelerator.backward(discriminator_loss)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
                discriminator_optimizer.step()

            is_eval_step = eval_every_steps is not None and global_step % eval_every_steps == 0
            if is_eval_step or global_step < 5:
                metrics = cat.evaluation_utils.evaluate_model(
                    model=model,
                    dataloader=test_dataloader,
                    labels_str=all_classes_str,
                    zeroshot_labels=test_classes_str,
                    ranking_dataloader=ranking_test_dataloader,
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

                if global_step > 5 and early_stopping is not None:
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
                            should_stop_early = True
                            break  # break the inner loop, the outer will be stopped via should_stop_early flag

        # validation
        metrics = cat.evaluation_utils.evaluate_model(
            model=model,
            dataloader=test_dataloader,
            labels_str=all_classes_str,
            zeroshot_labels=test_classes_str,
            predict_into_file=predict_into_file if (epoch == max_epochs - 1) else None,
            ranking_dataloader=ranking_test_dataloader,
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
                    break  # break the outer loop, no need to set should_stop_early=True

    # Training loop ends

    if early_stopping is not None and save_path is not None:
        logger.info(f"Loading the best model, expecting {monitor_name} to be {best_monitor}")
        model.load_state_dict_from_checkpoint(save_path)

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
        logger.warning(
            "Only one class is returned two times in a row, falling back to the original h_c"
        )
        return h_c

    c = c.to(x.device)

    _, _, h_c = model(
        text_input=x, labels_input=c, return_embeddings=True
    )  # [batch_size, n_classes]

    return h_c


def validate_dataloader(dataloader: DataLoader, test_classes, is_test=False):
    # fmt: off
    if is_test:
        assert isinstance(dataloader.sampler, torch.utils.data.sampler.SequentialSampler)
    else:
        assert isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler) \
            or isinstance(dataloader.dataset, cat.hard_negatives.HardNegativeDatasetWAug)
    # fmt: on

    if isinstance(dataloader.dataset, cat.CatDataset):
        dataset: cat.CatDataset = dataloader.dataset

        if is_test:
            assert set(dataset.labels).issuperset(set(test_classes))
        else:
            assert set(dataset.labels).isdisjoint(set(test_classes))


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
    device = logits.device

    # maximize the entropy of class distribution for the examples without the true label
    # compute cross entropy on the rest
    has_label_mask = y != -1  # [batch_size,]

    total_loss = torch.tensor(0, dtype=logits.dtype, device=device)

    ce_loss = None
    acc = None
    if torch.sum(has_label_mask) > torch.tensor(1, device=device):
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


def get_discriminator_loss_from_h(
    discriminator, h_x, h_c, invert_targets=False, wasserstein_loss=False
):
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
    dtype = h_x.dtype

    targets = torch.cat(
        [
            torch.zeros(n_texts, dtype=dtype, device=device),
            torch.ones(n_classes, dtype=dtype, device=device),
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
