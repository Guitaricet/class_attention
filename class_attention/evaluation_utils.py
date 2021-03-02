"""
Functions responsible for model evaluation
"""
import logging
import os
import sys

from typing import Union

import pandas as pd
import torch
import torch.utils.data

from tqdm.auto import tqdm

import class_attention as cat


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


def evaluate_model(
    model,
    dataloader,
    device,
    labels_str,
    zeroshot_labels=None,
    progress_bar=False,
    predict_into_file=None,
):
    """Makes predictions on dataloader, reports metrics.

    Args:
        model: ClassAttentionModel
        dataloader: pytorch DataLoader with CatTestCollator
        labels_str: List[str], names of classes, in the same order as in the CatTestCollator.possible_labels
        zeroshot_labels: if provided, additional metrics will be computed on this set of labels
    """
    if predict_into_file is not None and not isinstance(
        dataloader.sampler, torch.utils.data.sampler.SequentialSampler
    ):
        raise ValueError("test dataloader should not be shuffled")

    model = model.to(device)
    model.eval()

    if zeroshot_labels is not None and (not set(zeroshot_labels).issubset(labels_str)):
        raise ValueError("labels_str should include all labels")

    all_predictions = []
    all_labels = []
    all_texts = []
    text_tokenizer = dataloader.dataset.text_tokenizer

    if progress_bar:
        dataloader = tqdm(dataloader, desc="Evaluation")

    with torch.no_grad():
        for x, c, y in dataloader:
            # Note: `c` does not change in CatTestCollator
            x, c, y = x.to(device), c.to(device), y.to(device)

            logits = model(x, c)

            _, preds = logits.max(-1)

            predicted_labels = [labels_str[i] for i in preds]
            expected_labels = [labels_str[i] for i in y]

            all_predictions += predicted_labels
            all_labels += expected_labels
            if predict_into_file is not None:
                # NOTE: this may cause concurrency issues or semaphore failures if tokenizer parallelism is not disabled
                all_texts += text_tokenizer.batch_decode(x, skip_special_tokens=True)

    if predict_into_file is not None:
        assert (
            len(all_texts) == len(all_predictions) == len(expected_labels)
        ), f"{len(all_texts)} texts, {len(all_predictions)} preds, {len(all_labels)} labels"

        dataframe = pd.DataFrame(
            data=zip(all_texts, all_predictions, all_labels),
            columns=["text", "predicted label", "expected label"],
        )
        dataframe.to_csv(predict_into_file)
        logger.info(f"Saved predictions into {predict_into_file}")

    metrics, per_class_metrics = compute_metrics(y_true=all_labels, y_pred=all_predictions)
    res = {
        **{"eval/" + key: value for key, value in metrics.items()},
        **{"eval_per_class/" + key: value for key, value in per_class_metrics.items()},
    }

    if zeroshot_labels is not None:
        zeroshot_metrics, _ = compute_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            class_group=zeroshot_labels,
            prefix="zero_shot_eval/",
        )

        multishot_labels = set(labels_str).difference(set(zeroshot_labels))
        multishot_metrics, _ = compute_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            class_group=multishot_labels,
            prefix="multi_shot_eval/",
        )

        res.update(zeroshot_metrics)
        res.update(multishot_metrics)

    model.train()
    return res


def evaluate_model_on_subset(
    model,
    dataset_str,
    test_classes_str,
    text_tokenizer,
    label_tokenizer,
    predict_into_file=None,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset filtration happens in make_test_classes_only_dataloader
    test_dataloader = make_test_classes_only_dataloader(
        dataset=dataset_str,
        test_classes_str=test_classes_str,
        text_tokenizer=text_tokenizer,
        label_tokenizer=label_tokenizer,
    )

    subset_metrics = evaluate_model(
        model,
        test_dataloader,
        device=device,
        labels_str=test_classes_str,
        zeroshot_labels=None,  # it needs to be None, so that the metrics are not splitted in two subsets
        predict_into_file=predict_into_file,
    )

    return subset_metrics


def make_test_classes_only_dataloader(dataset, test_classes_str, text_tokenizer, label_tokenizer):
    """Filters dataset to only contain test_classes_str and makes a dataloader with CatTestCollator

    Args:
        dataset: ArrowDataset
        test_classes_str: list of test class names

    Returns:
        DataLoader with CatTestCollator
    """
    _, only_test_classes_data = cat.utils.split_classes(dataset, test_classes=test_classes_str)

    otc_dataset = cat.CatDataset(
        only_test_classes_data["headline"],
        text_tokenizer,
        only_test_classes_data["category"],
        label_tokenizer,
    )

    test_classes_ids = label_tokenizer.batch_encode_plus(
        test_classes_str,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )["input_ids"]

    otc_collator = cat.CatTestCollator(
        possible_labels_ids=test_classes_ids, pad_token_id=label_tokenizer.pad_token_id
    )

    otc_dataloader = torch.utils.data.DataLoader(
        otc_dataset, collate_fn=otc_collator, shuffle=False, pin_memory=True
    )
    return otc_dataloader


def compute_metrics(y_true, y_pred, class_group=None, prefix=None, suffix=None):
    """Computes accuracy and P, R, F1 for each class

    Args:
        y_true: list[str]
        y_pred: list[str]
        class_group: if specified, only these classes will be considered
        suffix: suffix for the dict keys

    Returns:
        dict metric_name:float
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true len {len(y_true)}, y_pred len {len(y_pred)}, but they should be equal"
        )

    prefix = _handle_prefix(prefix)
    suffix = _handle_suffix(suffix)

    # filter interesting classes if they are specified
    if class_group is not None:
        y_true, y_pred = _filter_classes(y_true=y_true, y_pred=y_pred, class_group=class_group)

    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)

    all_labels = set(y_true) | set(y_pred)
    label2n_correct = {l: 0 for l in all_labels}
    label2n_predicted = {l: 0 for l in all_labels}
    label2n_expected = {l: 0 for l in all_labels}

    for predicted, expected in zip(y_pred, y_true):
        if predicted == expected:
            label2n_correct[predicted] += 1

        label2n_predicted[predicted] += 1
        label2n_expected[expected] += 1

    assert set(label2n_correct.keys()) == all_labels
    assert set(label2n_expected.keys()) == all_labels
    assert set(label2n_predicted.keys()).issubset(all_labels)
    assert set(label2n_predicted.keys()) == all_labels

    label2_precision = dict()
    label2_recall = dict()
    label2_f1 = dict()

    for label in all_labels:
        p = label2n_correct[label] / (label2n_predicted[label] + 1e-7)
        r = label2n_correct[label] / (label2n_expected[label] + 1e-7)
        f1 = 2 * (p * r) / (p + r + 1e-7)

        label2_precision[label] = p
        label2_recall[label] = r
        label2_f1[label] = f1

    p_macro = sum(label2_precision.values()) / len(all_labels)
    r_macro = sum(label2_recall.values()) / len(all_labels)
    f1_macro = sum(label2_f1.values()) / len(all_labels)

    res = {
        "acc" + suffix: accuracy,
        "P_macro" + suffix: p_macro,
        "R_macro" + suffix: r_macro,
        "F1_macro" + suffix: f1_macro,
    }
    per_class_res = {
        **{label + "/P": metric for label, metric in label2_precision.items()},
        **{label + "/R": metric for label, metric in label2_recall.items()},
        **{label + "/F1": metric for label, metric in label2_f1.items()},
    }

    res = {prefix + key + suffix: value for key, value in res.items()}
    per_class_res = {prefix + key + suffix: value for key, value in per_class_res.items()}

    return res, per_class_res


def _filter_classes(y_true, y_pred, class_group):
    reduced_y_true = []
    reduced_y_pred = []

    for expected_class, predicted_class in zip(y_true, y_pred):
        if expected_class in class_group:
            reduced_y_true.append(expected_class)
            reduced_y_pred.append(predicted_class)

    return reduced_y_true, reduced_y_pred


def _handle_prefix(prefix: Union[str, None]):
    if not prefix:
        return ""

    if not (prefix.endswith("_") or prefix.endswith("/")):
        prefix = prefix + "_"

    return prefix


def _handle_suffix(suffix: Union[str, None]):
    if not suffix:
        return ""

    if not (suffix.startswith("_") or suffix.startswith("/")):
        suffix = "_" + suffix

    return suffix
