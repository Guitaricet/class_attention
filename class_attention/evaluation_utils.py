"""
Functions responsible for model evaluation
"""
import logging
import os
import sys

from typing import Union, List

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
    labels_str,
    zeroshot_labels=None,
    progress_bar=False,
    predict_into_file=None,
    ranking_dataloader=None,
):
    """Makes predictions on dataloader, reports metrics.

    Args:
        model: ClassAttentionModel
        dataloader: pytorch DataLoader with CatTestCollator
        labels_str: List[str], names of classes, in the same order as in the CatTestCollator.possible_label_ids
        zeroshot_labels: if provided, additional metrics will be computed on this set of labels
        ranking_dataloader: if provided, recall@k is computed on this one,
            can consume a lot of time and **memory** for datasets >> 1000 examples

    Example output:
    {
        'eval/acc': 0.25,
        'eval/P_macro': 0.08333333125000005,
        'eval/R_macro': 0.33333330000000333,
        'eval/F1_macro': 0.13333331733333392,
        'eval_per_class/Weather/P': 0.24999999375000015,
        'eval_per_class/Sport/P': 0.0,
        'eval_per_class/My News/P': 0.0,
        'eval_per_class/Weather/R': 0.9999999000000099,
        'eval_per_class/Sport/R': 0.0,
        'eval_per_class/My News/R': 0.0,
        'eval_per_class/Weather/F1': 0.39999995200000177,
        'eval_per_class/Sport/F1': 0.0,
        'eval_per_class/My News/F1': 0.0,
        'zero_shot_eval/acc': 0.0,
        'zero_shot_eval/P_macro': 0.0,
        'zero_shot_eval/R_macro': 0.0,
        'zero_shot_eval/F1_macro': 0.0,
        'multi_shot_eval/acc': 0.5,
        'multi_shot_eval/P_macro': 0.24999998750000066,
        'multi_shot_eval/R_macro': 0.49999995000000497,
        'multi_shot_eval/F1_macro': 0.3333332888888915
    }
    """
    if predict_into_file is not None and not isinstance(
        dataloader.sampler, torch.utils.data.sampler.SequentialSampler
    ):
        raise ValueError("test dataloader should not be shuffled")

    if zeroshot_labels is not None and (not set(zeroshot_labels).issubset(labels_str)):
        raise ValueError("labels_str should include all labels")

    all_predictions, all_labels, all_texts = predict(
        model=model,
        dataloader=dataloader,
        labels_str=labels_str,
        return_texts=(predict_into_file is not None),
        progress_bar=progress_bar,
    )

    if predict_into_file is not None:
        assert (
            len(all_texts) == len(all_predictions) == len(all_labels)
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

    if ranking_dataloader is not None:
        ranking_metrics = evaluate_ranking_model(model, ranking_dataloader)
        res.update(ranking_metrics)

    model.train()
    return res


# the code is adapted from
# https://github.com/KevinMusgrave/pytorch-metric-learning/blob/dff4ae570db89dcb59a102f13f665502f9c1c7c6/src/pytorch_metric_learning/testers/base_tester.py#L79
# MIT licence
def get_all_embeddings(model, ranking_dataloader):
    model.eval()
    start, end = 0, 0

    with torch.no_grad():
        for i, (text, title, labels) in enumerate(ranking_dataloader):
            # NOTE: you can get rid of title.index_select if you guarantee that labels == range(len(title))
            ordered_titles = title.index_select(0, labels)
            text_emb, title_emb = model.embed_texts_and_labels(text, ordered_titles)
            batch_size = text_emb.size(0)

            if i == 0:
                all_text_embs = torch.zeros(
                    len(ranking_dataloader.dataset),
                    text_emb.size(1),
                    device=text_emb.device,
                    dtype=text_emb.dtype,
                )
                all_title_embs = torch.zeros(
                    len(ranking_dataloader.dataset),
                    title_emb.size(1),
                    device=title_emb.device,
                    dtype=title_emb.dtype,
                )

            end = start + batch_size
            all_text_embs[start:end] = text_emb
            all_title_embs[start:end] = title_emb
            start = end

    model.train()

    return all_text_embs, all_title_embs


def recall_at_k(x_embs, y_embs, k=1):
    """we assume that x_embs[i] true label is y_embs[i]

    Args:
        x_embs:
        y_embs:
        k:

    Returns:

    """
    assert x_embs.shape == y_embs.shape

    n_matches = 0
    for i, x_emb in enumerate(x_embs):
        scores = y_embs @ x_emb.T  # [n_labels, 1]
        values, indices = torch.topk(scores.squeeze(-1), k=k)
        if i in indices:
            n_matches += 1

    return n_matches / len(x_embs)


def evaluate_ranking_model(model, ranking_dataloader):
    if len(ranking_dataloader.dataset) > 3000:
        logger.warning(
            "Evaluating on a relatively big dataset can take a long time as we perform brute-force KNN search"
        )

    all_text_embs, all_title_embs = get_all_embeddings(model, ranking_dataloader)

    recall_at_1 = recall_at_k(all_text_embs, all_title_embs, k=1)
    recall_at_5 = recall_at_k(all_text_embs, all_title_embs, k=5)

    return {"rank_eval/R@1": recall_at_1, "rank_eval/R@5": recall_at_5}


def predict(
    model, dataloader, labels_str, return_texts=False, progress_bar=False
) -> (List, List, List):
    """Makes predictions on dataloader, reports metrics.

    Args:
        model: ClassAttentionModel
        dataloader: pytorch DataLoader with CatTestCollator
        labels_str: List[str], names of classes, in the same order as in the CatTestCollator.possible_label_ids
        return_texts: if provided, the last element of the return tuple will be a list of classified texts
        progress_bar: use tqdm during prediction

    Returns:
        tuple (all_predictions, all_labels, all_texts) where everything
        is a list of strings
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_texts = [] if return_texts else None
    text_tokenizer = dataloader.dataset.text_tokenizer

    if progress_bar:
        dataloader = tqdm(dataloader, desc="Evaluation")

    with torch.no_grad():
        for x, c, y in dataloader:
            # Note: `c` does not change in CatTestCollator
            logits = model(x, c)

            _, preds = logits.max(-1)

            predicted_labels = [labels_str[i] for i in preds]
            expected_labels = [labels_str[i] for i in y]

            all_predictions += predicted_labels
            all_labels += expected_labels
            if return_texts:
                # NOTE: this may cause concurrency issues or semaphore failures if tokenizer parallelism is not disabled
                all_texts += text_tokenizer.batch_decode(x, skip_special_tokens=True)

    return all_predictions, all_labels, all_texts


def evaluate_model_on_subset(
    model,
    dataset_str,
    text_field,
    class_field,
    test_classes_str,
    text_tokenizer,
    label_tokenizer,
    accelerator,
    predict_into_file=None,
):
    """

    Args:
        model: CatModel
        dataset_str: ArrowDataset with texts and textual description of classes
        text_field: text field name in dataset_str
        class_field: class field name in dataset_str
        test_classes_str: a list of test class names that will be used for evaluation
        text_tokenizer: transformers.Tokenizer object used to encode texts
        label_tokenizer: transformers.Tokenizer object used to encode class descripitons
        accelerator: accelerate.Accelerator object
        predict_into_file: if specified, save predictions into a file with this path

    Returns:
        dict {str: float} with metrics, same as cat.evaluation_utils.evaluate_model
    """

    # dataset filtration happens in make_test_classes_only_dataloader
    test_dataloader = make_test_classes_only_dataloader(
        dataset=dataset_str,
        test_classes_str=test_classes_str,
        text_tokenizer=text_tokenizer,
        label_tokenizer=label_tokenizer,
        text_field=text_field,
        class_field=class_field,
    )
    test_dataloader = accelerator.prepare_data_loader(test_dataloader)

    subset_metrics = evaluate_model(
        model,
        test_dataloader,
        labels_str=test_classes_str,
        zeroshot_labels=None,  # it needs to be None, so that the metrics are not splitted in two subsets
        predict_into_file=predict_into_file,
    )

    return subset_metrics


def make_test_classes_only_dataloader(
    dataset, test_classes_str, text_tokenizer, label_tokenizer, text_field, class_field
):
    """Filters dataset to only contain test_classes_str and makes a dataloader with CatTestCollator

    Args:
        dataset: ArrowDataset
        test_classes_str: list of test class names

    Returns:
        DataLoader with CatTestCollator
    """
    if text_field is None or class_field is None:
        raise ValueError("text_field and class_field are required")

    _, only_test_classes_data = cat.utils.split_classes(
        dataset, class_field=class_field, test_classes=test_classes_str
    )
    assert set(only_test_classes_data[class_field]) == set(test_classes_str), (
        set(only_test_classes_data[class_field]),
        set(test_classes_str),
    )

    otc_dataset = cat.CatDataset(
        only_test_classes_data[text_field],
        text_tokenizer,
        only_test_classes_data[class_field],
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
    if len(y_true) < 1:
        raise ValueError("y_true and y_pred should be at least one element")

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
        "F1_weighted" + suffix: f1_weighted_score(label2_f1, label2n_expected),
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


def f1_weighted_score(label2_f1, label2_count):
    assert label2_f1.keys() == label2_count.keys()

    _norm = sum(label2_count.values())
    label2_weight = {c: n / _norm for c, n in label2_count.items()}
    assert round(sum(label2_weight.values()), 6) == 1.0

    score = sum(label2_weight[c] * label2_f1[c] for c in label2_f1.keys())
    return score
