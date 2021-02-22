import logging
import os
import random
import sys
from collections import Counter, defaultdict

import torch
import datasets
import tokenizers
import tokenizers.pre_tokenizers
import tokenizers.normalizers
from tokenizers.models import WordLevel


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


def make_whitespace_tokenizer(texts, max_vocab_size=10_000, unk_token="[UNK]", pad_token="[PAD]"):
    """
    Creates a simple tokenizer that splits a lowercased string into words via whitespace.

    Args:
        texts: a collection of texts to extract vocabulary from
        max_vocab_size: maximum size of the vocabulary
        unk_token: UNK token string representation, if None, UNK is not used
        pad_token: PAD token string representation

    Returns:
        tokenizers.Tokenizer object
    """
    pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    normalizer = tokenizers.normalizers.Lowercase()
    tokenized_texts = [
        [w for w, _ in pre_tokenizer.pre_tokenize_str(normalizer.normalize_str(t))] for t in texts
    ]

    c = Counter()
    for text in tokenized_texts:
        c.update(text)

    if unk_token is None:
        token2id = {word: i + 1 for i, (word, count) in enumerate(c.most_common(max_vocab_size))}
        token2id[pad_token] = 0
    else:
        token2id = {word: i + 2 for i, (word, count) in enumerate(c.most_common(max_vocab_size))}
        token2id[pad_token] = 0
        token2id[unk_token] = 1

    tokenizer = tokenizers.Tokenizer(WordLevel(token2id, unk_token))
    tokenizer.enable_padding(pad_token=pad_token, pad_id=0)
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    tokenizer.pad_token_id = 0
    return tokenizer


def sample_dataset(dataset, p):
    """Samples a smaller version of a dataset.

    Mainly used for debugging and testing.

    Args:
        dataset: datasets.arrow_dataset.Dataset object
        p: float, 0 < p <= 1

    Returns:
        datasets.arrow_dataset.Dataset of size len(dataset) * p with random examples from the dataset
        sampled without replacement
    """
    if not 0 < p <= 1:
        raise ValueError(p)

    dataset_len = len(dataset)
    sample_size = int(p * dataset_len)

    ids = random.sample(range(len(dataset)), sample_size)

    # indexing actually creates dict with elements of len(ids), not a list
    sampled_dataset_dict = dataset[ids]
    sampled_dataset = datasets.arrow_dataset.Dataset.from_dict(sampled_dataset_dict)
    return sampled_dataset


def split_classes(
    dataset, p_test_classes=None, test_classes=None, class_field_name="category", verbose=False
):
    """
    Move random classes to a class-test set (i.e. meta-test).

    All dataset examples with these classes are removed from the original dataset

    Args:
        dataset: datasets.arrow_dataset.Dataset object
        p_test_classes: 0 < float < 1
        test_classes: alternative to p_test_classes, a list of classes to move to the class-test set
        class_field_name: name of the class field in the dataset
        verbose: log splitted classes info

    Returns:
        (train_set, class_test_set)
        where both objects are ArrowDataset and all test classes are moved to class_test_set
    """

    if not ((p_test_classes is None) ^ (test_classes is None)):
        raise ValueError(
            "Only one of p_test_classes or test_classes should be specified. "
            f"Got p_test_classes = {p_test_classes}\n"
            f"test_classes = {test_classes}"
        )

    if p_test_classes == 0:
        if test_classes is not None:
            raise ValueError("test classes should not be specified if p_test_classes=0")

        return dataset, None

    if p_test_classes is not None:
        all_classes = list(set(dataset[class_field_name]))
        n_test_classes = int(len(all_classes) * p_test_classes)
        if n_test_classes == 0:
            raise ValueError(
                f"p_test_classes={p_test_classes} is too small for the dataset with {len(all_classes)} classes."
            )

        test_classes = random.sample(all_classes, k=n_test_classes)

    if verbose:
        print(f"Moving the following classes to a class-test set: {test_classes}")

    test_mask = [c in test_classes for c in dataset[class_field_name]]
    train_mask = [not m for m in test_mask]

    test_subset = dataset[test_mask]
    train_subset = dataset[train_mask]  # NOTE: dict of lists, not a list of dicts

    test_dataset = datasets.arrow_dataset.Dataset.from_dict(test_subset)
    train_dataset = datasets.arrow_dataset.Dataset.from_dict(train_subset)

    return train_dataset, test_dataset


def evaluate_model(model, dataloader, device):
    if "CatTestCollator" not in str(dataloader.collate_fn):
        raise RuntimeError(
            "Validation or test dataloader should have a CatTestCollator instead of CatCollator"
        )

    model = model.to(device)
    model.eval()
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for x, c, y in dataloader:
            # Note: `c` does not change in CatTestCollator
            x, c, y = x.to(device), c.to(device), y.to(device)

            logits = model(x, c)

            _, preds = logits.max(-1)

            n_correct += torch.sum(preds == y).float()
            n_total += x.shape[0]

    acc = n_correct / n_total

    model.train()

    return acc


def evaluate_model_per_class(model, dataloader, device, labels_str, zeroshot_labels=None):
    """
    Args:
        labels_str: List[str], names of classes, in the same order as in the CatTestCollator.possible_labels
    """
    model = model.to(device)
    model.eval()

    if zeroshot_labels is not None and (not set(zeroshot_labels).issubset(labels_str)):
        raise ValueError("labels_str should include all labels")

    n_correct = 0
    n_total = 0
    label2n_correct = defaultdict(int)
    label2n_predicted = defaultdict(int)
    label2n_expected = defaultdict(int)

    with torch.no_grad():
        for x, c, y in dataloader:
            # Note: `c` does not change in CatTestCollator
            x, c, y = x.to(device), c.to(device), y.to(device)

            logits = model(x, c)

            _, preds = logits.max(-1)

            predicted_labels = [labels_str[i] for i in preds]
            expected_labels = [labels_str[i] for i in y]

            for label_pred, label_exp in zip(predicted_labels, expected_labels):
                label2n_predicted[label_pred] += 1
                label2n_expected[label_exp] += 1
                label2n_correct[label_pred] += int(label_pred == label_exp)

            n_correct += torch.sum(preds == y).float()
            n_total += x.shape[0]

    res = {
        "acc": n_correct / n_total,
    }

    for label in labels_str:
        label_str = label
        p = label2n_correct[label] / (label2n_predicted[label] + 1e-7)
        r = label2n_correct[label] / (label2n_expected[label] + 1e-7)

        res[f"P/{label_str}"] = p
        res[f"R/{label_str}"] = r
        res[f"F1/{label_str}"] = 2 * (p * r) / (p + r + 1e-7)

    if zeroshot_labels is not None:
        zeroshot_metrics = _aggregate_metrics_by_class_group(res, zeroshot_labels, "zero_shot")
        multishot_labels = set(labels_str).difference(set(zeroshot_labels))
        multishot_metrics = _aggregate_metrics_by_class_group(res, multishot_labels, "multi_shot")

        res.update(zeroshot_metrics)
        res.update(multishot_metrics)

    model.train()
    return res


def _aggregate_metrics_by_class_group(metrics, class_group, suffix):
    """
    Averages metrics in the class_group.

    Used to compute metrics for zero-shot vs multi-shot groups.

    Assumes that metrics has the keys that look like
        f"{metric}/{class_name}"
    where metric is in ["R", "P", "F1"]
    and

    Args:
        metrics: dict as described above
        class_group: a list of classes
        suffix: suffix for the dict keys

    Returns:
        dict with keys "R_{suffix}", "P_{suffix}", "F1_{suffix}"
    """
    if not isinstance(class_group, (list, set)):
        raise ValueError(f"class_group should be a list of a set, got {type(class_group)} instead")

    res = dict()

    for metric in ["R", "P", "F1"]:
        class_group_metrics = [metrics[f"{metric}/{c}"] for c in class_group]
        if len(class_group_metrics) == 0:
            logger.warning(f"No classes for the group {class_group}")
            continue

        metric_value = sum(class_group_metrics) / len(class_group_metrics)
        res[f"{metric}_{suffix}"] = metric_value

    return res
