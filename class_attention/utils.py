import re
import logging
import os
import random
import sys
from collections import Counter

import torch
import torch.utils.data
import numpy as np

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


def split_classes(dataset, class_field, p_test_classes=None, test_classes=None, verbose=False):
    """
    Move classes to a class-test set (i.e. meta-test).

    All dataset examples with these classes are removed from the original dataset

    Args:
        dataset: datasets.arrow_dataset.Dataset object
        p_test_classes: 0 < float < 1
        test_classes: alternative to p_test_classes, a list of classes to move to the class-test set,
            capitalization is ignored
        class_field: name of the class field in the dataset
        verbose: log splitted classes info

    Returns:
        (train_set, class_test_set)
        where both objects are ArrowDataset and all test classes are moved to class_test_set
    """
    if class_field is None:
        raise ValueError("class_field is required")

    if not isinstance(dataset, datasets.arrow_dataset.Dataset):
        raise ValueError(type(dataset))

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
        all_classes = list(set(dataset[class_field]))
        n_test_classes = int(len(all_classes) * p_test_classes)
        if n_test_classes == 0:
            raise ValueError(
                f"p_test_classes={p_test_classes} is too small for the dataset with {len(all_classes)} classes."
            )

        test_classes = random.sample(all_classes, k=n_test_classes)

    if verbose:
        print(f"Moving the following classes to a class-test set: {test_classes}")

    test_classes = {t.lower() for t in test_classes}
    test_mask = [c.lower() in test_classes for c in dataset[class_field]]
    train_mask = [not m for m in test_mask]

    test_subset = dataset[test_mask]
    train_subset = dataset[train_mask]  # NOTE: dict of lists, not a list of dicts

    assert set(test_classes) == set(c.lower() for c in test_subset[class_field])

    test_dataset = datasets.arrow_dataset.Dataset.from_dict(test_subset)
    train_dataset = datasets.arrow_dataset.Dataset.from_dict(train_subset)

    return train_dataset, test_dataset


def monospace_html(text):
    return f"""<code><pre>{text}</code></pre>"""


def get_dataset_by_name_or_path(name_or_path):
    try:
        dataset = datasets.load_from_disk(name_or_path)
    except FileNotFoundError:
        try:
            dataset = datasets.load_dataset(name_or_path)
        except FileNotFoundError:
            raise ValueError(f"The dataset {name_or_path} wasn't found locally or downloaded")

    return dataset


def infinite_iterator(iterable):
    if isinstance(iterable, torch.utils.data.DataLoader):
        if not isinstance(iterable.sampler, torch.utils.data.sampler.RandomSampler):
            raise RuntimeError("this dataloader should use random sampling")

    while True:
        for x in iter(iterable):
            yield x


def filter_words(words, extra_filter=None):
    res = [w for w in words if (not w.isdigit() and len(w) > 2 and "'" not in w)]
    if extra_filter is not None:
        res = [w for w in res if extra_filter(w)]

    return res


def infer_field_names(dataset_name, text_field=None, class_field=None):
    if (text_field is None) ^ (class_field is None):
        raise ValueError("--text-field and --class-field need to be provided together")

    if text_field is not None:
        return text_field, class_field

    if "news-category" in dataset_name:
        return "headline", "category"

    if "emotion" in dataset_name or "topic" in dataset_name:
        return "text", "category"

    raise ValueError(f"Cannot infer field names from the dataset `{dataset_name}`")


def encode_classes(classes_str, label_tokenizer):
    return label_tokenizer.batch_encode_plus(
        classes_str,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
    )["input_ids"]


def get_cced(model, train_classes_str, test_classes_str, label_tokenizer, device):
    is_train = model.training
    model.eval()

    train_classes_h, test_classes_h = get_class_vectors(
        model, train_classes_str, test_classes_str, label_tokenizer, device
    )

    if is_train:
        model.train()

    train_classes_h_center = torch.mean(train_classes_h, dim=0)
    test_classes_h_center = torch.mean(test_classes_h, dim=0)

    return torch.dist(train_classes_h_center, test_classes_h_center)


def get_rmasp(model, train_classes_str, test_classes_str, label_tokenizer, device="cpu"):
    is_train = model.training
    model.eval()

    train_classes_h, test_classes_h = get_class_vectors(
        model, train_classes_str, test_classes_str, label_tokenizer, device
    )

    if is_train:
        model.train()

    correlation_matrix = train_classes_h @ test_classes_h.T
    return torch.sqrt(torch.mean(torch.abs(correlation_matrix)))


def get_class_vectors(model, train_classes_str, test_classes_str, label_tokenizer, device):
    train_classes_ids = encode_classes(train_classes_str, label_tokenizer).to(device)
    test_classes_ids = encode_classes(test_classes_str, label_tokenizer).to(device)

    # 5 because it is not a special token and because it is small
    fake_text_ids = torch.LongTensor([[5]]).to(device)  # (batch=1, seq=1)

    _, _, train_classes_h = model(
        text_input=fake_text_ids, labels_input=train_classes_ids, return_embeddings=True
    )
    _, _, test_classes_h = model(
        text_input=fake_text_ids, labels_input=test_classes_ids, return_embeddings=True
    )

    return train_classes_h, test_classes_h


# source: https://discuss.pytorch.org/t/how-to-get-the-row-index-of-specific-values-in-tensor/28036/7
def get_index(unique_tensors, instances):
    assert unique_tensors.shape[1] == instances.shape[1]
    diff = instances.unsqueeze(1) - unique_tensors.unsqueeze(0)
    dsum = torch.abs(diff).sum(-1)
    loc = torch.nonzero(dsum <= 1e-4)  # -4 because of fp16
    return loc[:, -1]


def get_difference(t1, t2):
    """Compute set difference t1 / t2"""
    sim_matrix = t1.unsqueeze(1) == t2
    sim_index = sim_matrix.all(-1).any(-1)

    difference = t1[~sim_index]
    return difference
