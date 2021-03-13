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


class GloVeTokenizer:
    """Pretends a Transformers.Tokenizer, used for GloVe embeddings"""

    def __init__(self, word2id, pad_token_id=0):
        # tokenizer.pad_token_id -> int, should be the same as text_tokenizer.pad_token_id
        if pad_token_id != 0:
            raise ValueError("GloVeTokenizer only supports pad_token_id=0")

        self.word2id = word2id
        self.pad_token_id = int(pad_token_id)

    def encode(self, text):
        # tokenizer.encode -> numpy tensor of shape [seq_len,]
        text = text.lower()

        # NOTE: hardcoded tokenization
        if "worldpost" not in self.word2id:
            text = re.sub("worldpost", "world post", text)

        tokens = text.split(" ")  # good enough for class names
        ids = [self.word2id[t] for t in tokens]
        return np.array(ids)

    def batch_encode_plus(self, texts, **kwargs):
        # tokenizer.batch_encode_plus -> returns an object with "input_ids" key that contains a Tensor
        if not isinstance(texts, list):
            raise ValueError(texts)

        batch = [self.encode(t) for t in texts]
        batch_size = len(batch)
        max_len = max(map(len, batch))

        batch_pt = torch.ones(batch_size, max_len, dtype=torch.int64) * self.pad_token_id
        for i, example in enumerate(batch):
            batch_pt[i, : len(example)] = torch.LongTensor(example)

        return {"input_ids": batch_pt}


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
    Move classes to a class-test set (i.e. meta-test).

    All dataset examples with these classes are removed from the original dataset

    Args:
        dataset: datasets.arrow_dataset.Dataset object
        p_test_classes: 0 < float < 1
        test_classes: alternative to p_test_classes, a list of classes to move to the class-test set,
            capitalization is ignored
        class_field_name: name of the class field in the dataset
        verbose: log splitted classes info

    Returns:
        (train_set, class_test_set)
        where both objects are ArrowDataset and all test classes are moved to class_test_set
    """
    test_classes = {t.lower() for t in test_classes}

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
        all_classes = list(set(dataset[class_field_name]))
        n_test_classes = int(len(all_classes) * p_test_classes)
        if n_test_classes == 0:
            raise ValueError(
                f"p_test_classes={p_test_classes} is too small for the dataset with {len(all_classes)} classes."
            )

        all_classes = {c.lower() for c in all_classes}
        test_classes = random.sample(all_classes, k=n_test_classes)

    if verbose:
        print(f"Moving the following classes to a class-test set: {test_classes}")

    test_mask = [c.lower() in test_classes for c in dataset[class_field_name]]
    train_mask = [not m for m in test_mask]

    test_subset = dataset[test_mask]
    train_subset = dataset[train_mask]  # NOTE: dict of lists, not a list of dicts

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


def load_glove_from_file(path):
    word_list = []
    embeddings = []

    with open(path) as f:
        for line in f:
            if not line:
                continue

            word, vector = line.split(" ", 1)
            vector = np.array(list(map(float, vector.split(" "))))

            word_list.append(word)
            embeddings.append(vector)

    if len(word_list) == 0:
        raise ValueError(f"Received an empty file via path {path}")

    emb_dim = embeddings[0].size
    # add padding vector
    embeddings = [np.zeros(emb_dim)] + embeddings

    embedding_matrix = np.array(embeddings)

    assert len(set(word_list)) == len(word_list)
    word2id = {w: i for i, w in enumerate(word_list, start=1)}
    assert "[PAD]" not in word2id
    word2id["[PAD]"] = 0

    assert len(word2id) == embedding_matrix.shape[0]

    return embedding_matrix, word2id


def infinite_iterator(iterable):
    if isinstance(iterable, torch.utils.data.DataLoader):
        if not isinstance(iterable.sampler, torch.utils.data.sampler.RandomSampler):
            raise RuntimeError("this dataloader should use random sampling")

    while True:
        for x in iter(iterable):
            yield x


def extract_words_from_glove(glove_path):
    with open(glove_path) as f:
        words = [line.strip().split(" ", 1)[0] for line in f]

    filtered_words = filter_words(words)
    return filtered_words


def filter_words(words, extra_filter=None):
    res = [w for w in words if (not w.isdigit() and len(w) > 2 and "'" not in w)]
    if extra_filter is not None:
        res = [w for w in res if extra_filter(w)]

    return res
