import random
from collections import Counter

import datasets
import tokenizers
import tokenizers.pre_tokenizers
import tokenizers.normalizers
from tokenizers.models import WordLevel


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
