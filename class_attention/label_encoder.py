import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class LabelEncoder:
    """
    Numericalizes batches of labels given possible labels, also returns targets tensor that can be used to compute loss.

    Note: this label encoder can only encode classes desribed by a single index and mainly used for debug.

    Args:
        all_classes: list[str], text description of the classes (e.g., class names)

    Usage:
        label_tokenizer = LabelEncoder(['Weather', 'News', 'Music', 'Sport'])

        classes_str = ['News', 'Weather', 'Weather', 'News']
        possible_classes = ['Weather', 'News', 'Sport']

        possible_class_ids, targets = label_tokenizer.encode(classes_str, possible_classes)
        # possible_class_ids: [0, 1, 2]
        # targets: [1, 2, 2, 1]

        logits = net(text_ids, possible_class_ids)
        loss = F.cross_entropy(logits, targets)
    """

    def __init__(self, all_classes):
        self._classes_set = set(all_classes)
        self._id2label = all_classes
        self._label2id = {l: i for i, l in enumerate(self._id2label)}

    def __repr__(self):
        return f"LabelEncoder{list(self._classes_set)})"

    @property
    def possible_classes(self):
        return self._id2label

    def encode(self, classes_str, possible_classes=None):
        """
        Args:
            classes_str: list[str] of size batch_size, class descriptions that need to be encoded
            possible_classes: list[str] of size n_classes, a subset of self.possible_classes - all possible classes for current batch

        Returns:
            tuple(label_ids, target) where
                label_ids: torch.LongTensor[n_classes] label ids of the possible classes, fed to the label encoder
                target: torch.LongTensor[batch_size] of targets that use possible_classes for ordering,
                    used for loss computation

        """
        if possible_classes is None:
            possible_classes = set(classes_str)

        if not set(possible_classes).issubset(self._classes_set):
            raise ValueError(
                f"possible_classes {possible_classes} contains classes not from the LabelEncoder .possible_classes"
            )

        if not set(classes_str).issubset(self._classes_set):
            raise ValueError(
                f"classes_str {classes_str} contains classes not from the LabelEncoder .possible_classes"
            )

        label_str_ids = torch.LongTensor([self._label2id[c] for c in possible_classes])

        return label_str_ids

    def decode_label_ids(self, label_str_ids):
        return [self._id2label[i] for i in label_str_ids]

    def decode_target(self, target, possible_classes):
        return [possible_classes[t] for t in target]
