import pytest

import torch
import class_attention as cat


def test_collator():
    collator = cat.CatCollator(pad_token_id=0)

    # fmt: off
    examples = [
        (torch.LongTensor([1, 2, 3]), torch.LongTensor([0, 1])),
        (torch.LongTensor([1, 2, 3, 4]), torch.LongTensor([1])),
        (torch.LongTensor([1, 2]), torch.LongTensor([2])),
        (torch.LongTensor([1, 2]), torch.LongTensor([0, 1])),
    ]

    expected_batch_x = torch.LongTensor([[1, 2, 3, 0], [1, 2, 3, 4], [1, 2, 0, 0], [1, 2, 0, 0]])

    expected_unique_labels = torch.LongTensor([
        [0, 1],
        [1, 0],
        [2, 0],
    ])
    # fmt: on

    expected_targets = torch.LongTensor([0, 1, 2, 0])

    batch_x, unique_labels, targets = collator(examples)

    assert torch.all(batch_x == expected_batch_x)
    assert torch.all(unique_labels == expected_unique_labels)
    assert torch.all(targets == expected_targets)
