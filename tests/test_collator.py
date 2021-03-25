import pytest

import torch
import class_attention as cat


def test_cat_collator():
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
    expected_targets = torch.LongTensor([0, 1, 2, 0])
    # fmt: on

    batch_x, unique_labels, targets = collator(examples)

    assert torch.all(batch_x == expected_batch_x)
    assert torch.all(unique_labels == expected_unique_labels)
    assert torch.all(targets == expected_targets)


def test_cat_collator_ps():
    # fmt: off
    examples = [
        (torch.LongTensor([1, 2, 3]), torch.LongTensor([0, 1])),
        (torch.LongTensor([1, 2, 3, 4]), torch.LongTensor([1])),
        (torch.LongTensor([1, 2]), torch.LongTensor([2])),
        (torch.LongTensor([1, 2]), torch.LongTensor([0, 1])),
    ]
    unique_labels = torch.LongTensor([
        [0, 1],
        [1, 0],
        [2, 0],
        [2, 1],
        [2, 2],
    ])

    expected_batch_x = torch.LongTensor([[1, 2, 3, 0], [1, 2, 3, 4], [1, 2, 0, 0], [1, 2, 0, 0]])
    # fmt: on

    collator = cat.CatCollator(
        pad_token_id=0, possible_label_ids=unique_labels, p_no_class=0.5, p_extra_classes=0.5
    )
    batch_x, unique_labels, targets = collator(examples)

    assert torch.all(batch_x == expected_batch_x)


def test_cat_test_collator():
    # fmt: off
    possible_labels = torch.LongTensor([
        [0, 1],
        [1, 0],
        [2, 0],
    ])

    examples = [
        (torch.LongTensor([1, 2, 3]), torch.LongTensor([0, 1])),
        (torch.LongTensor([1, 2, 3, 4]), torch.LongTensor([1])),
        (torch.LongTensor([1, 2]), torch.LongTensor([2])),
        (torch.LongTensor([1, 2]), torch.LongTensor([0, 1])),
    ]

    expected_batch_x = torch.LongTensor([[1, 2, 3, 0], [1, 2, 3, 4], [1, 2, 0, 0], [1, 2, 0, 0]])
    expected_unique_labels = possible_labels
    expected_targets = torch.LongTensor([0, 1, 2, 0])
    # fmt: on

    collator = cat.CatTestCollator(possible_labels, pad_token_id=0)
    batch_x, unique_labels, targets = collator(examples)

    assert torch.all(batch_x == expected_batch_x)
    assert torch.all(unique_labels == expected_unique_labels)
    assert torch.all(targets == expected_targets)


def test_get_difference():
    # fmt: off
    x = torch.LongTensor([
        [1, 2],
        [2, 3],
        [3, 4],
    ])

    y = torch.LongTensor([
        [3, 4],
        [7, 8],
        [1, 2],
    ])

    expected_diff = torch.LongTensor([
        [2, 3],
    ])
    # fmt: on

    diff = cat.collator.get_difference(x, y)
    assert torch.all(diff == expected_diff)

    no_diff = cat.collator.get_difference(x, x)
    assert no_diff.size(0) == 0


def test_get_index_with_default():
    # fmt: off
    x = torch.LongTensor([
        [1, 2],
        [2, 3],
        [3, 4],
    ])

    y = torch.LongTensor([
        [3, 4],
        [7, 8],
        [1, 2],
        [2, 3],
        [3, 4]
    ])

    expected_index = torch.LongTensor([2, -1, 0, 1, 2])
    # fmt: on

    idx = cat.collator.get_index_with_default_index(x, y, torch.tensor(-1))
    assert torch.all(idx == expected_index)
