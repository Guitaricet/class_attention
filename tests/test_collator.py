import pytest

import torch
import class_attention as cat

import tests.utils


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


def test_cat_test_collator_workers():
    # fmt: off
    (
        train_dataloader,
        test_dataloader,
        all_classes_str,
        test_classes_str,
        data,
        _,
    ) = tests.utils.default_prepare_dataloaders(
        dataset_frac=0.5,
        num_workers=8,
        batch_size=512,
    )
    # fmt: on

    for x, y, c in test_dataloader:
        pass


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

    diff = cat.utils.get_difference(x, y)
    assert torch.all(diff == expected_diff)

    no_diff = cat.utils.get_difference(x, x)
    assert no_diff.size(0) == 0
