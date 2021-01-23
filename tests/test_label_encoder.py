import copy
import random
import pytest

from class_attention.label_encoder import LabelEncoder


def test_encode():

    n_classes = 7
    batch_size = 5

    possible_classes = [f"Class {str(i)}" for i in range(n_classes)]
    y_str = random.choices(possible_classes, k=batch_size)

    label_encoder = LabelEncoder(possible_classes)

    for _ in range(2):
        for _possible_classes in [
            possible_classes,
        ]:
            _possible_classes = copy.copy(_possible_classes)
            random.shuffle(_possible_classes)
            y, targets = label_encoder.encode(y_str, _possible_classes)

            print("Possible labels: \n", _possible_classes)
            print("Str: \n", y_str)
            print("Encoded strs: \n", y)
            print("Targers corresponding to possible classes: \n", targets)
            print("Decoded label strings: \n", label_encoder.decode_label_ids(y))
            print("Decoded targets: \n", label_encoder.decode_target(targets, _possible_classes))
            print()

            self.assertEqual(_possible_classes, label_encoder.decode_label_ids(y))
            self.assertEqual(y_str, label_encoder.decode_target(targets, _possible_classes))
