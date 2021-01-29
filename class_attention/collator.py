import torch


class CatCollator:
    """
    Collates text into batches and creates targets.

    Remember that label is a tokenized string and not a target that is used to compute the loss.
    For more information, look up the documentation for the __call__ method.

    Args:
        pad_token_id: paddng token id used for BOTH texts and classes
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        """
        Collates examples into batches and creates targets for the ``contrastive loss''.

        Note that _pre_batch_y.shape[0] == n_unique_labels != batch_size!

        Args:
            examples: list of tuples (text_seq, label_seq)
                where
                text_seq: LongTensor[text_len,]
                label_seq: LongTensor[label_len,]

        Returns:
            a tuple (batch_x, unique_labels, targets)
                where
                batch_x: LongTensor[batch_size, max_text_len]
                unique_labels: LongTensor[n_unique_labels, max_label_len]
                targets: LongTensor[batch_size,]
        """
        _validate_input(examples)

        batch_size = len(examples)
        max_text_len = max(len(text) for text, label in examples)
        max_label_len = max(len(label) for text, label in examples)
        device = examples[0][0].device

        # we construct this tensor only to use it in the torch.unique operation, we do not return it
        _pre_batch_y = torch.full(
            size=[batch_size, max_label_len],
            fill_value=self.pad_token_id,
            dtype=torch.int64,
            device=device,
        )

        batch_x = torch.full(
            size=[batch_size, max_text_len],
            fill_value=self.pad_token_id,
            dtype=torch.int64,
            device=device,
        )

        for i, (text, label) in enumerate(examples):
            batch_x[i, : len(text)] = text
            _pre_batch_y[i, : len(label)] = label

        # inverse is the mapping from unique_labels to the original _pre_batch_y indices
        # and this is precisely our lables
        unique_labels, targets = torch.unique(_pre_batch_y, dim=0, return_inverse=True)

        # Q: can/should we shuffle the targets and unique_labels here?
        # A: no, because the dataloader shuffles for us

        return batch_x, unique_labels, targets


class CatTestCollator:
    """
    Collates text into batches with a fixed set of classes.

    During inference, we do not know in advance what class we have to predict.
    Thus, using a regular CatCollator that only inputs into the model
    a subset of classes is not possible as it would be cheating.

    The set of possible classes in this collator is defined during initialization
    and all of them are used for every batch.

    Args:
        possible_labels_ids: torch.LongTensor[n_labels, label_len], a matrix of padded label ids
        pad_token_id: paddng token id used for BOTH texts and classes
    """

    def __init__(self, possible_labels_ids: torch.LongTensor, pad_token_id):
        self.possible_labels = possible_labels_ids
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        """
        Collates examples into batches and creates targets for the ``contrastive loss''.

        The main difference with CatCollator is the unique_labels.
        In the case of test collator, it is always equal to self.possible_labels
        and do not depend on the batch classes.

        Args:
            examples: list of tuples (text_seq, label_seq)
                where
                text_seq: LongTensor[text_len,]
                label_seq: LongTensor[label_len,]

        Returns:
            a tuple (batch_x, unique_labels, targets)
                where
                batch_x: LongTensor[batch_size, max_text_len]
                unique_labels: LongTensor[batch_size,] = self.possible_labels
                targets: LongTensor[batch_size,]
        """
        _validate_input(examples)

        batch_size = len(examples)
        max_text_len = max(len(text) for text, label in examples)
        max_label_len = max(
            len(label) for label in self.possible_labels
        )  # 1st major difference from CatCollator
        device = examples[0][0].device

        # we construct this tensor only to use it in get_index, we do not return it
        _pre_batch_y = torch.full(
            size=[batch_size, max_label_len],
            fill_value=self.pad_token_id,
            dtype=torch.int64,
            device=device,
        )

        batch_x = torch.full(
            size=[batch_size, max_text_len],
            fill_value=self.pad_token_id,
            dtype=torch.int64,
            device=device,
        )

        for i, (text, label) in enumerate(examples):
            batch_x[i, : len(text)] = text
            _pre_batch_y[i, : len(label)] = label

        targets = get_index(
            self.possible_labels, _pre_batch_y
        )  # 2nd major difference from CatCollator

        if batch_size != targets.shape[0]:
            raise RuntimeError(
                f"Wrong number of targets. Expected {batch_size}, got {targets.shape[0]} instead."
            )

        return batch_x, self.possible_labels, targets


def _validate_input(examples):
    if not isinstance(examples[0], tuple):
        raise ValueError(examples)

    text_0, label_0 = examples[0]
    if not len(text_0.shape) == 1:
        raise ValueError(
            f"Wrong number of dimensions in the text tensor. "
            f"Expected a rank-one tensor, got rank-{len(text_0.shape)} instead"
        )

    if not len(label_0.shape) == 1:
        raise ValueError(
            f"Wrong number of dimensions in the label tensor. "
            f"Expected a rank-one tensor, got rank-{len(label_0.shape)} instead"
        )


# source: https://discuss.pytorch.org/t/how-to-get-the-row-index-of-specific-values-in-tensor/28036/7
def get_index(host, target):
    diff = target.unsqueeze(1) - host.unsqueeze(0)
    dsum = torch.abs(diff).sum(-1)
    loc = torch.nonzero(dsum == 0)
    return loc[:, -1]
