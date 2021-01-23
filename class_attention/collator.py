import torch


class CatCollator:
    """
    Collates text into batches and creates targets.

    Remember that label is a tokenized string and not a target that is used to compute the loss.
    For more information, look up the documentation for the __call__ method.

    Args:
        pad_token_id: paddng token id used for BOTH texts and labels
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
        self._validate_input(examples)

        batch_size = len(examples)
        max_text_len = max(len(text) for text, label in examples)
        max_label_len = max(len(label) for text, label in examples)
        device = examples[0][0].device

        # we construct this tensor only to for the torch.unique operation, we do not return it
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

    def _validate_input(self, examples):
        if not isinstance(examples[0], tuple):
            raise ValueError(examples)

        text_0, label_0 = examples[0]
        if not len(text_0.shape) == 1:
            raise ValueError(
                f"Wrong number of dimentions in the text tensor. "
                f"Expected a rank-one tensor, got rank-{len(text_0.shape)} instead"
            )

        if not len(label_0.shape) == 1:
            raise ValueError(
                f"Wrong number of dimentions in the label tensor. "
                f"Expected a rank-one tensor, got rank-{len(label_0.shape)} instead"
            )
