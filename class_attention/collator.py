import torch


class CatCollator:
    """
    Collates text into batches and creates targets.

    Remember that label is a tokenized string and not a target that is used to compute the loss.
    For more information, look up the documentation for the __call__ method.

    Args:
        pad_token_id: paddng token id used for BOTH texts and classes
        p_extra_classes: 0 <= float <= 1, if 0 no class names are added to the ones in the batch;
            if 1 then all possible_labels_ids are used in each batch
        possible_label_ids: torch.LongTensor[n_labels, label_len], a matrix of padded label ids;
            required if p_extra_classes is specified
        p_no_class: 0 <= float < 1, proportion of data without true label
    """

    def __init__(self, pad_token_id, possible_label_ids=None, p_extra_classes=0, p_no_class=0):
        if not 0 <= p_extra_classes <= 1:
            raise ValueError(p_extra_classes)

        if p_extra_classes > 0 and possible_label_ids is None:
            raise ValueError("if p_extra_classes > 0 possible_labels_ids should be specified")

        if possible_label_ids is not None:
            if torch.unique(possible_label_ids, dim=0).size() != possible_label_ids.size():
                raise ValueError("non-unique rows in possible label ids")

        self.pad_token_id = pad_token_id
        self.possible_label_ids = possible_label_ids
        self.p_extra_classes = p_extra_classes
        self.p_no_class = p_no_class

        if self.possible_label_ids is not None:
            self._max_label_len = max(len(label) for label in self.possible_label_ids)

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
        has_labels = bool(isinstance(examples[0], tuple) and len(examples[0]) > 1)

        if has_labels:
            return self._call_with_labels(examples)

        assert isinstance(examples[0], torch.Tensor)
        return self._call_without_labels(examples)

    def _call_with_labels(self, examples):
        batch_size = len(examples)
        max_text_len = max(len(text) for text, label in examples)
        max_label_len = max(len(label) for text, label in examples)

        if self.p_extra_classes > 0:
            max_label_len = self._max_label_len

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
        original_unique_labels = unique_labels.detach().clone()

        # Q: can/should we shuffle the targets and unique_labels here?
        # A: no, because the dataloader shuffles for us

        if self.p_no_class > 0:
            unique_labels, targets = self._remove_p_random_classes(unique_labels, targets, device)

        if self.p_extra_classes > 0:
            unique_labels, targets = self._add_p_random_classes(
                unique_labels, targets, device, original_unique_labels, _pre_batch_y
            )

        return batch_x, unique_labels, targets

    def _call_without_labels(self, examples):
        batch_size = len(examples)
        max_text_len = max(map(len, examples))
        device = examples[0][0].device

        batch_x = torch.full(
            size=[batch_size, max_text_len],
            fill_value=self.pad_token_id,
            dtype=torch.int64,
            device=device,
        )

        for i, text in enumerate(examples):
            batch_x[i, : len(text)] = text

        return batch_x, self.possible_label_ids

    def _remove_p_random_classes(self, unique_labels, targets, device):
        leave_label_mask = (
            torch.rand(unique_labels.size(0), device=unique_labels.device) > self.p_no_class
        )

        return self._remove_classes_by_mask(leave_label_mask, unique_labels, targets, device)

    @staticmethod
    def _remove_classes_by_mask(leave_class_mask, unique_labels, targets, device):
        """Note that in leave_class_mask 1 means to leave a class and 0 means to remove it"""
        if len(leave_class_mask.shape) != 1:
            raise ValueError(leave_class_mask)
        if len(unique_labels.shape) != 2:
            raise ValueError(unique_labels)
        if len(targets.shape) != 1:
            raise ValueError(targets)
        original_targets_shape = targets.shape

        # a little optimization
        if leave_class_mask.sum() == leave_class_mask.size(0):
            return unique_labels, targets

        remained_labels = unique_labels[leave_class_mask]
        # TODO: this is potentially very slow
        # you can speed it up by using the fact that .unique sorts rows
        for i, t in enumerate(targets):
            _label = unique_labels[t]
            new_index = get_index(remained_labels, _label.unsqueeze(0))

            if len(new_index) == 1:
                targets[i] = new_index.squeeze()
            elif len(new_index) == 0:
                targets[i] = torch.tensor(-1).to(device)
            else:
                raise ValueError(new_index)

        assert targets.shape == original_targets_shape
        return remained_labels, targets

    def _add_p_random_classes(
        self, unique_labels, targets, device, original_unique_labels, _pre_batch_y
    ):
        """
        We sample only from the labels absent from the original unique_labels.
        Note that unique labels may be smaller than the original unique labels,
        because we may have applied _remove_p_random_classes to it.
        """
        _possible_lids = get_difference(self.possible_label_ids, original_unique_labels)
        if _possible_lids.size(0) == 0:
            return unique_labels, targets

        extra_labels_mask = (
            torch.rand(_possible_lids.size(0), device=_possible_lids.device) < self.p_extra_classes
        )

        # a little optimization
        if extra_labels_mask.sum() == 0:
            return unique_labels, targets

        extra_labels = _possible_lids[extra_labels_mask]
        unique_labels = torch.cat([unique_labels, extra_labels], dim=0).unique(dim=0)

        no_class_mask = targets == torch.tensor(-1).to(device)
        default_index = torch.tensor(-1, device=device, dtype=torch.int64)
        targets = get_index_with_default_index(unique_labels, _pre_batch_y, default_index=default_index)
        assert targets.size(0) == _pre_batch_y.size(0)

        targets[no_class_mask] = torch.tensor(-1).to(device)

        return unique_labels, targets


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
        if not isinstance(possible_labels_ids, torch.LongTensor):
            raise ValueError(
                f"possible labels should be LongTensor, got {type(possible_labels_ids)} instead"
            )
        self.possible_labels = possible_labels_ids
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        """
        Collates examples into batches and creates targets for the ``contrastive loss''.

        The main difference with CatCollator is the unique_labels.
        In the case of test collator, it is always equal to self.possible_label_ids
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
                unique_labels: LongTensor[batch_size,] = self.possible_label_ids
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
                f"Wrong number of targets. Expected {batch_size}, got {targets.shape[0]} instead.\n"
                f"possible_label_ids: {self.possible_labels}\n _pre_batch_y: {_pre_batch_y}\n"
            )

        return batch_x, self.possible_labels, targets


def _validate_input(examples):
    if not isinstance(examples[0], tuple) and not isinstance(examples[0], torch.Tensor):
        raise ValueError(examples)

    if isinstance(examples[0], tuple):
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


# Utils


# source: https://discuss.pytorch.org/t/how-to-get-the-row-index-of-specific-values-in-tensor/28036/7
def get_index(host, target):
    assert host.shape[1] == target.shape[1]
    diff = target.unsqueeze(1) - host.unsqueeze(0)
    dsum = torch.abs(diff).sum(-1)
    loc = torch.nonzero(dsum == 0)
    return loc[:, -1]


def get_index_with_default_index(host, target, default_index):
    """Slower than get_index, because it uses for-loop,
    but it works correctly in case a host element is not an element of host"""
    assert host.shape[1] == target.shape[1]
    device = target.device
    default_index = default_index.to(device)

    indices = torch.zeros(target.size(0), device=target.device).long()

    for i, example in enumerate(target):
        sim = (example == host).all(-1)
        if any(sim):
            idx = torch.nonzero(sim)
        else:
            idx = default_index
        indices[i] = idx

    return indices


def get_difference(t1, t2):
    """Compute set difference t1 / t2"""
    sim_matrix = t1.unsqueeze(1) == t2
    sim_index = sim_matrix.all(-1).any(-1)

    difference = t1[~sim_index]
    return difference
