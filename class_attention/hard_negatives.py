from math import ceil, floor

import numpy as np
import torch
import torch.utils.data

import datasets
import datasets.search


class HardNegativeDatasetWAug(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
        text_ids_field: str,
        label_ids_field: str,
        index_field: str,
        collator,  # CatCollator
        percent_hard: float = 0.5,
        max_text_len=512,
        no_augmentations=False,
    ):
        """Samples batch_size * (1 - percent_hard) randomly and fill the rest of the batch with hard examples
        where by hard we mean similarity in index_field vector space.

        Args:
            dataset: ArrowDataset with FAISS index
            batch_size:
            percent_hard: proportion of hard samples in the batch
            text_ids_field: name of the field in the dataset containing numericalized text
            label_ids_field: name of the field in the dataset containing numericalized label name
            index_field: the field in the dataset that is FAISS indexed
            collator: CatCollator
        """
        # nothing happens with the dataset inside super().__init__
        super().__init__()

        self.dataset = dataset
        self.presampler = torch.utils.data.RandomSampler(dataset)

        self.batch_size = batch_size
        self.percent_hard = percent_hard
        self.text_ids_field = text_ids_field
        self.label_ids_field = label_ids_field
        self.index_field = index_field
        self.collator = collator
        self.max_text_len = max_text_len
        self.no_augmentations = no_augmentations

        # fmt: off
        if self.presample_batch_size() == self.batch_size:
            raise ValueError(f"percent_hard {self.percent_hard} is too small for batch_size {self.batch_size}. "
                             f"It would yield batches without hard samples and HardNegativeSampler should not be used "
                             f"in such cases for performance reasons.")
        # fmt: on

        self._len = int(len(self.dataset) // self.presample_batch_size())

    def __len__(self):
        # actual len depends on `percent_hard` and we are planning to change it over time
        return self._len

    def presample_batch_size(self):
        return int(self.batch_size * (1 - self.percent_hard))

    def _presample(self):
        batch = []
        for idx in self.presampler:
            batch.append(idx)
            if len(batch) == self.presample_batch_size():
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __iter__(self):
        presample_idx = next(self._presample())  # sample R examples randomly
        n_presampled = len(presample_idx)  # R, may vary

        if n_presampled == self.batch_size:
            assert NotImplementedError()
            # should we return ids or items?
            # return presample_idx

        random_samples = self.dataset[presample_idx]  # returns dict of lists

        n_hard_samples = self.batch_size - n_presampled  # H = B - R
        hard_samples_batch = self.find_hard_samples_for_random_samples(
            random_samples=random_samples,
            n_random_samples=n_presampled,
            n_hard_samples=n_hard_samples,
        )

        random_samples_list = self._dataset_dict_to_tuples(random_samples)
        hard_samples_list = self._flatten_batched_knn_results(hard_samples_batch, max_len=n_hard_samples)

        all_samples_list = random_samples_list + hard_samples_list
        all_samples_list = self._tensorify_examples(all_samples_list)
        all_samples_list = self._crop_tensorified_examples(all_samples_list)

        # collate into tensors
        batch = self.collator(all_samples_list)

        yield batch

    def find_hard_samples_for_random_samples(self, random_samples, n_random_samples, n_hard_samples):
        # it is slightly easier and way less buggy to just accept n_random_samples as an argument
        # than to compute it form random_samples

        # a minimum integer k neighbors to find enough hard samples given n_random_samples
        knn_k = ceil(n_random_samples / n_hard_samples)

        # number of randomly sampled examples that will receive hard samples
        # n_queries < n_random_samples to minimize the number of KNN calls
        n_queries = max(1, floor(n_hard_samples / knn_k))

        # FAISS that is used for KNN requires float32 and default numpy float is 64-bit
        query_embeds = np.array(random_samples[self.index_field][:n_queries], dtype=np.float32)

        hard_samples_batch = self.dataset.get_nearest_examples_batch(
            index_name=self.index_field, queries=query_embeds, k=knn_k
        )
        return hard_samples_batch

    def _dataset_dict_to_tuples(self, dataset_dict):
        """Creates a tuple of examples list[tuple(text_ids, label_ids)]

        this format is expected in the collator function
        """
        return list(zip(dataset_dict[self.text_ids_field], dataset_dict[self.label_ids_field]))

    def _flatten_batched_knn_results(
        self, results: datasets.search.BatchedNearestExamplesResults, max_len=None
    ):
        batch_of_scores, batch_of_examples = results

        all_examples = []
        for examples_dict in batch_of_examples:
            # we start at the element 1, because element 0 is exactly a text from random_samples
            examples_list = self._dataset_dict_to_tuples(examples_dict)
            all_examples.extend(examples_list)

        if max_len is not None:
            all_examples = all_examples[:max_len]

        return all_examples

    def _tensorify_examples(self, examples):
        return [(torch.tensor(e[0], dtype=torch.int64), torch.tensor(e[1], dtype=torch.int64)) for e in examples]

    def _crop_tensorified_examples(self, examples):
        return [(self.maybe_crop(e[0]), self.maybe_crop(e[1])) for e in examples]

    def maybe_crop(self, text_ids):
        # initially copied from PreprocessedCatDatasetWCropAug
        assert len(text_ids.shape) == 1
        if len(text_ids) <= self.max_text_len:
            return text_ids

        # minus CLS and SEP tokens
        delta = len(text_ids) - self.max_text_len

        if self.no_augmentations:
            random_start = torch.tensor(0, dtype=torch.int64)
            end = random_start + self.max_text_len - 2
        else:
            # torch.randint includes start=1 but excludes end=delta + 1
            random_start = torch.randint(0, delta + 1, size=(1,))[0]
            end = random_start + self.max_text_len - 2

        cropped_sequence = torch.cat(
            [
                text_ids[0].unsqueeze(0),  # CLS
                text_ids[1 + random_start : 1 + end],
                text_ids[-1].unsqueeze(0),  # SEP
            ]
        )
        return cropped_sequence
