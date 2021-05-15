import numpy as np
import torch
import torch.utils.data
import tokenizers

from tqdm.auto import tqdm


class CatDataset(torch.utils.data.Dataset):
    """
    A Dataset object to handle turning numericalized text into count tensors.

    Args:
        texts: List[str], a list of texts
        text_tokenizer: tokenizers.Tokenizer object
        labels: List[int] or numpy array, optional - classes corresponding to the texts
        label_tokenizer: tokenizers.Tokenizer object
    """

    def __init__(self, texts, text_tokenizer, labels=None, label_tokenizer=None, max_text_len=512):
        if labels is not None and len(texts) != len(labels):
            raise ValueError("classes and texts should have the same number of elements")
        if labels is not None and label_tokenizer is None:
            raise ValueError("label_tokenizer should be provided with the `labels`")

        if isinstance(text_tokenizer, tokenizers.Tokenizer):
            text_tokenizer.enable_truncation(max_length=max_text_len)

        self.texts = texts
        self.text_tokenizer = text_tokenizer
        self.label_tokenizer = label_tokenizer
        self.labels = labels
        self.max_text_len = max_text_len

        self._text_ids = [
            self._convert_text_to_tensor(t) for t in tqdm(self.texts, desc="Preprocessing Dataset")
        ]

        self._label_ids = None
        if self.labels is not None:
            self._label_ids = [self._convert_label_to_tensor(l) for l in self.labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Turn the text at index idx into count vector

        and return it along the corresponding label (if classes were provided to the __init__)

        Returns:
            torch.Tensor[text_len,], torch.Tensor[class_len,] - count vector and (optionally) a label

            if the classes were not provided
        """

        vector = self._text_ids[idx]
        label = None if self._label_ids is None else self._label_ids[idx]

        if label is None:
            return vector
        return vector, label

    def _convert_text_to_tensor(self, text):
        """
        Tokenizes the text and makes a torch.Tensor object.

        Args:
            text: str, a text to encode

        Returns:
            torch.Tensor[text_len,]
        """
        return self.encode_via_tokenizer(text, self.text_tokenizer, max_length=self.max_text_len)

    def _convert_label_to_tensor(self, label_str):
        return self.encode_via_tokenizer(label_str, self.label_tokenizer)

    @staticmethod
    def encode_via_tokenizer(string_to_encode, tokenizer, max_length=None) -> torch.LongTensor:
        truncation = bool(max_length is not None)
        if isinstance(tokenizer, tokenizers.Tokenizer):
            ids_numpy = tokenizer.encode(string_to_encode).ids
        else:
            # the case of transformers Tokenizer or GloVeTokenizer
            # (the latter ignores max_length, but it is only applied to
            # class names so it is not supposed to used max_length!=None)
            ids_numpy = tokenizer.encode(string_to_encode, max_length=max_length, truncation=truncation)

        ids_torch = torch.LongTensor(ids_numpy)
        return ids_torch


class PreprocessedCatDatasetWCropAug(torch.utils.data.Dataset):
    """
    Assumes that the text and labels are tokenized into text_ids and label_ids.
    If the text/label is larger than max_text_len, then a random crop is performed.

    Every text_id and label_id should start and end with special tokens (e.g., CLS and SEP),
    these tokens are not affected by random crop.

    This class is motivated by the wikipedia dataset.
    It is very large in terms of the number of examples (motivates prefetching)
    and in terms of text length (requires cropping, random cropping is chosen).

    Args:
        dataset: datasets.ArrowDataset
        text_field: name of the text key in the dataset
        class_field: name of the class key in the dataset
    """

    def __init__(
        self, dataset, text_field, class_field, tokenizer, max_text_len=512, no_augmentations=False
    ):
        self.dataset = dataset
        self.text_field = text_field
        self.class_field = class_field

        self.text_tokenizer = tokenizer
        self.label_tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.no_augmentations = no_augmentations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # TODO: figure out why idx can be np.int64 when using exp replay
        idx = int(idx)

        item = self.dataset[idx]

        text_ids = torch.tensor(item[self.text_field])
        label_ids = torch.tensor(item[self.class_field])

        text_ids = self.maybe_crop(text_ids)
        label_ids = self.maybe_crop(label_ids)

        return text_ids, label_ids

    def maybe_crop(self, text_ids):
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


class SampleConcatSubset(torch.utils.data.Dataset):
    # https://github.com/googleinterns/new-semantic-parsing
    # Copyright 2020 Google LLC
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # =============================================================================
    # modified in such a way that sample_probability is the probability of sampling
    # an element from sample_dataset
    def __init__(self, concat_dataset, sample_dataset, sample_probability):
        """Implements concatenation of concat_dataset and a subset of sample_dataset.


        Args:
            concat_dataset: torch Dataset
            sample_dataset: torch Dataset
            sample_probability: float, 0 < sample_probability < 1
        """

        self._concat_dataset = concat_dataset
        self._sample_dataset = sample_dataset
        self._sample_probability = sample_probability
        self.text_tokenizer = self.label_tokenizer = concat_dataset.text_tokenizer
        self.max_text_len = concat_dataset.max_text_len

        p = sample_probability
        self._sample_dataset_amount = int(p * len(concat_dataset) / (1 - p))

        if self._sample_dataset_amount > len(self._sample_dataset):
            raise NotImplementedError()

        self._data = None
        self.resample()

    def __len__(self):
        if self._data is None:
            raise RuntimeError("Call .resample first")
        return len(self._data)

    def resample(self):
        """Resamples from sample_dataset, inplace

        Returns:
            torch Dataset
        """
        subset = self.make_subset(self._sample_dataset, self._sample_dataset_amount)
        self._data = torch.utils.data.ConcatDataset([self._concat_dataset, subset])
        return self._data

    def __getitem__(self, item):
        return self._data[item]

    @staticmethod
    def make_subset(dataset, subset_size):
        """Makes torch Subset by randomly sampling indices from dataset
        Args:
            dataset: torch Dataset
            subset_size: int, size of the final dataset
        """
        _subset_ids = np.random.permutation(len(dataset))[:subset_size]
        _subset = torch.utils.data.Subset(dataset, indices=_subset_ids)
        return _subset
