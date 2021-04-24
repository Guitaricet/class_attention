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
    """

    def __init__(self, text_ids, label_ids, tokenizer, max_text_len=512):
        self.text_ids = text_ids
        self.label_ids = label_ids
        self.text_tokenizer = tokenizer
        self.label_tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        text_ids = torch.tensor(self.text_ids[idx])
        label_ids = torch.tensor(self.label_ids[idx])

        text_ids = self.maybe_crop(text_ids)
        label_ids = self.maybe_crop(label_ids)

        return text_ids, label_ids

    def maybe_crop(self, text_ids):
        assert len(text_ids.shape) == 1
        if len(text_ids) <= self.max_text_len:
            return text_ids

        # minus CLS and SEP tokens
        delta = len(text_ids) - self.max_text_len

        # torch.randint includes start=1 but excludes end=delta + 1
        random_start = torch.randint(0, delta + 1, size=(1,))[0]
        end = random_start + self.max_text_len - 2

        cropped_sequence = torch.cat(
            [
                text_ids[0].unsqueeze(0),  # CLS
                text_ids[1 + random_start: 1 + end],
                text_ids[-1].unsqueeze(0),  # SEP
            ]
        )
        return cropped_sequence
