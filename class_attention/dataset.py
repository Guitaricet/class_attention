import torch
import torch.utils.data
import tokenizers

from tqdm.auto import tqdm


import class_attention as cat


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
            raise ValueError("label_tokenizer should be provided with teh classes")

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
        return self.encode_via_tokenizer(text, self.text_tokenizer, max_length=self.max_text_len, truncation=True)

    def _convert_label_to_tensor(self, label_str):
        return self.encode_via_tokenizer(label_str, self.label_tokenizer)

    @staticmethod
    def encode_via_tokenizer(string_to_encode, tokenizer, max_length=None) -> torch.LongTensor:
        if isinstance(tokenizer, tokenizers.Tokenizer):
            ids_numpy = tokenizer.encode(string_to_encode).ids
        else:
            # the case of transformers Tokenizer or GloVeTokenizer
            # (the latter ignores max_length, but it is only applied to
            # class names so it is not supposed to used max_length!=None)
            ids_numpy = tokenizer.encode(string_to_encode, max_length=max_length)

        ids_torch = torch.LongTensor(ids_numpy)
        return ids_torch
