import torch
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

    def __init__(self, texts, text_tokenizer, labels=None, label_tokenizer=None):
        if labels is not None and len(texts) != len(labels):
            raise ValueError("classes and texts should have the same number of elements")
        if labels is not None and label_tokenizer is None:
            raise ValueError("label_tokenizer should be provided with teh classes")

        self.texts = texts
        self.text_tokenizer = text_tokenizer
        self.label_tokenizer = label_tokenizer
        self.labels = labels

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
        return self.encode_via_tokenizer(text, self.text_tokenizer)

    def _convert_label_to_tensor(self, label_str):
        return self.encode_via_tokenizer(label_str, self.label_tokenizer)

    @staticmethod
    def encode_via_tokenizer(string_to_encode, tokenizer) -> torch.LongTensor:
        # TODO: use return_tensors='pt' to get the input_dict?
        # TODO: if yes, you also need to update collator
        ids_numpy = tokenizer.encode(string_to_encode)
        if isinstance(ids_numpy, tokenizers.Encoding):
            ids_numpy = ids_numpy.ids

        ids_torch = torch.LongTensor(ids_numpy)
        return ids_torch
