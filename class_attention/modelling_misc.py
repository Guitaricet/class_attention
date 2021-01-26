"""
Miscellaneous models and methods that are probably only used for testing purposes and not main experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from class_attention import modelling_utils


class FcClassAttentionModel(nn.Module):
    """
    Class attention that uses fully-connected networks to encode both text and label
    """
    def __init__(self, text_vocab_size, n_classes, hidden_size):
        super().__init__()

        self.x_enc = nn.Sequential(nn.EmbeddingBag(text_vocab_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size))

        self.c_enc = nn.Sequential(nn.EmbeddingBag(n_classes, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size))

    def forward(self, text_input, labels_input):
        text_input, labels_input = modelling_utils.maybe_format_inputs(text_input, labels_input)
        modelling_utils.validate_inputs(text_input, labels_input)

        x = text_input['input_ids']
        c = labels_input['input_ids']

        h_x = self.x_enc(x)
        h_c = self.c_enc(c)

        scaling = h_c.size(-1) ** 0.5
        # # + bias
        logits = (h_x @ h_c.T) / scaling  # [bs, n_classes]

        return logits


class EmbClassAttentionModel(nn.Module):
    """
    Class atteniton with complex text encoder (e.g., BERT) and FCN label encoder
    """

    def __init__(self, txt_encoder, n_classes, hidden_size, use_txt_out=True):
        super().__init__()

        txt_encoder_h = modelling_utils.get_output_dim(txt_encoder)

        self.use_txt_out = use_txt_out
        if not use_txt_out:
            if hidden_size != txt_encoder_h:
                raise ValueError(
                    "hidden_size should be equal to the hidden size of the text encoder if use_txt_out=False")

        self.txt_encoder = txt_encoder

        if use_txt_out:
            self.txt_out = nn.Linear(txt_encoder_h, hidden_size)

        self.cls_encoder = nn.Sequential(
            nn.EmbeddingBag(n_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, text_input, labels_input):
        """
        Compute logits for input (input_dict,) corresponding to the classes (classes_dict)

        Optionally, you can provide additional keys in either input_dict or classes_dict
        Specifically, attention_mask, head_mask and inputs_embeds
        Howerver, one should not provide output_attentions and output_hidden_states

        Args:
            text_input: dict with key input_ids
                input_ids: LongTensor[batch_size, text_seq_len], input to the text network
            labels_input: dict with key input_ids
                input_ids: LongTensor[n_classes, class_seq_len], a list of possible classes, each class described via text
        """
        text_input, labels_input = modelling_utils.maybe_format_inputs(text_input, labels_input)
        modelling_utils.validate_inputs(text_input, labels_input)

        h_x = self.txt_encoder(**text_input)  # some tuple
        h_x = h_x[0]  # FloatTensor[bs, text_seq_len, hidden]
        h_x = h_x[:, 0]  # get CLS token representations, FloatTensor[bs, hidden]
        #         h_x, _ = torch.max(h_x, axis=1)

        c = labels_input['input_ids']
        h_c = self.cls_encoder(c)  # FloatTensor[n_classes, hidden]

        if self.use_txt_out:
            h_x = self.txt_out(h_x)

        scaling = h_c.size(-1) ** 0.5
        logits = (h_x @ h_c.T) / scaling  # [bs, n_classes]
        return logits


class BertMockModel(nn.Module):
    """
    A model that imitates BERT, but actually just an FCN network.

    Used to determine that FcnClassAttentionModel works exactly like a ClassAttentionModel
    """

    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.x_enc = nn.Sequential(nn.EmbeddingBag(vocab_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size))

        class MockConfig:
            def __init__(self, hidden_size):
                self.hidden_size = hidden_size

        self.config = MockConfig(hidden_size)

    def forward(self, input_ids):
        x = input_ids
        h = self.x_enc(x)
        h = h.unsqueeze(1)
        return (h,)  # following BERT interface
