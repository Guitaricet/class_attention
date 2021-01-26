import torch
import torch.nn as nn
import torch.nn.functional as F

from class_attention import modelling_utils


class ClassAttentionModel(nn.Module):
    def __init__(self, txt_encoder, cls_encoder, hidden_size):
        super().__init__()

        self.txt_encoder = txt_encoder
        self.cls_encoder = cls_encoder

        txt_encoder_h = modelling_utils.get_output_dim(txt_encoder)
        self.txt_out = nn.Linear(txt_encoder_h, hidden_size)

        cls_encoder_h = modelling_utils.get_output_dim(cls_encoder)
        self.cls_out = nn.Linear(cls_encoder_h, hidden_size)

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

        h_c = self.cls_encoder(**labels_input)  # some tuple
        h_c = h_c[0]  # FloatTensor[n_classes, class_seq_len, hidden]

        h_c, _ = torch.max(h_c, dim=1)  # [n_classes, hidden]

        # attention map
        h_x = self.txt_out(h_x)
        h_c = self.cls_out(h_c)

        # the scaling is extremely important
        scaling = h_c.size(-1) ** 0.5
        logits = (h_x @ h_c.T) / scaling  # [bs, n_classes]

        return logits
