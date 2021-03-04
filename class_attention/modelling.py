import math

import torch
import torch.nn as nn
import wandb

from class_attention import modelling_utils


class ClassAttentionModel(nn.Module):
    def __init__(self, txt_encoder, cls_encoder, **kwargs):
        """
        Args:
            txt_encoder: transformer encoder for text
            cls_encoder: transformer encoder for class names
            kwargs: optional arguments
                hidden_size,
                normalize_txt,
                normalize_cls,
                scale_attention,
                remove_n_lowest_pc,
                use_n_projection_layers,
                attention_type,
                bahdanau_layers,
                temperature,
        """
        super().__init__()

        self.txt_encoder = txt_encoder
        self.cls_encoder = cls_encoder
        txt_encoder_h = modelling_utils.get_output_dim(self.txt_encoder)
        cls_encoder_h = modelling_utils.get_output_dim(self.cls_encoder)

        self.validate_kwargs(kwargs)
        self.kwargs = kwargs or dict()

        self.txt_out = nn.Identity()
        self.cls_out = nn.Identity()

        use_n_projection_layers = self.kwargs.get("use_n_projection_layers", None)
        if use_n_projection_layers is not None and use_n_projection_layers > 0:
            hidden_size = self.kwargs["hidden_size"]  # guaranteed in validate_kwargs
            self.txt_out = modelling_utils.make_mlp(
                n_layers=use_n_projection_layers,
                input_size=txt_encoder_h,
                hidden_size=hidden_size,
            )
            self.cls_out = modelling_utils.make_mlp(
                n_layers=use_n_projection_layers,
                input_size=cls_encoder_h,
                hidden_size=hidden_size,
            )

        # make bahdanau attention
        if kwargs.get("attention_type") == "bahdanau":
            self.bahdanau_network = self.make_bahdanau_attention(kwargs)

        initial_temperature = kwargs.get("temperature", 0.0)
        self.temperature = nn.Parameter(
            torch.tensor(initial_temperature, requires_grad=kwargs.get("learn_temperature", False))
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

        h_c = self.cls_encoder(**labels_input)  # some tuple
        h_c = h_c[0]  # FloatTensor[n_classes, class_seq_len, hidden]
        h_c = h_c[:, 0]  # get CLS token representations, FloatTensor[n_classes, hidden]

        # maybe projections
        h_x = self.txt_out(h_x)
        h_c = self.cls_out(h_c)

        if self.kwargs.get("remove_n_lowest_pc"):
            h_c = modelling_utils.remove_smallest_princpial_component(
                h_c, remove_n=self.kwargs.get("remove_n_lowest_pc")
            )

        # make all class embeddings to have the same Euclidean norm
        if self.kwargs.get("normalize_txt"):
            h_x = modelling_utils.normalize_embeds(h_x)
        if self.kwargs.get("normalize_cls"):
            h_c = modelling_utils.normalize_embeds(h_c)

        # the scaling is extremely important if normalization is not used
        scaling = 1
        if self.kwargs.get("scale_attention"):
            scaling = h_c.size(-1) ** 0.5

        # either compute a dot product or a Bahdanau-like attention score
        if self.kwargs.get("attention_type") == "bahdanau":
            logits = self._bahdanau_attention_fn(h_x, h_c) / scaling
        else:
            logits = (h_x @ h_c.T) / scaling  # [bs, n_classes]

        assert logits.shape == (h_x.shape[0], h_c.shape[0]), logits.shape

        # apply temperature, clamp it if it is trainable
        if self.kwargs.get("learn_temperature"):
            self.temperature.data.clamp_(math.log(1e-3), math.log(1e3))

        logits *= torch.exp(self.temperature)

        # fmt: off
        if wandb.run is not None and self.training:
            wandb.log({
                "train/temperature": self.temperature.data,
                "train/exp_temperature": torch.exp(self.temperature.data),
            })
        # fmt: on

        return logits

    @staticmethod
    def validate_kwargs(kwargs):
        if kwargs is None:
            return

        use_n_projection_layers = kwargs.get("use_n_projection_layers")
        if use_n_projection_layers is not None and use_n_projection_layers < 1:
            raise ValueError(use_n_projection_layers)

        hidden_size = kwargs.get("hidden_size")

        if use_n_projection_layers is not None:
            if use_n_projection_layers < 0:
                raise ValueError(use_n_projection_layers)
            if use_n_projection_layers > 0 and hidden_size is None:
                raise ValueError("hidden size should be specified with use_n_projection_layers")

        if kwargs.get("share_txt_cls_network_params") and kwargs.get("freeze_cls_network"):
            raise ValueError(
                "Parameters are shared, but class network is not trained. "
                "Both BERT bodies not being updated"
            )

    def make_bahdanau_attention(self, kwargs):
        # if we have a projection, our input size may be different
        if kwargs.get("use_n_projection_layers"):
            attention_size = 2 * kwargs.get("hidden_size")
        else:
            txt_encoder_h = modelling_utils.get_output_dim(self.txt_encoder)
            cls_encoder_h = modelling_utils.get_output_dim(self.cls_encoder)
            attention_size = txt_encoder_h + cls_encoder_h

        n_attention_layers = kwargs.get("bahdanau_layers", 1)
        bahdanau_network = modelling_utils.make_mlp(
            n_layers=n_attention_layers,
            input_size=attention_size,
            hidden_size=attention_size // 2,
            output_size=1,
        )
        return bahdanau_network

    def get_trainable_parameters(self):
        conditions = []
        if self.kwargs.get("freeze_projections"):
            conditions.append(self._is_not_proj)
        if self.kwargs.get("freeze_cls_network"):
            conditions.append(self._is_not_cls_network)

        if not self.kwargs.get("learn_temperature"):
            conditions.append(self._is_not_temperature)

        return (
            param
            for name, param in self.named_parameters()
            if all(cond(name) for cond in conditions)
        )

    def _bahdanau_attention_fn(self, ht, hc):
        batch_size, t_hidden = ht.shape
        n_classes, c_hidden = hc.shape

        ht_repeated = ht.repeat([1, n_classes]).view(batch_size * n_classes, t_hidden)
        hc_repeated = hc.repeat([batch_size, 1])

        hx = torch.cat([ht_repeated, hc_repeated], dim=-1)  # the order is important
        # pair i+j means i-th example and j-th class
        attn_scores = self.bahdanau_network(hx)  # [batch_size * n_classes, 1]

        # reshaped in such a way that batch_size is actually a batch dimension
        # and n_classes is a class dimension
        attn_scores = attn_scores.view(batch_size, n_classes)
        return attn_scores

    @staticmethod
    def _is_not_proj(param_name):
        return "txt_out" not in param_name and "cls_out" not in param_name

    @staticmethod
    def _is_not_cls_network(param_name):
        return "cls_encoder" not in param_name

    @staticmethod
    def _is_not_temperature(param_name):
        return "temperature" not in param_name

