import math

import torch
import transformers
import wandb
from torch import nn as nn

import class_attention as cat
from class_attention.modelling import get_output_dim, ClassTransformerBlock


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
                n_projection_layers,
                attention_type,
                bahdanau_layers,
                temperature,
        """
        super().__init__()

        self.txt_encoder = txt_encoder
        self.cls_encoder = cls_encoder
        txt_encoder_h = get_output_dim(self.txt_encoder)
        cls_encoder_h = get_output_dim(self.cls_encoder)

        self.validate_kwargs(kwargs)
        self.kwargs = kwargs or dict()

        self.txt_out = nn.Identity()
        self.cls_out = nn.Identity()
        self.class_transformer = None

        dropout = kwargs.get("dropout", 0.0)
        self.dropout_x = nn.Dropout(dropout ** 2)
        self.dropout_c = nn.Dropout(dropout ** 2)
        self.dropout_logits = nn.Dropout(dropout)

        n_projection_layers = self.kwargs.get("n_projection_layers", None)
        if n_projection_layers is not None and n_projection_layers > 0:
            hidden_size = self.kwargs["hidden_size"]  # guaranteed in validate_kwargs
            self.txt_out = cat.modelling_utils.make_mlp(
                n_layers=n_projection_layers,
                input_size=txt_encoder_h,
                hidden_size=hidden_size,
                use_bias=kwargs.get("use_bias", True),
                dropout=dropout,
            )
            self.cls_out = cat.modelling_utils.make_mlp(
                n_layers=n_projection_layers,
                input_size=cls_encoder_h,
                hidden_size=hidden_size,
                use_bias=kwargs.get("use_bias", True),
                dropout=dropout,
            )

        cross_attention_layers = self.kwargs.get("cross_attention_layers", None)
        if cross_attention_layers is not None and cross_attention_layers > 0:
            hidden_size = self.kwargs["hidden_size"]  # guaranteed in validate_kwargs
            n_heads = self.kwargs["cross_attention_heads"]  # guaranteed in validate_kwargs
            self.class_transformer = ClassTransformerBlock(
                n_layers=cross_attention_layers,
                hidden=hidden_size,
                ffn_hidden=4 * hidden_size,
                n_heads=n_heads,
            )

        # make bahdanau attention
        if kwargs.get("attention_type") == "bahdanau":
            self.bahdanau_network = self.make_bahdanau_attention(kwargs)

        initial_temperature = kwargs.get("temperature", 0.0)
        self.temperature = nn.Parameter(
            torch.tensor(initial_temperature, requires_grad=kwargs.get("learn_temperature", False))
        )

    @classmethod
    def from_kwargs(cls, **kwargs):
        if kwargs.get("debug", False):
            raise ValueError("Debug mode is not supported in .from_kwargs")

        text_encoder = transformers.AutoModel.from_pretrained(kwargs["model"])
        label_encoder = cat.training_utils.make_label_encoder(
            model_name_or_path=kwargs["model"],
            glove=kwargs.get("glove"),
        )

        return cat.ClassAttentionModel(
            text_encoder,
            label_encoder,
            **kwargs,
        )

    @classmethod
    def get_state_dict_from_checkpoint(cls, checkpoint_path, map_location="cpu"):
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        model_state_dict = state_dict["model_state_dict"]
        return model_state_dict

    def load_state_dict_from_checkpoint(self, checkpoint_path):
        model_state_dict = self.get_state_dict_from_checkpoint(checkpoint_path)
        self.load_state_dict(model_state_dict, strict=True)

    def save(self, file_path, optimizer=None, **kwargs):
        """Saves model, model args and additional stuff to a file

        Args:
            file_path: path to a .torch file
            optimizer: saves optimizer state_dict if provided
            **kwargs: additional artuments
        """

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_args": self.kwargs,
            **kwargs,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, file_path)

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
        text_input, labels_input = cat.modelling_utils.maybe_format_inputs(
            text_input, labels_input
        )
        cat.modelling_utils.validate_inputs(text_input, labels_input)

        h_x = self.txt_encoder(**text_input)  # some tuple
        h_x = h_x[0]  # FloatTensor[bs, text_seq_len, hidden]
        h_x = h_x[:, 0]  # get CLS token representations, FloatTensor[bs, hidden]

        h_c = self.cls_encoder(**labels_input)  # some tuple
        h_c = h_c[0]  # FloatTensor[n_classes, class_seq_len, hidden]
        h_c = h_c[:, 0]  # get CLS token representations, FloatTensor[n_classes, hidden]

        # maybe projections
        h_x = self.txt_out(h_x)
        h_c = self.cls_out(h_c)

        h_x = self.dropout_x(h_x)
        h_c = self.dropout_c(h_c)

        if self.class_transformer is not None:
            h_x = self.class_transformer(text_emb=h_x, class_emb=h_c)

        if self.kwargs.get("remove_n_lowest_pc"):
            h_c = cat.modelling_utils.remove_smallest_princpial_component(
                h_c, remove_n=self.kwargs.get("remove_n_lowest_pc")
            )

        # make all class embeddings to have the same Euclidean norm
        if self.kwargs.get("normalize_txt"):
            h_x = cat.modelling_utils.normalize_embeds(h_x)
        if self.kwargs.get("normalize_cls"):
            h_c = cat.modelling_utils.normalize_embeds(h_c)

        # either compute a dot product or a Bahdanau-like attention score
        if self.kwargs.get("attention_type") == "bahdanau":
            logits = self._bahdanau_attention_fn(ht=h_x, hc=h_c)
        else:
            logits = h_x @ h_c.T  # [bs, n_classes]

        assert logits.shape == (h_x.shape[0], h_c.shape[0]), logits.shape

        # the scaling is extremely important if normalization is not used
        if self.kwargs.get("scale_attention"):
            logits = logits / (logits.size(-1) ** 0.5)

        # apply temperature, clamp it if it is trainable
        if self.kwargs.get("learn_temperature"):
            self.temperature.data.clamp_(math.log(1e-3), math.log(1e3))

        logits *= torch.exp(self.temperature)
        logits = self.dropout_logits(logits)

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

        n_projection_layers = kwargs.get("n_projection_layers")
        if n_projection_layers is not None and n_projection_layers < 1:
            raise ValueError(n_projection_layers)

        cross_attention_layers = kwargs.get("cross_attention_layers")
        if cross_attention_layers is not None and cross_attention_layers < 1:
            raise ValueError(cross_attention_layers)

        hidden_size = kwargs.get("hidden_size")
        cross_attention_heads = kwargs.get("cross_attention_heads")

        if n_projection_layers is not None:
            if n_projection_layers > 0 and hidden_size is None:
                raise ValueError("hidden size should be specified with n_projection_layers")

        if cross_attention_layers is not None and cross_attention_layers > 0:
            if hidden_size is None:
                raise ValueError("hidden size should be specified with cross_attention_layers")
            if cross_attention_heads is None:
                raise ValueError(
                    "--cross-attention-heads should be specified with cross_attention_layers"
                )
            if hidden_size % cross_attention_heads != 0:
                raise ValueError("hiden size should be divisible by --cross-attention-heads")

        if kwargs.get("share_txt_cls_network_params") and kwargs.get("freeze_cls_network"):
            raise ValueError(
                "Parameters are shared, but class network is not trained. "
                "Both BERT bodies not being updated"
            )

    def make_bahdanau_attention(self, kwargs):
        # if we have a projection, our input size may be different
        if kwargs.get("n_projection_layers"):
            attention_size = 2 * kwargs.get("hidden_size")
        else:
            txt_encoder_h = get_output_dim(self.txt_encoder)
            cls_encoder_h = get_output_dim(self.cls_encoder)
            attention_size = txt_encoder_h + cls_encoder_h

        n_attention_layers = kwargs.get("bahdanau_layers", 1)
        bahdanau_network = cat.modelling_utils.make_mlp(
            n_layers=n_attention_layers,
            input_size=attention_size,
            hidden_size=attention_size // 2,
            output_size=1,
            use_bias=kwargs.get("use_bias", True),
        )
        return bahdanau_network

    def get_trainable_parameters(self):
        conditions = []
        if self.kwargs.get("freeze_projections"):
            conditions.append(self._is_not_proj)
        if self.kwargs.get("freeze_cls_network"):
            conditions.append(self._is_not_cls_network)
        if self.kwargs.get("freeze_cls_embeddings"):
            conditions.append(self._is_not_cls_network_embeddings)

        if not self.kwargs.get("learn_temperature"):
            conditions.append(self._is_not_temperature)

        return (
            param
            for name, param in self.named_parameters()
            if all(cond(name) for cond in conditions)
        )

    def _bahdanau_attention_fn(self, ht, hc):
        return cat.modelling_utils.bahdanau_fn(key=ht, query=hc, scoring_fn=self.bahdanau_network)

    @staticmethod
    def _is_not_proj(param_name):
        return "txt_out" not in param_name and "cls_out" not in param_name

    @staticmethod
    def _is_not_cls_network(param_name):
        return "cls_encoder" not in param_name

    @staticmethod
    def _is_not_temperature(param_name):
        return "temperature" not in param_name

    @staticmethod
    def _is_not_cls_network_embeddings(param_name):
        return "cls_encoder.embeddings" not in param_name