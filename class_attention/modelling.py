import torch
import torch.nn as nn
import transformers


class PreTrainedEmbeddingEncoder(nn.Module):
    def __init__(self, embedding_matrix, word2id):
        super().__init__()

        self.vocab_size = embedding_matrix.shape[0]
        self.emb_size = embedding_matrix.shape[1]
        self.word2id = word2id

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)  # [batch_size, seq_len, hidden]

        # we keep the dim to have the same interface as Transformer does
        # sum is used for aggregation, because PAD is a zero vector
        emb = torch.sum(emb, dim=1, keepdim=True)  # [batch_size, 1, hidden]

        # Transformers return tuples, and we pretend to be one
        return (emb,)


def get_output_dim(model: [transformers.PreTrainedModel, PreTrainedEmbeddingEncoder]):
    if isinstance(model, PreTrainedEmbeddingEncoder):
        return model.emb_size

    if isinstance(model.config, transformers.PretrainedConfig):
        return model.config.hidden_size

    raise ValueError(type(model))


class ClassTransformerBlock(nn.Module):
    def __init__(self, n_layers, hidden, ffn_hidden, n_heads=1, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ClassTransformerLayer(
                    hidden=hidden,
                    ffn_hidden=ffn_hidden,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, text_emb, class_emb):
        """

        NOTE: Multihead attention inside ClassTransformerLayer expects
        a tensor of shape [seq, batch, hidden]
        in our case batch always == 1
        because we interact each example from text_emb batch dimension
        with each class in the class_emb batch dimension
        and they effectively serve as seq dimensions and not batch dimensions

        Args:
            text_emb: FloatTensor[batch, 1, hidden] or FloatTensor[batch, hidden]
            class_emb: FloatTensor[n_classes, 1, hidden] or FloatTensor[n_classes, hidden]

        Returns:
            FloatTensor of same shape as text_emb
        """
        should_squeeze = False
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)
            should_squeeze = True
        if class_emb.dim() == 2:
            class_emb = class_emb.unsqueeze(1)

        for layer in self.layers:
            text_emb = layer(text_emb=text_emb, class_emb=class_emb)

        if should_squeeze:
            text_emb = text_emb.squeeze(1)

        return text_emb


class ClassTransformerLayer(nn.Module):
    """Pre-norm version of transformer block with cross-attention

    Why pre-norm: https://openreview.net/pdf?id=B1x8anVFPr
    """

    def __init__(self, hidden, ffn_hidden, n_heads=1, dropout=0.0):
        super().__init__()
        self.cross_attention = nn.modules.transformer.MultiheadAttention(
            embed_dim=hidden,
            num_heads=n_heads,
            dropout=dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_hidden),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ffn_hidden, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, text_emb, class_emb):
        """Note that text_emb and class_emb should have an empty dimension at index 1 for efficient computation.

        Args:
            text_emb: torch.FloatTensor[batch_size, 1, hidden]
            class_emb: torch.FloatTensor[n_classes, 1, hidden]

        Returns:

        """
        if text_emb.dim() != 3:
            raise ValueError(text_emb.shape)
        if class_emb.dim() != 3:
            raise ValueError(class_emb.shape)

        residual = text_emb

        x = self.norm1(text_emb)
        # multi-head attention returns tuple(attn_output, attn_output_weights(
        x, _ = self.cross_attention(query=x, key=class_emb, value=class_emb)
        x = x + residual

        residual = x

        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x
