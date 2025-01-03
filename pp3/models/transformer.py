"""A class for a transformer predictor model."""
import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from pp3.utils.constants import MAX_SEQ_LEN


class PositionalEncoding(nn.Module):
    """A class for positional encoding for Transformers."""

    def __init__(
            self,
            emb_size: int,
            dropout: float,
            maxlen: int = MAX_SEQ_LEN
    ) -> None:
        super(PositionalEncoding, self).__init__()

        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Assumes embeddings are (batch_size, sequence_length, emb_size)
        return self.dropout(embeddings + self.pos_embedding[:, :embeddings.size(1), :])


class Transformer(nn.Module):
    """A transformer predictor model."""

    def __init__(
            self,
            vocab_size: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float = 0.0,
            nhead: int = 4
    ) -> None:
        """Initialize the model.

        :param vocab_size: The size of the vocabulary (i.e., number of amino acids).
        :param hidden_dim: The dimensionality of the hidden layers.
        :param num_layers: The number of layers.
        :param dropout: The dropout rate.
        :param nhead: The number of heads in the multihead attention models.
        """
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=0)

        self.positional_encoding = PositionalEncoding(
            emb_size=hidden_dim,
            dropout=dropout
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )

        self.model = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

    def forward(
            self,
            embeddings: torch.Tensor,
            coords: torch.Tensor,
            padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Runs the Transformer model on the data.

        :param embeddings: A tensor containing an embedding.
        :param coords: A tensor containing the coordinates.
        :param padding_mask: A tensor containing a padding mask.
        :return: A tensor containing the model's updated embedding.
        """
        # Embed the amino acids
        embeddings = self.embedding(embeddings)

        # Apply positional encodings
        embeddings = self.positional_encoding(embeddings)

        # Set up padding mask
        src_key_padding_mask = (1 - padding_mask).bool()

        # Apply Transformer encoder layers
        embeddings = self.model(embeddings, src_key_padding_mask=src_key_padding_mask)

        return embeddings
