"""A class for an RNN predictor model."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    """An RNN predictor model."""

    def __init__(
            self,
            vocab_size: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float = 0.0
    ) -> None:
        """Initialize the model.

        :param vocab_size: The size of the vocabulary (i.e., number of amino acids).
        :param hidden_dim: The dimensionality of the hidden layers.
        :param num_layers: The number of layers.
        :param dropout: The dropout rate.
        """
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=0)

        self.model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
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

        # Get sequence lengths
        seq_lengths = torch.sum(padding_mask, dim=1)

        # Sort the sequences by sequence length
        seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
        embeddings = embeddings[perm_idx]

        # Pack the sequences
        embeddings = pack_padded_sequence(embeddings, seq_lengths.cpu(), batch_first=True)

        # Apply RNN
        embeddings, _ = self.model(embeddings)

        # Unpack the sequences
        embeddings, _ = pad_packed_sequence(embeddings, batch_first=True)

        # Unsort the sequences
        _, unperm_idx = perm_idx.sort(dim=0)
        embeddings = embeddings[unperm_idx]

        return embeddings
