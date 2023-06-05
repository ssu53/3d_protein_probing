"""A class for a transformer predictor model."""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Transformer(nn.Module):
    """A transformer predictor model."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        interaction_hidden_dims: int = 64,
        dropout: float = 0.0,
        nhead: int = 4,
    ) -> None:
        """Initialize the model.

        :param input_dim: The dimensionality of the input to the model.
        :param hidden_dim: The dimensionality of the hidden layers.
        :param output_dim: The dimensionality of the output of the model.
        :param num_layers: The number of layers.
        :param last_layer_activation: Whether to apply an activation function to the last layer.
        :param dropout: The dropout rate.
        """
        super(Transformer, self).__init__()
        
        encoder_layer = TransformerEncoderLayer(
            d_model = input_dim,
            nhead = nhead,
            dim_feedforward = interaction_hidden_dims,
            dropout = dropout,
            batch_first = True
        )

        self.model = TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers = num_layers,
        )

    # TODO: add positional encoding and padding
    def forward(
            self,
            embeddings: torch.Tensor,
            coords: torch.Tensor = None,
            padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Runs the model on the data.

        :param embeddings: A tensor containing an embedding.
        :param coords: A tensor containing the coordinates.
        :param padding_mask: A tensor containing the padding mask.
        :return: A tensor containing the model's prediction.
        """
        # Apply layers
        embeddings = self.model(embeddings)
        return embeddings
