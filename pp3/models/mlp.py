"""A class for a multilayer perceptron model."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """A multilayer perceptron model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the model.

        :param input_dim: The dimensionality of the input to the model.
        :param output_dim: The dimensionality of the output of the model.
        :param hidden_dim: The dimensionality of the hidden layers.
        :param num_layers: The number of layers.
        :param dropout: The dropout rate.
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.layer_dims = (
            [self.input_dim] + [hidden_dim] * self.num_layers
        )

        # Create layers
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
                for i in range(len(self.layer_dims) - 1)
            ]
        )

        # Create activation function
        self.activation = nn.ReLU()

        # Create dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(
            self,
            embeddings: torch.Tensor,
            coords: torch.Tensor,
            padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Runs the model on the data.

        :param embeddings: A tensor containing an embedding.
        :param coords: A tensor containing the coordinates.
        :param padding_mask: A tensor containing the padding mask.
        :return: A tensor containing the model's prediction.
        """
        # Apply layers
        for layer in self.layers:
            embeddings = self.dropout(embeddings)
            embeddings = layer(embeddings)
            embeddings = self.activation(embeddings)

        return embeddings
