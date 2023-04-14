"""A class for a multilayer perceptron model."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """A multilayer perceptron model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        last_layer_activation: bool = False,
        dropout: float = 0.0
    ) -> None:
        """Initialize the model.

        :param input_dim: The dimensionality of the input to the model.
        :param hidden_dim: The dimensionality of the hidden layers.
        :param output_dim: The dimensionality of the output of the model.
        :param num_layers: The number of layers.
        :param last_layer_activation: Whether to apply an activation function to the last layer.
        :param dropout: The dropout rate.
        """
        super(MLP, self).__init__()

        self.last_layer_activation = last_layer_activation

        # Create layer dimensions
        if num_layers < 1:
            layer_dims = []
        else:
            layer_dims = (
                [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
            )

        # Create layers
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_dims[i], layer_dims[i + 1])
                for i in range(len(layer_dims) - 1)
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
        for i, layer in enumerate(self.layers):
            embeddings = self.dropout(embeddings)
            embeddings = layer(embeddings)

            if self.last_layer_activation or i < len(self.layers) - 1:
                embeddings = self.activation(embeddings)

        return embeddings
