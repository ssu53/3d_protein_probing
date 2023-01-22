"""A class for a multilayer perceptron model."""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class MLP(pl.LightningModule):
    """A multilayer perceptron model."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: tuple[int, ...],
            scaler: StandardScaler,
            learning_rate: float = 1e-4
    ) -> None:
        """Initialize the model.

        :param input_dim: The dimensionality of the input to the model.
        :param output_dim: The dimensionality of the output of the model.
        :param hidden_dims: The dimensionalities of the hidden layers.
        :param scaler: A scalar to scale the target values.
        :param learning_rate: The learning rate.
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.scaler = scaler
        self.learning_rate = learning_rate

        self.layer_dims = [self.input_dim] + list(self.hidden_dims) + [self.output_dim]

        # Create layers
        self.layers = nn.ModuleList([
            nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])
            for i in range(len(self.layer_dims) - 1)
        ])

        # Create activation function
        self.activation = nn.ReLU()

        # Create loss function
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the model on the data.

        :param x: A tensor containing an embedding.
        :return: A tensor containing the model's prediction.
        """
        # Apply layers
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i != len(self.layers) - 1:
                x = self.activation(x)

        return x

    def step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            step_type: str
    ) -> float:
        """Runs a training, validation, or test step.

        :param batch: A tuple containing the input and target.
        :param batch_idx: The index of the batch.
        :param step_type: The type of step (train, val, or test).
        :return: The loss.
        """
        x, y = batch
        y = y.float()
        y_scaled = torch.from_numpy(self.scaler.transform(y.numpy()))

        y_hat_scaled = self(x)
        y_hat = torch.from_numpy(self.scaler.inverse_transform(y_hat_scaled.detach().numpy()))
        y_hat_scaled, y_hat = y_hat_scaled.squeeze(dim=1), y_hat.squeeze(dim=1)

        loss_scaled = self.loss(y_hat_scaled, y_scaled)
        loss = self.loss(y_hat, y)

        self.log(f'{step_type}_loss_scaled', loss_scaled)
        self.log(f'{step_type}_loss', loss)

        return loss_scaled

    def training_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> float:
        """Runs a training step.

        :param batch: A tuple containing the input and target.
        :param batch_idx: The index of the batch.
        :return: The loss.
        """
        return self.step(
            batch=batch,
            batch_idx=batch_idx,
            step_type='train'
        )

    def validation_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> float:
        """Runs a validation step.

        :param batch: A tuple containing the input and target.
        :param batch_idx: The index of the batch.
        :return: The loss.
        """
        return self.step(
            batch=batch,
            batch_idx=batch_idx,
            step_type='val'
        )

    def test_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> float:
        """Runs a test step.

        :param batch: A tuple containing the input and target.
        :param batch_idx: The index of the batch.
        :return: The loss.
        """
        return self.step(
            batch=batch,
            batch_idx=batch_idx,
            step_type='test'
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
