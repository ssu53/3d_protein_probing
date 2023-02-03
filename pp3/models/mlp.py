"""A class for a multilayer perceptron model."""
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MLP(pl.LightningModule):
    """A multilayer perceptron model."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: tuple[int, ...],
            target_mean: float,
            target_std: float,
            learning_rate: float = 1e-4,
            loss_fn: str = "mse"
    ) -> None:
        """Initialize the model.

        :param input_dim: The dimensionality of the input to the model.
        :param output_dim: The dimensionality of the output of the model.
        :param hidden_dims: The dimensionalities of the hidden layers.
        :param target_mean: The mean target value across the training set.
        :param target_std: The standard deviation of the target values across the training set.
        :param learning_rate: The learning rate.
        :param loss_fn: The loss function to use.
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.target_mean = target_mean
        self.target_std = target_std
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
        self.loss = self._get_loss_fn(loss_fn)
    
    
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
        y_scaled = (y - self.target_mean) / self.target_std

        y_hat_scaled = self(x).squeeze(dim=1)
        y_hat = y_hat_scaled * self.target_std + self.target_mean

        loss = self.loss(y_hat_scaled, y_scaled)

        self.log(f'{step_type}_loss', loss)

        y_np = y.detach().cpu().numpy()
        y_hat_np = y_hat.detach().cpu().numpy()

        # TODO: add MAPE (mean average percentage error)
        self.log(f'{step_type}_mae', mean_absolute_error(y_np, y_hat_np))
        self.log(f'{step_type}_rmse', np.sqrt(mean_squared_error(y_np, y_hat_np)))
        self.log(f'{step_type}_r2', r2_score(y_np, y_hat_np))

        # TODO: for test, save true and predicted values for scatter plots

        return loss

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
    
    def _get_loss_fn(self, loss_fn: str) -> nn.Module:
        """Returns the loss function.

        :param loss_fn: The name of the loss function.
        :return: The loss function.
        """
        if loss_fn == "mse":
            return nn.MSELoss()
        elif loss_fn == "mae":
            return nn.L1Loss()
        elif loss_fn == "huber":
            return nn.HuberLoss()
        elif loss_fn == "ce":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss function {loss_fn} not recognized.")
        
