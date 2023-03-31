"""A class for a multilayer perceptron model."""
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)
from pp3.models.mlp import MLP


class Model(pl.LightningModule):
    """A multilayer perceptron model."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        target_type: str,
        target_mean: float | None,
        target_std: float | None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        model_type: Literal["mlp", "egnn", "tfn"] = 'mlp',
        concept_level: Literal["residue", "protein", "residue_pair", "residue_triplets"] = False,
        **kwargs
    ) -> None:
        """Initialize the model.

        :param input_dim: The dimensionality of the input to the model.
        :param output_dim: The dimensionality of the output of the model.
        :param hidden_dim: The dimensionality of the hidden layers.
        :param num_layers: The number of layers.
        :param target_type: The type of the target values (e.g., regression or classification)
        :param target_mean: The mean target value across the training set.
        :param target_std: The standard deviation of the target values across the training set.
        :param learning_rate: The learning rate.
        :param weight_decay: The weight decay.
        :param dropout: The dropout rate.
        :param loss_fn: The loss function to use.
        """
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.target_type = target_type
        self.target_mean = target_mean
        self.target_std = target_std
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.concept_level = concept_level

        if model_type == 'mlp':
            self.module = MLP(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=dropout,
                **kwargs
            )
        elif model_type == 'egnn':
            pass
        elif model_type == 'tfn':
            pass
        else:
            raise ValueError(f'Invalid model type: {model_type}')

        # Create final layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

        # Create loss function
        self.loss = self._get_loss_fn()

    def forward(self, x: torch.Tensor, c: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Runs the model on the data.

        :param x: A tensor containing an embedding.
        :param c: A tensor containing the coordinates.
        :param pad: A tensor containing the padding mask.
        :return: A tensor containing the model's prediction.
        """
        encodings = self.module(x, c, pad)

        if self.concept_level == "protein":
            pad_sum = pad.sum(dim=1)
            pad_sum[pad_sum == 0] = 1
            encodings = (encodings * pad).sum(dim=1) / pad_sum
            # If needed, modify embedding structure based on concept level
        elif self.concept_level == 'residue_pair':
            # Create pair embeddings
            left = encodings[:, None, :, :].expand(-1, encodings.shape[1], -1, -1)
            right = encodings[:, :, None, :].expand(-1, -1, encodings.shape[1], -1)
            encodings = torch.cat([left, right], dim=-1)
        elif self.concept_level == 'residue_triplet':
            # Create adjacent triples of residue embeddings
            encodings = torch.cat([encodings[:, :-2], encodings[:, 1:-1], encodings[:, 2:]], dim=1)

        encodings = self.fc(encodings)
        return encodings

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
        # Unpack batch
        x, c, y, pad = batch

        # Make predictions
        y_hat_scaled = self(x, c, pad).squeeze(dim=1)

        # Remove NaNs
        nan_mask = torch.isnan(y)
        y_hat_scaled = y_hat_scaled[~nan_mask]
        y = y[~nan_mask]

        # Scale/unscale target and predictions
        if self.target_type == 'regression':
            y = y.float()
            y_scaled = (y - self.target_mean) / self.target_std
            y_hat = y_hat_scaled * self.target_std + self.target_mean
        elif self.target_type == 'binary_classification':
            y_scaled = y.float()
            y_hat = y_hat_scaled
        elif self.target_type == 'multi_classification':
            y_scaled = F.one_hot(y, num_classes=self.output_dim).float()
            y_hat = y_hat_scaled
        else:
            raise ValueError(f'Invalid target type: {self.target_type}')

        # Compute loss
        loss = self.loss(y_hat_scaled, y_scaled)

        if self.concept_level == "residue":
            pad_sum = pad.sum(dim=1)
            pad_sum[pad_sum == 0] = 1
            loss = (loss * pad).sum(dim=1) / pad_sum
        elif self.concept_level == 'residue_pair':
            # Create pair embeddings
            pair_mask = pad[:, None, :] * pad[:, :, None]
            pair_mask_sum = pair_mask.sum(dim=(1, 2))
            pair_mask_sum[pair_mask_sum == 0] = 1
            loss = (loss * pair_mask).sum(dim=(1, 2)) / pair_mask_sum
        elif self.concept_level == 'residue_triplet':
            pad = pad[:, :-2] * pad[:, 1:-1] * pad[:, 2:]
            pad_sum = pad.sum(dim=1)
            pad_sum[pad_sum == 0] = 1
            loss = (loss * pad).sum(dim=1) / pad_sum

        loss = loss.mean()

        # Convert target and predictions to NumPy
        y_np = y.detach().cpu().numpy()
        y_hat_np = y_hat.detach().cpu().numpy()

        # Log metrics
        self.log(f'{step_type}_loss', loss)

        if self.target_type == 'regression':
            # TODO: add MAPE (mean average percentage error)
            self.log(f'{step_type}_mae', mean_absolute_error(y_np, y_hat_np))
            self.log(f'{step_type}_rmse', np.sqrt(mean_squared_error(y_np, y_hat_np)))
            self.log(f'{step_type}_r2', r2_score(y_np, y_hat_np))
        elif self.target_type == 'binary_classification':
            self.log(f'{step_type}_auc', roc_auc_score(y_np, y_hat_np))
            self.log(f'{step_type}_ap', average_precision_score(y_np, y_hat_np))
        elif self.target_type == 'multi_classification':
            self.log(f'{step_type}_accuracy', (y_np == np.argmax(y_hat_np, axis=1)).mean())
        else:
            raise ValueError(f'Invalid target type: {self.target_type}')

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

    def predict_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs a prediction step.

        :param batch: A tuple containing the input and target.
        :param batch_idx: The index of the batch.
        :param dataloader_idx: The index of the dataloader.
        :return: A tensor of predictions and true values for the batch.
        """
        # Unpack batch
        x, c, y, pad = batch

        # Make predictions
        y_hat_scaled = self(x, c, pad).squeeze(dim=1)

        # Unscale predictions
        if self.target_type == 'regression':
            y_hat = y_hat_scaled * self.target_std + self.target_mean
        elif self.target_type == 'binary_classification':
            y_hat = F.sigmoid(y_hat_scaled)
        elif self.target_type == 'multi_classification':
            y_hat = F.softmax(y_hat_scaled, dim=-1)
        else:
            raise ValueError(f'Invalid target type: {self.target_type}')

        return y_hat, y

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def _get_loss_fn(self) -> nn.Module:
        """Gets the loss function."""
        if self.target_type == 'regression':
            return nn.HuberLoss(reduction="none")
        elif self.target_type == 'binary_classification':
            return nn.BCEWithLogitsLoss(reduction="none")
        elif self.target_type == 'multi_classification':
            return nn.CrossEntropyLoss(reduction="none")
        else:
            raise ValueError(f'Invalid target type: {self.target_type}')
