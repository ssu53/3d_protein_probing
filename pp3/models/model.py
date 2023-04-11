"""A class for a multilayer perceptron model."""
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
from pp3.models.egnn import EGNN
from pp3.models.mlp import MLP
from pp3.utils.constants import BATCH_TYPE, MODEL_TYPES


class Model(pl.LightningModule):
    """A multilayer perceptron model."""

    def __init__(
        self,
        model_type: MODEL_TYPES,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        concept_level: str,
        target_type: str,
        target_mean: float | None,
        target_std: float | None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        dropout: float = 0.0
    ) -> None:
        """Initialize the model.

        :param model_type: The model type to use.
        :param input_dim: The dimensionality of the input to the model.
        :param output_dim: The dimensionality of the output of the model.
        :param hidden_dim: The dimensionality of the hidden layers.
        :param num_layers: The number of layers.
        :param concept_level: The concept level (e.g., protein, residue, residue_pair, residue_triplet).
        :param target_type: The type of the target values (e.g., regression or classification)
        :param target_mean: The mean target value across the training set.
        :param target_std: The standard deviation of the target values across the training set.
        :param learning_rate: The learning rate.
        :param weight_decay: The weight decay.
        :param dropout: The dropout rate.
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
                dropout=dropout
            )
        elif model_type == 'egnn':
            self.module = EGNN(
                node_dim=self.input_dim,
                dist_dim=self.hidden_dim,
                message_dim=self.hidden_dim,
                proj_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=dropout
            )
        elif model_type == 'tfn':
            raise NotImplementedError
        else:
            raise ValueError(f'Invalid model type: {model_type}')

        # Create final layer
        if self.concept_level in {'protein', 'residue'}:
            self.last_hidden_dim = self.hidden_dim
        elif self.concept_level == 'residue_pair':
            self.last_hidden_dim = self.hidden_dim * 2
        elif self.concept_level == 'residue_triplet':
            self.last_hidden_dim = self.hidden_dim * 3
        else:
            raise ValueError(f'Invalid concept level: {self.concept_level}')

        self.fc = nn.Linear(self.last_hidden_dim, self.output_dim)

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

        if self.concept_level == 'protein':
            pad_sum = pad.sum(dim=1)
            pad_sum[pad_sum == 0] = 1
            encodings = (encodings * pad).sum(dim=1) / pad_sum
            # If needed, modify embedding structure based on concept level
        elif self.concept_level == 'residue_pair':
            # Create pair embeddings
            breakpoint()  # TODO: check this and fix memory issues / random sampling
            left = encodings[:, None, :, :].expand(-1, encodings.shape[1], -1, -1)
            right = encodings[:, :, None, :].expand(-1, -1, encodings.shape[1], -1)
            encodings = torch.cat([left, right], dim=-1)
        elif self.concept_level == 'residue_triplet':
            # Create adjacent triples of residue embeddings
            encodings = torch.cat([encodings[:, :-2], encodings[:, 1:-1], encodings[:, 2:]], dim=-1)

        encodings = self.fc(encodings)

        return encodings

    def step(
            self,
            batch: BATCH_TYPE,
            batch_idx: int,
            step_type: str
    ) -> float:
        """Runs a training, validation, or test step.

        :param batch: A tuple containing the inputs and target.
        :param batch_idx: The index of the batch.
        :param step_type: The type of step (train, val, or test).
        :return: The loss.
        """
        # Unpack batch
        embeddings, coords, y, padding_mask = batch

        # Make predictions
        y_hat_scaled = self(embeddings, coords, padding_mask).squeeze(dim=-1)

        # Compute not NaN mask
        not_nan_mask = ~torch.isnan(y)

        # Set up padding
        if self.concept_level == 'residue':
            keep_mask = not_nan_mask * padding_mask
            pad_sum = padding_mask.sum(dim=1, keepdim=True).repeat(1, padding_mask.shape[1])
        elif self.concept_level == 'residue_pair':
            # TODO: keep mask
            breakpoint()
            padding_mask = padding_mask[:, None, :] * padding_mask[:, :, None]
            pad_sum = padding_mask.sum(dim=(1, 2), keepdim=True)  # TODO: Check this
        elif self.concept_level == 'residue_triplet':
            padding_mask = padding_mask[:, :-2] * padding_mask[:, 1:-1] * padding_mask[:, 2:]
            keep_mask = not_nan_mask * padding_mask
            pad_sum = padding_mask.sum(dim=1, keepdim=True).repeat(1, padding_mask.shape[1])
        else:
            raise ValueError(f'Invalid concept level: {self.concept_level}')

        # Flatten and remove padding and NaN
        keep_mask = keep_mask.bool()
        y = y[keep_mask]
        y_hat_scaled = y_hat_scaled[keep_mask]
        pad_sum = pad_sum[keep_mask]

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
        loss = (loss / pad_sum).mean()

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

        return loss

    def training_step(
            self,
            batch: BATCH_TYPE,
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
            batch: BATCH_TYPE,
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
            batch: BATCH_TYPE,
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
            batch: BATCH_TYPE,
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
