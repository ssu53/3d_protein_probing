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

        self.max_residue_pairs_per_protein = 100

        if model_type == 'mlp':
            self.module = MLP(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=dropout
            )
            last_hidden_dim = self.hidden_dim
        elif model_type == 'egnn':
            self.module = EGNN(
                node_dim=self.input_dim,
                dist_dim=self.hidden_dim,
                message_dim=self.hidden_dim,
                proj_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=dropout
            )
            last_hidden_dim = self.input_dim
        elif model_type == 'tfn':
            raise NotImplementedError
        else:
            raise ValueError(f'Invalid model type: {model_type}')

        # Create final layer
        if self.concept_level in {'protein', 'residue'}:
            last_dim_multiplier = 1
        elif self.concept_level == 'residue_pair':
            last_dim_multiplier = 2
        elif self.concept_level == 'residue_triplet':
            last_dim_multiplier = 3
        else:
            raise ValueError(f'Invalid concept level: {self.concept_level}')

        self.fc = nn.Linear(last_hidden_dim * last_dim_multiplier, self.output_dim)

        # Create loss function
        self.loss = self._get_loss_fn()

    def forward(
            self,
            embeddings: torch.Tensor,
            coords: torch.Tensor,
            padding_mask: torch.Tensor,
            keep_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Runs the model on the data.

        :param embeddings: A tensor containing an embedding.
        :param coords: A tensor containing the coordinates.
        :param padding_mask: A tensor containing the padding mask.
        :param keep_mask: A tensor containing the target mask.
        :return: A tensor containing the model's prediction.
        """
        encodings = self.module(embeddings, coords, padding_mask)

        # If needed, modify embedding structure based on concept level
        if self.concept_level == 'protein':
            pad_sum = padding_mask.sum(dim=1)
            pad_sum[pad_sum == 0] = 1
            encodings = (encodings * padding_mask).sum(dim=1) / pad_sum

            if keep_mask is not None:
                encodings = encodings[keep_mask]
        elif self.concept_level == 'residue_pair':
            # Create pair embeddings
            num_proteins, num_residues = padding_mask.shape

            if keep_mask is not None:
                keep_mask_indices = torch.nonzero(keep_mask.view(num_proteins, num_residues, num_residues))
            else:
                keep_mask_indices = torch.nonzero(torch.ones(num_proteins, num_residues, num_residues))

            left = encodings[keep_mask_indices[:, 0], keep_mask_indices[:, 1]]
            right = encodings[keep_mask_indices[:, 0], keep_mask_indices[:, 2]]
            encodings = torch.cat([left, right], dim=-1)
        elif self.concept_level == 'residue_triplet':
            # Create adjacent triples of residue embeddings
            encodings = torch.cat([encodings[:, :-2], encodings[:, 1:-1], encodings[:, 2:]], dim=-1)

            if keep_mask is not None:
                encodings = encodings[keep_mask]
        elif self.concept_level == 'residue':
            if keep_mask is not None:
                encodings = encodings[keep_mask]
        else:
            raise ValueError(f'Invalid concept level: {self.concept_level}')

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
        num_proteins, num_residues = padding_mask.shape

        # Compute y mask
        y_mask = ~torch.isnan(y)

        # Set up masks
        if self.concept_level == 'residue':
            # Keep mask
            keep_mask = (y_mask * padding_mask).bool()

            # Keep sum (for normalization per protein)
            keep_sum = keep_mask.sum(dim=1, keepdim=True).repeat(1, num_residues)
            keep_sum = keep_sum[keep_mask]
        elif self.concept_level == 'residue_pair':
            # Padding mask
            pair_padding_mask = padding_mask[:, None, :] * padding_mask[:, :, None]

            # Random sampling of residue pairs (to avoid memory issues)
            num_pairs = num_proteins * self.max_residue_pairs_per_protein

            pair_padding_mask_flat = pair_padding_mask.view(num_proteins, -1)
            pair_indices = torch.nonzero(y_mask * pair_padding_mask_flat)
            pair_indices = pair_indices[torch.randperm(pair_indices.shape[0])[:num_pairs]]

            # Keep mask
            keep_mask = torch.zeros_like(y_mask)
            keep_mask[pair_indices[:, 0], pair_indices[:, 1]] = 1
            keep_mask = keep_mask.bool()

            # Keep sum (for normalization per protein)
            keep_sum = keep_mask.sum(dim=1, keepdim=True).repeat(1, num_residues ** 2)
            keep_sum = keep_sum[keep_mask]
        elif self.concept_level == 'residue_triplet':
            # Keep mask
            triplet_padding_mask = padding_mask[:, :-2] * padding_mask[:, 1:-1] * padding_mask[:, 2:]
            keep_mask = (y_mask * triplet_padding_mask).bool()

            # Keep sum (for normalization per protein)
            keep_sum = keep_mask.sum(dim=1, keepdim=True).repeat(1, num_residues)
            keep_sum = keep_sum[keep_mask]
        else:
            raise ValueError(f'Invalid concept level: {self.concept_level}')

        # Select target using keep mask
        y = y[keep_mask]

        # Make predictions
        y_hat_scaled = self(embeddings, coords, padding_mask, keep_mask).squeeze(dim=-1)

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

        # Compute loss (average per protein and then averaged across proteins)
        loss = self.loss(y_hat_scaled, y_scaled)
        loss = (loss / keep_sum).sum() / num_proteins

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
            if len(np.unique(y_np)) == 2:
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
        embeddings, coords, y, padding_mask = batch

        # Make predictions
        y_hat_scaled = self(embeddings, coords, padding_mask).squeeze(dim=1)

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
