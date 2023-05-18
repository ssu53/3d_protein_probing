"""A class for a multilayer perceptron model."""
from collections import defaultdict
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)
from pp3.models.egnn import EGNN
from pp3.models.mlp import MLP
from pp3.models.transformer import Transformer
from pp3.models.tfn import TFN
from pp3.utils.constants import BATCH_TYPE, ENCODER_TYPES, MAX_SEQ_LEN


class Model(pl.LightningModule):
    """A multilayer perceptron model."""

    def __init__(
        self,
        encoder_type: ENCODER_TYPES,
        input_dim: int,
        output_dim: int,
        encoder_num_layers: int,
        encoder_hidden_dim: int,
        predictor_num_layers: int,
        predictor_hidden_dim: int,
        concept_level: str,
        target_type: str,
        target_mean: float | None,
        target_std: float | None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        max_neighbors: int | None = None,
        interaction_model: Literal['transformer'] | None = None,
        interaction_num_layers: int = 2,
        interaction_hidden_dim: int = 64
    ) -> None:
        """Initialize the model.

        :param encoder_type: The encoder type to use for encoding residue embeddings.
        :param input_dim: The dimensionality of the input to the model.
        :param output_dim: The dimensionality of the output of the model.
        :param encoder_num_layers: The number of layers in the encoder model.
        :param encoder_hidden_dim: The hidden dimension of the encoder model.
        :param predictor_num_layers: The number of layers in the final predictor MLP model.
        :param predictor_hidden_dim: The hidden dimension of the final predictor MLP model.
        :param concept_level: The concept level (e.g., protein, residue, residue_pair, residue_triplet).
        :param target_type: The type of the target values (e.g., regression or classification)
        :param target_mean: The mean target value across the training set.
        :param target_std: The standard deviation of the target values across the training set.
        :param learning_rate: The learning rate.
        :param weight_decay: The weight decay.
        :param dropout: The dropout rate.
        :param max_neighbors: The maximum number of neighbors to consider for each residue.
        :param interaction_num_layers: The number of layers in the interaction model.
        :param interaction_hidden_dim: The hidden dimension of the interaction model.
        """
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_num_layers = encoder_num_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.predictor_num_layers = predictor_num_layers
        self.predictor_hidden_dim = predictor_hidden_dim
        self.target_type = target_type
        self.target_mean = target_mean
        self.target_std = target_std
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.concept_level = concept_level
        self.interaction_model = interaction_model

        self.train_y = []
        self.train_y_hat = []
        self.val_y = []
        self.val_y_hat = []
        self.test_y = []
        self.test_y_hat = []

        if encoder_type == 'mlp':
            self.encoder = MLP(
                input_dim=self.input_dim,
                hidden_dim=self.encoder_hidden_dim,
                output_dim=self.encoder_hidden_dim,
                num_layers=self.encoder_num_layers,
                last_layer_activation=True,
                dropout=dropout
            )
            last_hidden_dim = self.encoder_hidden_dim if self.encoder_num_layers > 0 else self.input_dim
        elif encoder_type == 'egnn':
            self.encoder = EGNN(
                node_dim=self.input_dim,
                hidden_dim=self.encoder_hidden_dim,
                num_layers=self.encoder_num_layers,
                max_neighbors=max_neighbors,
                dropout=dropout
            )
            last_hidden_dim = self.input_dim
        elif encoder_type == 'tfn':
            self.encoder = TFN(
                node_dim=self.input_dim,
                num_layers=self.encoder_num_layers,
                max_neighbors=max_neighbors,
                dropout=dropout
            )
            last_hidden_dim = self.input_dim
        else:
            raise ValueError(f'Invalid model type: {encoder_type}')

        # Create final layer
        if self.concept_level in {'protein', 'residue'}:
            predictor_dim_multiplier = 1
        elif self.concept_level == 'residue_pair':
            predictor_dim_multiplier = 2
        elif self.concept_level == 'residue_triplet':
            predictor_dim_multiplier = 3
        elif self.concept_level == 'residue_quadruplet':
            predictor_dim_multiplier = 4
        else:
            raise ValueError(f'Invalid concept level: {self.concept_level}')

        if self.interaction_model == 'transformer':
            self.interaction_model = Transformer(
                input_dim=input_dim,
                num_layers=interaction_num_layers,
                interaction_hidden_dims=interaction_hidden_dim
            )
        else:
            self.interaction_model = None
        
        self.predictor = MLP(
            input_dim=last_hidden_dim * predictor_dim_multiplier,
            hidden_dim=self.predictor_hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.predictor_num_layers,
            last_layer_activation=False,
            dropout=dropout
        )

        # Create loss function
        self.loss = self._get_loss_fn()

    def forward(
            self,
            embeddings: torch.Tensor,
            coords: torch.Tensor,
            padding_mask: torch.Tensor,
            keep_mask: torch.Tensor = None
    ) -> list[torch.Tensor]:
        """Runs the model on the data.

        :param embeddings: A tensor containing an embedding.
        :param coords: A tensor containing the coordinates.
        :param padding_mask: A tensor containing the padding mask.
        :param keep_mask: A tensor containing the target mask.
        :return: A tensor containing the model's prediction.
        """
        # Get dimensions
        num_proteins, num_residues = padding_mask.shape

        # Encode embeddings
        encodings = self.encoder(embeddings, coords, padding_mask)

        # Modeling the interactions
        if self.interaction_model is not None:
            encodings = self.interaction_model(encodings)

        # If needed, modify embedding structure based on concept level
        if self.concept_level == 'protein':
            pad_sum = padding_mask.sum(dim=1, keepdim=True)
            pad_sum[pad_sum == 0] = 1

            # Average over all residues
            encodings = (encodings * padding_mask.unsqueeze(dim=-1)).sum(dim=1) / pad_sum
        elif self.concept_level == 'residue_pair':
            # Create pairs of residue embeddings
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
        elif self.concept_level == 'residue_quadruplet':
            # Create adjacent quadruples of residue embeddings
            encodings = torch.cat([encodings[:, :-3], encodings[:, 1:-2], encodings[:, 2:-1], encodings[:, 3:]], dim=-1)
        elif self.concept_level != 'residue':
            raise ValueError(f'Invalid concept level: {self.concept_level}')

        # Select encodings using keep mask
        if keep_mask is not None and self.concept_level != 'residue_pair':
            encodings = encodings[keep_mask]

        # Predict using MLP
        output = self.predictor(encodings)

        return output

    def step(
            self,
            batch: BATCH_TYPE,
            batch_idx: int,
            step_type: str
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """Runs a training, validation, or test step.

        :param batch: A tuple containing the inputs and target.
        :param batch_idx: The index of the batch.
        :param step_type: The type of step (train, val, or test).
        :return: The loss, targets, and predictions (both per protein).
        """
        # Unpack batch
        embeddings, coords, y, padding_mask = batch
        num_proteins, num_residues = padding_mask.shape

        # Compute y mask
        y_mask = ~torch.isnan(y)

        # Set up masks
        if self.concept_level == 'protein':
            keep_mask = y_mask

            # Handle multiple targets per protein
            if keep_mask.ndim == 2:
                keep_mask = keep_mask.all(dim=1)

            keep_sum = 1
        elif self.concept_level == 'residue':
            # Keep mask
            keep_mask = (y_mask * padding_mask).bool()

            # Keep sum (for normalization per protein)
            keep_sum = keep_mask.sum(dim=1, keepdim=True).repeat(1, num_residues)
            keep_sum = keep_sum[keep_mask]
        elif self.concept_level == 'residue_pair':
            # Padding mask
            pair_padding_mask = padding_mask[:, None, :] * padding_mask[:, :, None]

            # Random sampling of residue pairs (to avoid memory issues)
            num_pairs = num_proteins * MAX_SEQ_LEN

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
            keep_sum = keep_mask.sum(dim=1, keepdim=True).repeat(1, num_residues - 2)
            keep_sum = keep_sum[keep_mask]
        elif self.concept_level == 'residue_quadruplet':
            # Keep mask
            quadruplet_padding_mask = padding_mask[:, :-3] * padding_mask[:, 1:-2] * padding_mask[:, 2:-1] * padding_mask[:, 3:]
            keep_mask = (y_mask * quadruplet_padding_mask).bool()

            # Keep sum (for normalization per protein)
            keep_sum = keep_mask.sum(dim=1, keepdim=True).repeat(1, num_residues - 3)
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
            y_hat = torch.sigmoid(y_hat_scaled)
        elif self.target_type == 'multi_classification':
            y_scaled = F.one_hot(y, num_classes=self.output_dim).float()
            y_hat = torch.softmax(y_hat_scaled, dim=-1)
        else:
            raise ValueError(f'Invalid target type: {self.target_type}')

        # Compute loss (average per protein and then averaged across proteins)
        loss = self.loss(y_hat_scaled, y_scaled)
        loss = (loss / keep_sum).sum() / num_proteins

        # Convert target and predictions to NumPy
        y = y.detach().cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()

        # Separate y and y_hat by protein
        if self.concept_level == 'protein':
            y_per_protein = torch.ones(num_proteins, dtype=torch.long)
        else:
            y_per_protein = keep_mask.sum(dim=-1)

        y_per_protein_cumsum = [0] + y_per_protein.cumsum(dim=0).tolist()

        assert y_per_protein_cumsum[-1] == y.shape[0]
        assert len(y_per_protein_cumsum) - 1 == num_proteins

        y = [
            y[y_per_protein_cumsum[i]:y_per_protein_cumsum[i + 1]]
            for i in range(num_proteins)
        ]
        y_hat = [
            y_hat[y_per_protein_cumsum[i]:y_per_protein_cumsum[i + 1]]
            for i in range(num_proteins)
        ]

        return loss, y, y_hat

    def evaluate(
            self,
            y: list[np.ndarray],
            y_hat: list[np.ndarray],
            step_type: str
    ) -> None:
        # Flatten y and y_hat for micro metrics
        y_flat = [np.concatenate(y)]
        y_hat_flat = [np.concatenate(y_hat)]

        # Compute metrics
        for metric_level, y_arrs, y_hat_arrs in zip(
                ['micro', 'macro'],
                [y_flat, y],
                [y_hat_flat, y_hat]
        ):
            if self.concept_level == 'protein' and metric_level == 'macro':
                continue

            results = defaultdict(list)

            for y_arr, y_hat_arr in zip(y_arrs, y_hat_arrs):
                if len(y_arr) == 0:
                    continue

                if self.target_type == 'regression':
                    results['mape'].append(mean_absolute_percentage_error(y_arr, y_hat_arr))
                    results['mae'].append(mean_absolute_error(y_arr, y_hat_arr))
                    results['rmse'].append(np.sqrt(mean_squared_error(y_arr, y_hat_arr)))
                    results['r2'].append(r2_score(y_arr, y_hat_arr))
                elif self.target_type == 'binary_classification':
                    if y_arr.ndim == 1:
                        y_arr = y_arr[:, None]
                        y_hat_arr = y_hat_arr[:, None]

                    roc_aucs, aps, indices = [], [], []
                    for i in range(y_arr.shape[1]):
                        # Check if there is only one class
                        if set(np.unique(y_arr[:, i])) != {0, 1}:
                            # Error if micro, skip if macro (skipping protein)
                            if metric_level == 'micro':
                                raise ValueError('Micro metrics cannot be computed when there is only one class')
                            else:
                                continue

                        roc_aucs.append(roc_auc_score(y_arr[:, i], y_hat_arr[:, i]))
                        aps.append(average_precision_score(y_arr[:, i], y_hat_arr[:, i]))
                        indices.append(i)

                    # Handle one or multiple labels
                    if len(roc_aucs) > 0:
                        if y_arr.shape[1] == 1:
                            results['auc'].append(np.mean(roc_aucs))
                            results['ap'].append(np.mean(aps))
                        else:
                            results['mean_auc'].append(np.mean(roc_aucs))
                            results['mean_ap'].append(np.mean(aps))

                            for roc_auc, ap, index in zip(roc_aucs, aps, indices):
                                results[f'auc_{index}'].append(roc_auc)
                                results[f'ap_{index}'].append(ap)
                elif self.target_type == 'multi_classification':
                    results['accuracy'].append((y_arr == np.argmax(y_hat_arr, axis=1)).mean())
                else:
                    raise ValueError(f'Invalid target type: {self.target_type}')

            # Compute mean of metric values
            for metric_name, metric_values in results.items():
                self.log(f'{step_type}_{metric_level}_{metric_name}', float(np.nanmean(metric_values)))

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
        loss, y, y_hat = self.step(
            batch=batch,
            batch_idx=batch_idx,
            step_type='train'
        )

        self.log('train_loss', loss)

        self.train_y += y
        self.train_y_hat += y_hat

        return loss

    def on_train_epoch_end(self) -> None:
        """Evaluate train predictions at the end of an epoch."""
        self.evaluate(
            y=self.train_y,
            y_hat=self.train_y_hat,
            step_type='train'
        )

        self.train_y = []
        self.train_y_hat = []

    def validation_step(
            self,
            batch: BATCH_TYPE,
            batch_idx: int
    ) -> None:
        """Runs a validation step.

        :param batch: A tuple containing the input and target.
        :param batch_idx: The index of the batch.
        :return: The loss.
        """
        loss, y, y_hat = self.step(
            batch=batch,
            batch_idx=batch_idx,
            step_type='val'
        )

        self.log('val_loss', loss)

        self.val_y += y
        self.val_y_hat += y_hat

    def on_validation_epoch_end(self) -> None:
        """Evaluate validation predictions at the end of an epoch."""
        self.evaluate(
            y=self.val_y,
            y_hat=self.val_y_hat,
            step_type='val'
        )

        self.val_y = []
        self.val_y_hat = []

    def test_step(
            self,
            batch: BATCH_TYPE,
            batch_idx: int
    ) -> None:
        """Runs a test step.

        :param batch: A tuple containing the input and target.
        :param batch_idx: The index of the batch.
        :return: The loss.
        """
        loss, y, y_hat = self.step(
            batch=batch,
            batch_idx=batch_idx,
            step_type='test'
        )

        self.log('test_loss', loss)

        self.test_y += y
        self.test_y_hat += y_hat

    def on_test_epoch_end(self) -> None:
        """Evaluate test predictions at the end of an epoch."""
        self.evaluate(
            y=self.test_y,
            y_hat=self.test_y_hat,
            step_type='test'
        )

        self.test_y = []
        self.test_y_hat = []

    def predict_step(
            self,
            batch: BATCH_TYPE,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Runs a prediction step.

        :param batch: A tuple containing the input and target.
        :param batch_idx: The index of the batch.
        :param dataloader_idx: The index of the dataloader.
        :return: A tuple of lists of true and predicted values for the batch.
        """
        loss, y, y_hat = self.step(
            batch=batch,
            batch_idx=batch_idx,
            step_type='test'
        )

        return y, y_hat

    def configure_optimizers(self) -> dict[str, torch.optim.Optimizer | ReduceLROnPlateau | str]:
        """Configures the optimizer and scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        return {
            'optimizer': optimizer,
            'monitor': 'val_loss'
        }

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
