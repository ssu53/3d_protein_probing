"""Plotting functions."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, r2_score


def plot_preds_vs_targets(
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_type: str,
        concept: str,
        save_path: Path
) -> None:
    """Plot predicted values against targets.

    :param preds: The predicted values.
    :param targets: The true values.
    :param target_type: The type of the targets.
    :param concept: The concept being predicted.
    :param save_path: The path to save the plot to.
    """
    plt.clf()

    if target_type == 'regression':
        # Randomly sample large datasets to avoid memory issues
        if len(preds) > 10000:
            indices = torch.randperm(len(preds))[:10000]
            preds = preds[indices]
            targets = targets[indices]

        plt.scatter(preds, targets, s=3, label=rf'$R^2: {r2_score(targets, preds):.3f}$')

        # Add x = y line
        lims = [
            np.min([plt.xlim(), plt.ylim()]),
            np.max([plt.xlim(), plt.ylim()]),
        ]

        plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        plt.gca().set_aspect('equal')
        plt.xlim(lims)
        plt.ylim(lims)

        plt.xlabel('Predicted')
        plt.ylabel('Target')
        plt.legend()
        plt.title(f'Predicted vs. Target {concept}')
        plt.savefig(save_path, bbox_inches='tight')
    elif target_type in {'binary_classification', 'multi_classification'}:
        if target_type == 'binary_classification':
            preds = preds > 0.5
        elif target_type == 'multi_classification':
            preds = torch.argmax(preds, dim=1)

        ConfusionMatrixDisplay.from_predictions(targets, preds)
        plt.savefig(save_path, bbox_inches='tight')
    else:
        raise ValueError(f'Invalid target type: {target_type}')
