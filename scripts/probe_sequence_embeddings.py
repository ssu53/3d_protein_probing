"""Probe sequence embeddings for 3D geometric concepts."""
import sys
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

sys.path.append(Path(__file__).parent.parent.as_posix())

from pp3.concepts import get_concept_names
from pp3.data import ProteinConceptDataModule


def probe_sequence_embeddings(
        proteins_path: Path,
        embeddings_path: Path,
        save_dir: Path,
        concepts_dir: Path,
        concept: str,
        protein_embedding_method: Literal['sum', 'mean'],
        hidden_dims: tuple[int, ...],
        batch_size: int
) -> None:
    """Probe sequence embeddings for a 3D geometric concept.

    :param proteins_path: Path to PT file containing a dictionary mapping PDB ID to structure and sequence.
    :param embeddings_path: Path to PT file containing a dictionary mapping PDB ID to embeddings.
    :param save_dir: Path to directory where results and predictions will be saved.
    :param concepts_dir: Path to a directory containing PT files with dictionaries mapping PDB ID to concept values.
    :param concept: The concept to learn.
    :param protein_embedding_method: The method to use to compute the protein embedding from the residue embeddings
    :param hidden_dims: The hidden dimensions of the MLP.
    :param batch_size: The batch size.
    """
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Random seed
    pl.seed_everything(0)

    # Build data module
    # TODO: compute and predict per-residue SASA
    # TODO: remove concept outliers (like 5 huge SASAs)
    data_module = ProteinConceptDataModule(
        proteins_path=proteins_path,
        embeddings_path=embeddings_path,
        concepts_dir=concepts_dir,
        concept=concept,
        protein_embedding_method=protein_embedding_method,
        batch_size=batch_size
    )
    data_module.setup()

    # Build MLP
    # TODO: add learning rate as hyperparameter
    # TODO: add loss as hyperparameter
    # TODO: change protein embedding to sum
    mlp = MLP(
        input_dim=data_module.embedding_dim,
        output_dim=1,
        hidden_dims=hidden_dims,
        target_mean=data_module.train_dataset.target_mean,
        target_std=data_module.train_dataset.target_std,
    )

    print(mlp)

    # Build trainer
    # TODO: checkpoints
    # TODO: how to split metrics by data, maybe in the model
    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=str(save_dir)),
        accelerator='gpu',
        devices=1,
        deterministic=True,
        max_epochs=1000,
        log_every_n_steps=25
    )

    # Train model
    trainer.fit(
        model=mlp,
        datamodule=data_module
    )

    # Test model
    metrics = trainer.test(datamodule=data_module, ckpt_path='best')
    print(metrics)

    # Make test predictions
    test_preds = trainer.predict(datamodule=data_module, ckpt_path='best')

    # Save test predictions and true values
    torch.save({
        'preds': test_preds,
        'true': data_module.test_dataset.targets
    }, save_dir / 'preds_and_true.pt')


from pp3.models.mlp import MLP


if __name__ == '__main__':
    from tap import Tap

    class Args(Tap):
        proteins_path: Path
        """Path to PT file containing a dictionary mapping PDB ID to structure and sequence."""
        embeddings_path: Path
        """"Path to PT file containing a dictionary mapping PDB ID to embeddings."""
        save_dir: Path
        """Path to directory where results and predictions will be saved."""
        concepts_dir: Path
        """Path to a directory containing PT files with dictionaries mapping PDB ID to concept values."""
        concept: str
        """The concept to learn."""
        protein_embedding_method: Literal['sum', 'mean'] = 'sum'
        """The method to use to compute the protein embedding from the residue embeddings."""
        hidden_dims: tuple[int, ...] = tuple()
        """Hidden dimensions of the MLP."""
        batch_size: int = 100
        """The batch size."""

        def configure(self) -> None:
            self.add_argument('--concept', choices=get_concept_names())

    probe_sequence_embeddings(**Args().parse_args().as_dict())
