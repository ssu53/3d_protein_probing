"""Probe sequence embeddings for 3D geometric concepts."""
import sys
from pathlib import Path

import pytorch_lightning as pl

sys.path.append(Path(__file__).parent.parent.as_posix())

from pp3.concepts import get_concept_names
from pp3.data import ProteinConceptDataModule
from pp3.models.mlp import MLP


def probe_sequence_embeddings(
        proteins_path: Path,
        embeddings_path: Path,
        concepts_dir: Path,
        concept: str,
        batch_size: int
) -> None:
    """Probe sequence embeddings for a 3D geometric concept.

    :param proteins_path: Path to PT file containing a dictionary mapping PDB ID to structure and sequence.
    :param embeddings_path: Path to PT file containing a dictionary mapping PDB ID to embeddings.
    :param concepts_dir: Path to a directory containing PT files with dictionaries mapping PDB ID to concept values.
    :param concept: The concept to learn.
    :param batch_size: The batch size.
    """
    # Random seed
    pl.seed_everything(0)

    # Build data module
    data_module = ProteinConceptDataModule(
        proteins_path=proteins_path,
        embeddings_path=embeddings_path,
        concepts_dir=concepts_dir,
        concept=concept,
        batch_size=batch_size
    )

    # Build MLP
    # TODO: add learning rate as hyperparameter
    # TODO: add loss as hyperparameter
    mlp = MLP(
        input_dim=data_module.embedding_dim,
        output_dim=1,
        hidden_dims=tuple()
    )

    print(mlp)

    # Build trainer
    # TODO: checkpoints
    # TODO: specify logging location and metrics/loss and how to split metrics by data, maybe in the model
    # TODO: maybe in the val step, log the loss and metrics
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        deterministic=True,
        max_epochs=10
    )

    # Train model
    trainer.fit(
        model=mlp,
        datamodule=data_module
    )

    # Test model
    metrics = trainer.test(datamodule=data_module)
    print(metrics)


if __name__ == '__main__':
    from tap import Tap

    class Args(Tap):
        proteins_path: Path
        """Path to PT file containing a dictionary mapping PDB ID to structure and sequence."""
        embeddings_path: Path
        """"Path to PT file containing a dictionary mapping PDB ID to embeddings."""
        concepts_dir: Path
        """Path to a directory containing PT files with dictionaries mapping PDB ID to concept values."""
        concept: str = None
        """The concept to learn."""
        batch_size: int = 100
        """The batch size."""

        def configure(self) -> None:
            self.add_argument('--concept', choices=get_concept_names())

    probe_sequence_embeddings(**Args().parse_args().as_dict())
