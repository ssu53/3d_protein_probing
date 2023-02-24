"""Probe sequence embeddings for 3D geometric concepts."""
from pathlib import Path
from typing import Literal

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pp3.concepts import get_concept_names, get_concept_output_dim, get_concept_type
from pp3.models.mlp import MLP
from pp3.data import ProteinConceptDataModule


def probe_sequence_embeddings(
        proteins_path: Path,
        embeddings_path: Path,
        save_dir: Path,
        concepts_dir: Path,
        concept: str,
        protein_embedding_method: Literal['plm', 'baseline'],
        plm_residue_to_protein_method: Literal['mean', 'max', 'sum'],
        hidden_dim: int,
        num_layers: int,
        batch_size: int,
        logger_type: str,
        loss_fn: str = 'huber',
        learning_rate: float = 1e-4,
        max_epochs: int = 1000,
        ckpt_every_k_epochs: int = 10,
        split_seed: int = 0
) -> None:
    """Probe sequence embeddings for a 3D geometric concept.

    :param proteins_path: Path to PT file containing a dictionary mapping PDB ID to structure and sequence.
    :param embeddings_path: Path to PT file containing a dictionary mapping PDB ID to embeddings.
    :param save_dir: Path to directory where results and predictions will be saved.
    :param concepts_dir: Path to a directory containing PT files with dictionaries mapping PDB ID to concept values.
    :param concept: The concept to learn.
    :param protein_embedding_method: The method to use to compute the protein or residue embeddings.
    :param plm_residue_to_protein_method: The method to use to compute the PLM protein embedding from the residue embeddings for protein concepts.
    :param hidden_dim: The hidden dimension of the MLP.
    :param num_layers: The number of layers in the MLP.
    :param batch_size: The batch size.
    :param logger_type: The logger_type to use.
    :param loss_fn: The loss function to use.
    :param learning_rate: The learning rate for the optimizer.
    :param max_epochs: The maximum number of epochs to train for.
    :param ckpt_every_k_epochs: Save a checkpoint every k epochs.
    :param split_seed: The random seed to use for the train/val/test split.
    """
    # Create save directory
    run_name = f'{concept}_mlp_{num_layers}_layers'
    save_dir = save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Random seed
    pl.seed_everything(0)

    # Build data module
    data_module = ProteinConceptDataModule(
        proteins_path=proteins_path,
        embeddings_path=embeddings_path,
        concepts_dir=concepts_dir,
        concept=concept,
        protein_embedding_method=protein_embedding_method,
        plm_residue_to_protein_method=plm_residue_to_protein_method,
        batch_size=batch_size,
        split_seed=split_seed
    )
    data_module.setup()

    # Build MLP
    mlp = MLP(
        input_dim=data_module.embedding_dim,
        output_dim=get_concept_output_dim(concept),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        target_type=get_concept_type(concept),
        target_mean=data_module.train_dataset.target_mean,
        target_std=data_module.train_dataset.target_std,
        loss_fn=loss_fn,
        learning_rate=learning_rate
    )

    print(mlp)

    if logger_type == 'wandb':
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=f'Probing', save_dir=str(save_dir), name=run_name)
        logger.experiment.config.update({
            'concept': concept,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'loss_fn': loss_fn,
            'learning_rate': learning_rate,
            'split_seed': split_seed
        })
    elif logger_type == 'tensorboard':
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=str(save_dir), name=run_name)
    else:
        raise ValueError(f'Invalid logger type {logger_type}')

    # Build model checkpoint
    # TODO: how to split metrics by data, maybe in the model
    ckpt_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=2,
        monitor='val_loss',
        every_n_epochs=ckpt_every_k_epochs
    )

    # Build trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        devices=1,
        deterministic=True,
        max_epochs=max_epochs,
        log_every_n_steps=25,
        callbacks=[ckpt_callback]
    )

    # Train model
    trainer.fit(
        model=mlp,
        datamodule=data_module
    )

    # Test model
    trainer.test(datamodule=data_module, ckpt_path='best')

    # Make test predictions
    test_preds = trainer.predict(datamodule=data_module, ckpt_path='best')
    test_preds = torch.cat(test_preds)

    # Save test targets and predictions
    torch.save({
        'target': torch.from_numpy(data_module.test_dataset.targets),
        'prediction': test_preds
    }, save_dir / 'target_and_prediction.pt')


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
        protein_embedding_method: Literal['plm', 'baseline'] = 'plm'
        """The method to use to compute the protein or residue embeddings."""
        plm_residue_to_protein_method: Literal['mean', 'max', 'sum'] = 'sum'
        """The method to use to compute the PLM protein embedding from the residue embeddings for protein concepts."""
        hidden_dim: int = 100
        """Hidden dimension of the MLP."""
        num_layers: int = 1
        """The number of layers in the MLP."""
        batch_size: int = 100
        """The batch size."""
        logger_type: Literal['wandb', 'tensorboard'] = 'wandb'
        """The logger_type to use."""
        loss_fn: Literal['mse', 'mae', 'huber', 'ce'] = 'huber'
        """The loss function to use."""
        learning_rate: float = 1e-4
        """The learning rate for the optimizer."""
        max_epochs: int = 1000
        """The maximum number of epochs to train for."""
        ckpt_every_k_epochs: int = 10
        """Checkpoint every k epochs."""
        split_seed: int = 0
        """The random seed to use for the train/val/test split."""

        def configure(self) -> None:
            self.add_argument('--concept', choices=get_concept_names())

    probe_sequence_embeddings(**Args().parse_args().as_dict())
