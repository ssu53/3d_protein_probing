"""Probe sequence embeddings for 3D geometric concepts."""
from pathlib import Path
from typing import Literal

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pp3.concepts import get_concept_output_dim, get_concept_type, get_concept_level
from pp3.models.model import Model
from pp3.data import ProteinConceptDataModule
from pp3.utils.constants import MODEL_TYPES
from pp3.utils.plot import plot_preds_vs_targets


def probe_sequence_embeddings(
    proteins_path: Path,
    embeddings_path: Path,
    save_dir: Path,
    concepts_dir: Path,
    concept: str,
    model_type: MODEL_TYPES = 'mlp',
    project_name: str = 'Probing',
    protein_embedding_method: Literal['plm', 'baseline'] = 'plm',
    plm_residue_to_protein_method: Literal['mean', 'max', 'sum'] = 'sum',
    hidden_dim: int = 100,
    num_layers: int = 1,
    batch_size: int = 100,
    logger_type: Literal['wandb', 'tensorboard'] = 'wandb',
    loss_fn: str = 'huber',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    dropout: float = 0.0,
    max_epochs: int = 1000,
    ckpt_every_k_epochs: int = 10,
    num_workers: int = 8,
    split_seed: int = 0,
    max_neighbors: int | None = None,
) -> None:
    """Probe sequence embeddings for a 3D geometric concept.

    :param proteins_path: Path to PT file containing a dictionary mapping PDB ID to structure and sequence.
    :param embeddings_path: Path to PT file containing a dictionary mapping PDB ID to embeddings.
    :param save_dir: Path to directory where results and predictions will be saved.
    :param concepts_dir: Path to a directory containing PT files with dictionaries mapping PDB ID to concept values.
    :param concept: The concept to learn.
    :param model_type: The model type to use.
    :param project_name: The name of the project to use for WandB logging.
    :param protein_embedding_method: The method to use to compute the protein or residue embeddings.
    :param plm_residue_to_protein_method: The method to use to compute the PLM protein embedding from the residue embeddings for protein concepts.
    :param hidden_dim: The hidden dimension of the MLP.
    :param num_layers: The number of layers in the MLP.
    :param batch_size: The batch size.
    :param logger_type: The logger_type to use.
    :param loss_fn: The loss function to use.
    :param learning_rate: The learning rate for the optimizer.
    :param weight_decay: The weight decay for the optimizer.
    :param dropout: The dropout rate.
    :param max_epochs: The maximum number of epochs to train for.
    :param ckpt_every_k_epochs: Save a checkpoint every k epochs.
    :param num_workers: The number of workers to use for data loading.
    :param split_seed: The random seed to use for the train/val/test split.
    """
    # Create save directory
    run_name = f'{concept}_{model_type}_{protein_embedding_method}_{num_layers}_layers'
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
        num_workers=num_workers,
        split_seed=split_seed
    )
    data_module.setup()

    # Build model
    model = Model(
        model_type=model_type,
        input_dim=data_module.embedding_dim,
        output_dim=get_concept_output_dim(concept),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        concept_level=get_concept_level(concept),
        target_type=get_concept_type(concept),
        target_mean=data_module.train_dataset.target_mean,
        target_std=data_module.train_dataset.target_std,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        max_neighbors=max_neighbors,
    )

    print(model)

    if logger_type == 'wandb':
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=project_name, save_dir=str(save_dir), name=run_name)
        logger.experiment.config.update({
            'concept': concept,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'loss_fn': loss_fn,
            'learning_rate': learning_rate,
            'split_seed': split_seed,
            'num_neighbors': max_neighbors,
        })
    elif logger_type == 'tensorboard':
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=str(save_dir), name=run_name)
    else:
        raise ValueError(f'Invalid logger type {logger_type}')

    # Build model checkpoint callback
    # TODO: how to split metrics by data, maybe in the model
    ckpt_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=2,
        monitor='val_loss',
        every_n_epochs=ckpt_every_k_epochs
    )

    # Build early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        mode='min'
    )

    # Build trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        devices=1,
        deterministic=True,
        max_epochs=max_epochs,
        log_every_n_steps=25,
        callbacks=[ckpt_callback, early_stopping]
    )

    # Train model
    trainer.fit(
        model=model,
        datamodule=data_module
    )

    # Test model
    trainer.test(datamodule=data_module, ckpt_path='best')

    # Make test predictions
    # TODO: fix this for residue_pair concepts (both memory issues and concat issues)
    test_preds, test_targets = zip(*trainer.predict(datamodule=data_module, ckpt_path='best'))
    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)

    # Plot predictions vs targets
    plot_preds_vs_targets(
        preds=test_preds,
        targets=test_targets,
        target_type=get_concept_type(concept),
        concept=concept,
        save_path=save_dir / 'target_vs_prediction.pdf',
    )

    # Save test targets and predictions
    torch.save({
        'prediction': test_preds,
        'target': test_targets
    }, save_dir / 'target_and_prediction.pt')


if __name__ == '__main__':
    from tap import tapify

    tapify(probe_sequence_embeddings)
