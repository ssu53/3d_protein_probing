"""Probe a model for 3D geometric protein concepts."""
from pathlib import Path
from typing import Literal

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pp3.concepts import get_concept_output_dim, get_concept_type, get_concept_level
from pp3.models.model import Model
from pp3.data import ProteinConceptDataModule
from pp3.utils.constants import ENCODER_TYPES
from pp3.utils.plot import plot_preds_vs_targets


def probe(
    project_name: str,
    proteins_path: Path,
    embeddings_path: Path,
    save_dir: Path,
    concepts_dir: Path,
    concept: str,
    embedding_method: Literal['plm', 'baseline'],
    encoder_type: ENCODER_TYPES,
    encoder_num_layers: int,
    encoder_hidden_dim: int,
    predictor_num_layers: int,
    predictor_hidden_dim: int,
    batch_size: int,
    logger_type: Literal['wandb', 'tensorboard'] = 'wandb',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    dropout: float = 0.0,
    max_epochs: int = 1000,
    ckpt_every_k_epochs: int = 10,
    num_workers: int = 8,
    split_seed: int = 0,
    max_neighbors: int | None = None,
) -> None:
    """Probe a model for a 3D geometric protein concepts.

    :param project_name: The name of the project to use for WandB logging.
    :param proteins_path: Path to PT file containing a dictionary mapping PDB ID to structure and sequence.
    :param embeddings_path: Path to PT file containing a dictionary mapping PDB ID to embeddings.
    :param save_dir: Path to directory where results and predictions will be saved.
    :param concepts_dir: Path to a directory containing PT files with dictionaries mapping PDB ID to concept values.
    :param concept: The concept to learn.
    :param embedding_method: The method to use to compute the initial residue embeddings.
    :param encoder_type: The encoder type to use for encoding residue embeddings.
    :param encoder_num_layers: The number of layers in the encoder model.
    :param encoder_hidden_dim: The hidden dimension of the encoder model.
    :param predictor_num_layers: The number of layers in the final predictor MLP model.
    :param predictor_hidden_dim: The hidden dimension of the final predictor MLP model.
    :param batch_size: The batch size.
    :param logger_type: The logger_type to use.
    :param learning_rate: The learning rate for the optimizer.
    :param weight_decay: The weight decay for the optimizer.
    :param dropout: The dropout rate.
    :param max_epochs: The maximum number of epochs to train for.
    :param ckpt_every_k_epochs: Save a checkpoint every k epochs.
    :param num_workers: The number of workers to use for data loading.
    :param split_seed: The random seed to use for the train/val/test split.
    :param max_neighbors: The maximum number of neighbors to use for the graph in EGNN.
    """
    # Create save directory
    run_name = f'{concept}_{embedding_method}_{encoder_type}_{encoder_num_layers}L_{predictor_num_layers}L'
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
        embedding_method=embedding_method,
        batch_size=batch_size,
        num_workers=num_workers,
        split_seed=split_seed
    )
    data_module.setup()

    # Build model
    model = Model(
        encoder_type=encoder_type,
        input_dim=data_module.embedding_dim,
        output_dim=get_concept_output_dim(concept),
        encoder_num_layers=encoder_num_layers,
        encoder_hidden_dim=encoder_hidden_dim,
        predictor_num_layers=predictor_num_layers,
        predictor_hidden_dim=predictor_hidden_dim,
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
            'project_name': project_name,
            'proteins_path': str(proteins_path),
            'embeddings_path': str(embeddings_path),
            'save_dir': str(save_dir),
            'concepts_dir': str(concepts_dir),
            'concept': concept,
            'embedding_method': embedding_method,
            'encoder_type': encoder_type,
            'encoder_num_layers': encoder_num_layers,
            'encoder_hidden_dim': encoder_hidden_dim,
            'predictor_num_layers': predictor_num_layers,
            'predictor_hidden_dim': predictor_hidden_dim,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'max_epochs': max_epochs,
            'ckpt_every_k_epochs': ckpt_every_k_epochs,
            'num_workers': num_workers,
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

    tapify(probe)
