from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pp3.models_prot.default_paths import (
    get_valid_pdb_ids_train_path, get_valid_pdb_ids_val_path, get_pairfile_train_path, get_pairfile_val_path
)
from pp3.models_prot.data import ProteinPairDataModule
from pp3.models_prot.model import ModelProt



def embed_for_retrieval(
    project_name: str,
    save_dir: Path,
    num_layers: int,
    num_heads: int,
    embeddings_path: Path,
    valid_pdb_ids_train_path: Path | None = None,
    valid_pdb_ids_val_path: Path | None = None,
    pairfile_train_path: Path | None = None,
    pairfile_val_path: Path | None = None,
    temperature: float = 1e-1,
    similarity_func: str = 'cosine',
    learning_rate: float = 1e-4,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    num_workers: int = 4,
    max_epochs: int = 500,
    patience: int = 25,
    ckpt_every_k_epochs: int = 1,
    num_sanity_val_steps: int = 2,
    run_id_number: int | None = None,
    run_name_suffix: str = '',
    entity: str = 'ssu53',
    disable_wandb: bool = False,
):
    """
    :param:
    """

    # Create save directory
    embeddings_name = str(embeddings_path).split('/')[-1].replace('.pt','')
    run_name = f'{embeddings_name}_{num_layers}L_{learning_rate}lr_{batch_size}bs_{temperature}temp'
    if run_name_suffix:
        run_name += f'_{run_name_suffix}'

    save_dir = save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)


    # Get default paths
    if valid_pdb_ids_train_path is None:
        valid_pdb_ids_train_path = get_valid_pdb_ids_train_path()
    if valid_pdb_ids_val_path is None:
        valid_pdb_ids_val_path = get_valid_pdb_ids_val_path()
    if pairfile_train_path is None:
        pairfile_train_path = get_pairfile_train_path()
    if pairfile_val_path is None:
        pairfile_val_path = get_pairfile_val_path()


    # Random seed
    pl.seed_everything(0)

    # Build data module
    data_module = ProteinPairDataModule(
        embeddings_path=embeddings_path,
        pdb_ids_train_path=valid_pdb_ids_train_path,
        pdb_ids_val_path=valid_pdb_ids_val_path,
        pairfile_train_path=pairfile_train_path,
        pairfile_val_path=pairfile_val_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_module.setup()


    # Build model
    model = ModelProt(
        num_layers=num_layers,
        num_heads=num_heads,
        num_channels=data_module.embedding_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        temperature=temperature,
        weight_decay=weight_decay,
        similarity_func=similarity_func,
    )

    print(model)


    # Get logger
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(
        project=project_name,
        save_dir=str(save_dir),
        name=run_name,
        entity=entity,
        mode='disabled' if disable_wandb else 'online',
    )
    logger.experiment.config.update({
        'project_name': project_name,
        'embeddings_path': str(embeddings_path),
        'valid_pdb_ids_train_path': str(valid_pdb_ids_train_path),
        'valid_pdb_ids_val_path': str(valid_pdb_ids_val_path),
        'pairfile_train_path': str(pairfile_train_path),
        'pairfile_val_path': str(pairfile_val_path),
        'save_dir': str(save_dir),
        'num_layers': num_layers,
        'num_heads': num_heads,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'temperature': temperature,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'max_epochs': max_epochs,
        'ckpt_every_k_epochs': ckpt_every_k_epochs,
        'num_workers': num_workers,
        'patience': patience,
        'run_id_number': run_id_number,
        'num_sanity_val_steps': num_sanity_val_steps,
    })

    # Build model checkpoint callback
    ckpt_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=2,
        monitor='val_loss',
        every_n_epochs=ckpt_every_k_epochs
    )

    # Build early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )

    # Building learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval='step')


    # Build trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        devices=1,
        deterministic=True,
        max_epochs=max_epochs,
        log_every_n_steps=25,
        callbacks=[ckpt_callback, early_stopping, lr_monitor],
        num_sanity_val_steps=num_sanity_val_steps
    )


    # Train model
    trainer.fit(
        model=model,
        datamodule=data_module
    )


if __name__ == '__main__':
    from tap import tapify

    tapify(embed_for_retrieval)
