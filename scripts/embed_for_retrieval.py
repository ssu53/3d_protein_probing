from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

import pp3.models_prot.default_paths as default_paths
from pp3.models_prot.data import ProteinPairDataModule
from pp3.models_prot.model import ModelProt



def embed_for_retrieval(
    project_name: str,
    save_dir: Path,
    preencoder_type,
    preencoder_num_layers: int,
    preencoder_hidden_dim: int,
    preencoder_max_neighbors: int,
    preencoder_noise_std: float,
    num_layers: int,
    num_heads: int,
    embedding_dim: int,
    embeddings_path: Path,
    proteins_path: Path,
    valid_pdb_ids_train_path: Path | None = None,
    valid_pdb_ids_val_path: Path | None = None,
    pairfile_train_path: Path | None = None,
    pairfile_val_path: Path | None = None,
    temperature: float = 1e-1,
    similarity_func: str = 'cosine',
    loss_func: str = 'infonce',
    loss_thresh: float | None = None,
    learning_rate: float = 1e-4,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    batch_size: int = 32,
    num_workers: int = 4,
    max_epochs: int = 500,
    patience: int = 25,
    accumulate_grad_batches: int = 1,
    ckpt_every_k_epochs: int = 1,
    # ckpt_every_n_train_steps: int | None = 2500,
    model_ckpt_path: Path | None = None,
    num_sanity_val_steps: int = 2,
    val_check_interval: float = 1.0,
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
    if loss_func != 'infonce':
        print("Ignoring temperature!")
        run_name = f'{embeddings_name}_{loss_func}_{embedding_dim}d_{preencoder_num_layers}pL_{num_layers}L'
    else:
        run_name = f'{embeddings_name}_{loss_func}_{num_layers}L_{learning_rate}lr_{batch_size}bs_{temperature}temp'
    if run_name_suffix:
        run_name += f'_{run_name_suffix}'

    save_dir = save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)


    # Get default paths
    if proteins_path is None:
        proteins_path = default_paths.proteins_path()
    if valid_pdb_ids_train_path is None:
        valid_pdb_ids_train_path = default_paths.get_valid_pdb_ids_train_path()
    if valid_pdb_ids_val_path is None:
        valid_pdb_ids_val_path = default_paths.get_valid_pdb_ids_val_path()
    if pairfile_train_path is None:
        if loss_func == 'infonce': 
            pairfile_train_path = default_paths.get_pairfile_train_path() # contrastive
        else:
            pairfile_train_path = default_paths.get_tmaln_data_train_path() # supervised
    if pairfile_val_path is None:
        if loss_func == 'infonce': 
            pairfile_val_path = default_paths.get_pairfile_val_path() # contrastive
        else: 
            pairfile_val_path = default_paths.get_tmaln_data_val_path() # supervised


    # Random seed
    pl.seed_everything(0)

    # Build data module
    data_module = ProteinPairDataModule(
        embeddings_path=embeddings_path,
        proteins_path=proteins_path,
        pdb_ids_train_path=valid_pdb_ids_train_path,
        pdb_ids_val_path=valid_pdb_ids_val_path,
        pairfile_train_path=pairfile_train_path,
        pairfile_val_path=pairfile_val_path,
        is_supervised=loss_func!='infonce',
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_module.setup()


    # Build model
    if model_ckpt_path is None:
        model = ModelProt(
            preencoder_type=preencoder_type,
            preencoder_num_layers=preencoder_num_layers,
            preencoder_hidden_dim=preencoder_hidden_dim,
            preencoder_max_neighbors=preencoder_max_neighbors,
            preencoder_noise_std=preencoder_noise_std,
            num_layers=num_layers,
            num_heads=num_heads,
            input_dim=data_module.embedding_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            temperature=temperature,
            weight_decay=weight_decay,
            similarity_func=similarity_func,
            loss_func=loss_func,
            loss_thresh=loss_thresh,
        )
    else:
        model = ModelProt.load_from_checkpoint(model_ckpt_path)
        print('Loaded from checkpoint; ignoring many model hyperparameters!')

        assert model.similarity_func == similarity_func
        assert model.loss_func == loss_func

        model.preencoder_noise_std = preencoder_noise_std
        model.learning_rate = learning_rate
        model.temperature = temperature
        model.weight_decay = weight_decay

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
        'proteins_path': str(proteins_path),
        'valid_pdb_ids_train_path': str(valid_pdb_ids_train_path),
        'valid_pdb_ids_val_path': str(valid_pdb_ids_val_path),
        'pairfile_train_path': str(pairfile_train_path),
        'pairfile_val_path': str(pairfile_val_path),
        'save_dir': str(save_dir),
        'embedding_dim': embedding_dim,
        'preencoder_type': preencoder_type,
        'preencoder_num_layers': preencoder_num_layers,
        'preencoder_hidden_dim': preencoder_hidden_dim,
        'preencoder_max_neighbors': preencoder_max_neighbors,
        'preencoder_noise_std': preencoder_noise_std,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'similarity_func': similarity_func,
        'loss_func': loss_func,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'temperature': temperature,
        'loss_func': loss_func,
        'loss_thresh': loss_thresh,
        'similarity_func': similarity_func,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'max_epochs': max_epochs,
        'accumulate_grad_batches': accumulate_grad_batches,
        'ckpt_every_k_epochs': ckpt_every_k_epochs,
        'num_workers': num_workers,
        'patience': patience,
        'run_id_number': run_id_number,
        'num_sanity_val_steps': num_sanity_val_steps,
        'model_ckpt_path': model_ckpt_path,
    })
    print(logger.experiment.config)

    # Build model checkpoint callback
    ckpt_callback = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=2,
        save_last=True,
        monitor='val_loss/dataloader_idx_0',
        every_n_epochs=ckpt_every_k_epochs,
        # every_n_train_steps=ckpt_every_n_train_steps,
        save_on_train_epoch_end=False, # ckpt after each validation
    )

    # Build early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss/dataloader_idx_0',
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
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=max_epochs,
        log_every_n_steps=25,
        val_check_interval=val_check_interval,
        callbacks=[ckpt_callback, early_stopping, lr_monitor],
        num_sanity_val_steps=num_sanity_val_steps
    )

    # trainer.validate(
    #     model=model,
    #     ckpt_path='last',
    #     datamodule=data_module,
    # )

    # return

    # Train model
    trainer.fit(
        # ckpt_path=model_ckpt_path,
        model=model,
        datamodule=data_module
    )


if __name__ == '__main__':
    from tap import tapify

    tapify(embed_for_retrieval)
