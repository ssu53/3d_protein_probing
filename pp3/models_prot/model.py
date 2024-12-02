# %%

from typing import Optional, Literal

import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pp3.models.egnn import EGNN
from pp3.models.mlp import MLP
from pp3.models_prot.modules import MeanAggSelfAttentionBlock 
from pp3.models_prot.data import get_train_dataloader

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

torch.autograd.set_detect_anomaly(True)


class ModelProt(pl.LightningModule):

    def __init__(
        self,
        preencoder_num_layers: int, # 0 to turn off the EGNN layers
        preencoder_hidden_dim: int,
        preencoder_max_neighbors: int,
        preencoder_noise_std: float,
        num_layers: int,
        num_heads: int,
        input_dim: int,
        embedding_dim: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_rotary_layers: int = 1,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
        init_scale: float = 0.02,
        learning_rate: float = 1e-4,
        temperature: float = 1e-1,
        weight_decay: float = 0.0,
        similarity_func: Literal['cosine','euclidean'] = 'cosine',
        loss_func: Literal['l1','infonce'] = 'l1',
    ):
        """
        :param num_channels: number of dimensions of the residue-level embedding
        """
        
        super().__init__()

        self.save_hyperparameters()

        assert temperature > 0.0, "The temperature must be a positive float!"
        assert similarity_func in {'cosine', 'euclidean'}, "Unknown similarity function."

        if preencoder_num_layers > 0:
            self.preencoder_noise_std = preencoder_noise_std
            self.preencoder = EGNN(
                node_dim = input_dim,
                hidden_dim = preencoder_hidden_dim,
                num_layers = preencoder_num_layers,
                max_neighbors = preencoder_max_neighbors,
                dropout = dropout,
            )
            # last hidden layer has dim = node_dim = input_dim
        self.proj = MLP(
            input_dim = input_dim,
            hidden_dim = 0, # not used, 1-layer model
            output_dim = embedding_dim,
            num_layers = 1,
            last_layer_activation = False,
            dropout = dropout,
        )
        self.encoder = MeanAggSelfAttentionBlock(
            num_layers = num_layers,
            num_heads = num_heads,
            num_channels = embedding_dim,
            num_qk_channels = num_qk_channels,
            num_v_channels = num_v_channels,
            num_rotary_layers = num_rotary_layers,
            max_heads_parallel = max_heads_parallel,
            causal_attention = causal_attention,
            widening_factor = widening_factor,
            dropout = dropout,
            residual_dropout = residual_dropout,
            qkv_bias = qkv_bias,
            out_bias = out_bias,
            mlp_bias = mlp_bias,
            init_scale = init_scale,
        )
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.similarity_func = similarity_func
        self.loss_func = loss_func

        if loss_func == 'infonce': 
            self.loss_function = self.info_nce_loss_pairs
        elif loss_func == 'l1' or loss_func == 'l2':
            assert similarity_func == 'cosine'
            self.loss_function = self.supervised_loss
        else:
            raise NotImplementedError

        self.train_y = []
        self.train_y_hat = []
        self.val_y = []
        self.val_y_hat = []


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

    def forward(self, embeddings, coords, padding_mask, mode):
        if self.preencoder is not None:
            if mode == 'train' and self.preencoder_noise_std > 0.:
                coords = coords + \
                    torch.randn(coords.shape, device=coords.device) * self.preencoder_noise_std
            embeddings = self.preencoder(embeddings, coords, padding_mask)
        embeddings = self.proj(embeddings)
        feats = self.encoder(x=embeddings, pad_mask=padding_mask,)
        return feats
    

    def info_nce_loss(self, batch, mode, log_rank_metrics=True):
        """
        Batch should be packed such that the first two form a positive pair
        and the remaining are negative examples
        """

        embeddings = batch['embeddings']
        coords = batch['coords']
        padding_mask = batch['padding_mask']

        # Encode each protein
        feats = self.forward(embeddings, coords, padding_mask, mode)

        # Calculate similarity
        if self.similarity_func == 'cosine':
            similarity = nn.functional.cosine_similarity(feats[0], feats[1:], dim=-1)
        if self.similarity_func == 'euclidean':
            similarity = torch.sqrt(torch.sum(torch.square(feats[0] - feats[1:]), dim=-1))
            similarity = -similarity
            # similarity = 1. / similarity
        # print(similarity)
                
        # InfoNCE loss
        similarity = similarity / self.temperature
        nll = -similarity[0] + torch.logsumexp(similarity, dim=-1)
        # print(nll)

        # Log ranking metrics
        if log_rank_metrics:
            sim_argsort = similarity.argsort(descending=True).argmin() # find where 0th is
            self.log(f'{mode}_acc_top1', (sim_argsort == 0).float().item())
            self.log(f'{mode}_acc_top5', (sim_argsort < 5).float().item())
            self.log(f'{mode}_acc_mean_pos', float(1 + sim_argsort))
        
        return nll

    
    def info_nce_loss_pairs(self, batch, mode, log_rank_metrics=True):
        """
        Batch should be packed such that the first two form a positive pair
        and the remaining are negative examples
        """

        embeddings = batch['embeddings']
        coords = batch['coords']
        padding_mask = batch['padding_mask']

        # Encode each protein
        feats = self.forward(embeddings, coords, padding_mask, mode)

        # Calculate similarity
        if self.similarity_func == 'cosine':
            similarity = nn.functional.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        if self.similarity_func == 'euclidean':
            similarity = torch.sqrt(torch.sum(torch.square(feats[:, None, :] - feats[None, :, :]), dim=-1))
            similarity = -similarity
            # similarity = 1. / similarity
        # print(similarity)

        # Mask out similarity to itself
        self_mask = torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
        similarity.masked_fill_(self_mask, -9e15)
        # similarity.fill_diagonal_(-9e15)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=similarity.shape[0] // 2, dims=0)
        # pos_mask = torch.tensor([[0,1],[1,0]], dtype=torch.bool, device=similarity.device).repeat(similarity.shape[0]//2, 1, 1)
        # pos_mask = torch.block_diag(*pos_mask)
                
        # InfoNCE loss
        similarity = similarity / self.temperature
        nll = -similarity[pos_mask] + torch.logsumexp(similarity, dim=-1)
        nll = nll.mean()

        # Log ranking metrics
        if log_rank_metrics:
            comb_sim = torch.cat(
                [similarity[pos_mask][:, None], similarity.masked_fill(pos_mask, -9e15)],  # First position positive example
                dim=-1,
            )
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1) # find where 0th is
            self.log(f'{mode}_acc_top1', (sim_argsort == 0).float().mean())
            self.log(f'{mode}_acc_top5', (sim_argsort < 5).float().mean())
            self.log(f'{mode}_acc_mean_pos', 1 + sim_argsort.float().mean())
        
        return nll


    def supervised_loss(self, batch, mode):
        """
        Batch should be packed such that the first half and second half of the batch
        are pairs with targets
        """

        embeddings = batch['embeddings']
        coords = batch['coords']
        padding_mask = batch['padding_mask']
        targets = batch['targets']
        num_pairs = len(targets)

        # Encode each protein
        feats = self.forward(embeddings, coords, padding_mask, mode)
        assert len(feats) % 2 == 0
        assert len(feats) // 2 == num_pairs
        
        similarity = nn.functional.cosine_similarity(feats[:num_pairs], feats[num_pairs:], dim=-1)
        loss_l1 = nn.functional.l1_loss(similarity, targets)
        loss_l2 = nn.functional.mse_loss(similarity, targets)

        self.log(f'{mode}_l1', loss_l1)
        self.log(f'{mode}_l2', loss_l2)
        
        if mode == 'train':
            self.train_y += targets.detach().cpu().tolist()
            self.train_y_hat += similarity.detach().cpu().tolist()
        if mode == 'val':
            self.val_y += targets.detach().cpu().tolist()
            self.val_y_hat += similarity.detach().cpu().tolist()
        
        loss = loss_l1 if self.loss_func == 'l1' \
                else loss_l2 if self.loss_func == 'l2' else None

        return loss


    def training_step(self, batch, batch_idx):
        # for name, param in self.encoder.named_parameters():
            # print(torch.isfinite(param).all().item(), torch.isfinite(param.grad).all().item() if param.grad is not None else "none", name)
        loss = self.loss_function(batch, mode='train')
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.loss_function(batch, mode='val')
        self.log('val_loss', loss)
        return loss

    def evaluate(self, y, y_hat, mode):
        self.log(f'{mode}_y_mean', np.mean(y))
        self.log(f'{mode}_y_hat_mean', np.mean(y_hat))
        self.log(f'{mode}_y_std', np.std(y))
        self.log(f'{mode}_y_hat_std', np.std(y_hat))
        self.log(f'{mode}_r2', r2_score(y, y_hat))
        self.log(f'{mode}_pearson', pearsonr(y, y_hat).statistic)
        self.log(f'{mode}_spearman', spearmanr(y, y_hat).statistic)

    def on_train_epoch_end(self) -> None:
        pass
    
    def on_validation_epoch_end(self) -> None:

        if len(self.train_y) > 0 and len(self.train_y_hat) > 0:
            # train
            self.evaluate(
                y=self.train_y,
                y_hat=self.train_y_hat,
                mode='train',
            )

            self.train_y = []
            self.train_y_hat = []

        # val
        self.evaluate(
            y=self.val_y,
            y_hat=self.val_y_hat,
            mode='val',
        )

        self.val_y = []
        self.val_y_hat = []


