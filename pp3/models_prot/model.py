# %%

from typing import Optional, Literal

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim.lr_scheduler import ReduceLROnPlateau

from pp3.models_prot.modules import MeanAggSelfAttentionBlock 
from pp3.models_prot.data import get_train_dataloader

torch.autograd.set_detect_anomaly(True)


class ModelProt(pl.LightningModule):

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_channels: int,
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
    ):
        """
        :param num_channels: number of dimensions of the residue-level embedding
        """
        
        super().__init__()

        self.save_hyperparameters()

        assert temperature > 0.0, "The temperature must be a positive float!"
        assert similarity_func in {'cosine', 'euclidean'}, "Unknown similarity function."

        self.encoder = MeanAggSelfAttentionBlock(
            num_layers = num_layers,
            num_heads = num_heads,
            num_channels = num_channels,
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
    

    def info_nce_loss(self, batch, mode, log_rank_metrics=True):
        """
        Recall that embeddings are packed such that the first two form a positive pair
        and the remaining are negative examples
        """

        pdb_ids, embeddings, padding_mask = batch
        assert not torch.isnan(embeddings).any()
        assert not torch.isnan(padding_mask).any()

        # Encode each protein
        feats = self.encoder(x=embeddings, pad_mask=padding_mask,)
        # print(feats.shape)

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
            self.log(f'{mode}_acc_mean_pos', 1 + sim_argsort)
        
        return nll


    def training_step(self, batch, batch_idx):
        # for name, param in self.encoder.named_parameters():
            # print(torch.isfinite(param).all().item(), torch.isfinite(param.grad).all().item() if param.grad is not None else "none", name)
        loss = self.info_nce_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.info_nce_loss(batch, mode='val')
        self.log('val_loss', loss)
        return loss


# %%


def test():

    train_dataloader = get_train_dataloader()


    foo = 0
    for batch in train_dataloader:
        assert len(batch) == 3
        print(f"{len(batch[0])} {batch[1].shape} {batch[2].shape}")
        batch_ids = batch[0]
        foo += 1
        if foo == 2: break
    # %%

    model = ModelProt(
        num_layers=3,
        num_heads=2,
        num_channels=8,
    )

    foo = model.training_step(batch, batch_idx=0)

# %%
