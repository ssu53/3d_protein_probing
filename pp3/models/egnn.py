from typing import Callable
import math

import torch
import torch.nn as nn
from einops import rearrange


class SinusoidalEmbeddings(nn.Module):
    """A simple sinusoidal embedding layer."""

    def __init__(self, dim, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class EGNN_Layer(nn.Module):
    """A simple fully connected EGNN Layer."""

    def __init__(
        self,
        node_dim: int,
        dist_dim: int,
        proj_dim: int,
        message_dim: int,
        edge_dim: int,
        dropout: float = 0.0,
        use_sinusoidal: bool = True,
        activation: Callable = nn.ReLU,
        update_feats: bool = True,
        update_coors: bool = True,
    ) -> None:
        super().__init__()
        self.update_feats = update_feats
        self.update_coors = update_coors

        if not update_feats and not update_coors:
            raise ValueError(
                "At least one of update_feats or update_coors must be True."
            )

        if use_sinusoidal:
            self.dist_embedding = SinusoidalEmbeddings(dist_dim)
        else:
            self.dist_embedding = nn.Linear(1, dist_dim)  # type: ignore

        self.phi_e = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + dist_dim, message_dim),
            nn.Dropout(dropout),
            activation(),
            nn.Linear(message_dim, message_dim),
        )

        if update_coors:
            self.gate_x = nn.Parameter(torch.zeros(1))
            self.phi_x = nn.Sequential(
                nn.Linear(message_dim, proj_dim),
                nn.Dropout(dropout),
                activation(),
                nn.Linear(proj_dim, 1),
            )

        if update_feats:
            self.gate_h = nn.Parameter(torch.zeros(1))
            self.phi_h = nn.Sequential(
                nn.Linear(node_dim + message_dim, proj_dim),
                nn.Dropout(dropout),
                activation(),
                nn.Linear(proj_dim, node_dim),
            )

    def forward(
            self,
            embeddings: torch.Tensor,
            coords: torch.Tensor,
            padding_mask: torch.Tensor,
            edges: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute pairwise distances
        B, N = embeddings.shape[:2]
        rel_coors = coords.unsqueeze(2) - coords.unsqueeze(1)
        rel_dist = (rel_coors ** 2).sum(dim=-1)
        dists = self.dist_embedding(rearrange(rel_dist, 'b i j -> (b i j)'))
        dists = rearrange(dists, '(b i j) d -> b i j d', b=B, i=N, j=N)

        # Compute pairwise features
        feats1 = embeddings.unsqueeze(1).expand(-1, N, -1, -1)
        feats2 = embeddings.unsqueeze(2).expand(-1, -1, N, -1)

        if edges is not None:
            feats_pair = torch.cat((feats1, feats2, dists, edges), dim=-1)
        else:
            feats_pair = torch.cat((feats1, feats2, dists), dim=-1)

        # Compute messages
        m_ij = self.phi_e(feats_pair)
        self_mask = 1.0 - torch.eye(N).view(1, N, N, 1).to(m_ij)
        pad_mask_sum = (padding_mask.sum(dim=1, keepdim=True) - 1).unsqueeze(-1)

        # Compute coordinate update
        if self.update_coors:
            rel_coors = torch.nan_to_num(rel_coors / rel_dist.unsqueeze(-1)).detach()
            delta = rel_coors * self.phi_x(m_ij)
            delta = delta * padding_mask.view(-1, N, 1, 1)
            delta = delta * self_mask

            delta = delta.sum(dim=2) / pad_mask_sum
            coords = coords + self.gate_x * delta

        # Compute feature update
        if self.update_feats:
            m_ij = m_ij * padding_mask.view(-1, N, 1, 1)
            m_ij = m_ij * self_mask

            m_i = m_ij.sum(dim=2) / pad_mask_sum
            embeddings = embeddings + self.gate_h * self.phi_h(torch.cat((embeddings, m_i), dim=-1))

        return embeddings, coords


class EGNN(nn.Module):
    """A simple fully connected EGNN."""

    # TODO: check these defaults
    def __init__(
        self,
        node_dim: int,
        dist_dim: int,
        message_dim: int,
        proj_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.dist_embedding = SinusoidalEmbeddings(dist_dim)
        layers = [
            EGNN_Layer(
                node_dim=node_dim,
                edge_dim=dist_dim,
                dist_dim=dist_dim,
                message_dim=message_dim,
                proj_dim=proj_dim,
                dropout=dropout,
                update_coors=i < (num_layers - 1),
                update_feats=True,
            )
            for i in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(
            self,
            embeddings: torch.Tensor,
            coords: torch.Tensor,
            padding_mask: torch.Tensor
    ) -> torch.Tensor:
        # Compute pairwise distances in original coordinates
        B, N = embeddings.shape[:2]
        rel_coors = coords.unsqueeze(2) - coords.unsqueeze(1)
        rel_dist = (rel_coors**2).sum(dim=-1)
        dists = self.dist_embedding(rearrange(rel_dist, 'b i j -> (b i j)'))
        dists = rearrange(dists, '(b i j) d -> b i j d', b=B, i=N, j=N)

        # Add to edges so we always keep the original distances
        edges = dists

        # Run layers, updating both features and coordinates
        for layer in self.layers:
            embeddings, coords = layer(embeddings, coords, padding_mask, edges)

        return embeddings
