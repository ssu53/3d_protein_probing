import math
from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from e3nn.nn import BatchNorm
from einops import rearrange


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py"""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2)).float()


class TensorProductConvLayer(torch.nn.Module):
    def __init__(
        self,
        in_irreps,
        in_tp_irreps,
        out_tp_irreps,
        sh_irreps,
        out_irreps,
        n_edge_features,
        batch_norm=False,
        dropout=0.0,
        node_feature_dim=4,
        fc_dim=32,
        lin_self=False,
        attention=False,
    ):
        super(TensorProductConvLayer, self).__init__()

        # consider attention...
        if attention:
            raise NotImplementedError

        self.nf = node_feature_dim
        self.lin_in = o3.Linear(in_irreps, in_tp_irreps, internal_weights=True)
        self.tp = o3.FullyConnectedTensorProduct(
            in_tp_irreps, sh_irreps, out_tp_irreps, shared_weights=False
        )

        self.lin_out = o3.Linear(out_tp_irreps, out_irreps, internal_weights=True)
        if lin_self:
            self.lin_self = o3.Linear(in_irreps, out_irreps, internal_weights=True)
        else:
            self.lin_self = False

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, self.tp.weight_numel),
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(
        self, node_attr, edge_attr, edge_sh, mask, residual=True, apply_mask=True
    ):
        B, N = mask.shape
        node_attr_in = self.lin_in(node_attr)
        tp = self.tp(node_attr_in[:, :, None], edge_sh, self.fc(edge_attr))
        if apply_mask:
            tp = tp * mask.view(B, 1, N, 1)
            self_mask = 1 - torch.eye(N, device=edge_sh.device).view(1, N, N, 1)
            tp = tp * self_mask
            out = tp.sum(dim=2) / (mask.sum(-1).view(B, 1, 1) - 1)
        else:
            out = tp.mean(dim=2)

        out = self.lin_out(out)

        if not residual:
            return out
        if self.lin_self:
            out = out + self.lin_self(node_attr)
        else:
            out = out + F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class TFN(torch.nn.Module):
    def __init__(
        self,
        node_dim: int,
        num_layers: int,
        sh_lmax: int = 1,
        ns: int = 32,
        nv: int = 4,
        ntps: int = 16,
        ntpv: int = 4,
        fc_dim: int = 128,
        edge_dim: int = 50,
        pos_emb_dim: int = 0,
        radius_emb_dim: int = 50,
        radius_emb_type: str = "gaussian",
        radius_emb_max: int = 50,
        order: int = 1,
        lin_nf: int = 1,
        lin_self: bool = False,
        parity: int = 1,
        dropout: float = 0,
        max_neighbors: int | None = None,
    ) -> None:
        super(TFN, self).__init__()

        self.pos_emb_dim = pos_emb_dim
        self.parity = parity
        self.ns = ns
        self.num_layers = num_layers
        self.max_neighbors = max_neighbors

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.node_embedding_in = nn.Sequential(
            nn.Linear(node_dim, ns),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(ns, ns),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(radius_emb_dim + pos_emb_dim + edge_dim, ns),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(ns, ns),
        )

        # self.resi_node_norm = nn.LayerNorm(node_dim)
        # self.resi_edge_norm = nn.LayerNorm(2 * edge_dim)
        if radius_emb_type == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0, radius_emb_max**0.5, radius_emb_dim
            )
        elif radius_emb_type == "sinusoidal":
            self.distance_expansion = lambda x: sinusoidal_embedding(  # type: ignore
                10000 * x / radius_emb_max**0.5, radius_emb_dim
            )
        else:
            raise ValueError("Unknown radius embedding type")

        if order == 2:
            irrep_seq = [
                [(0, 1)],
                [(0, 1), (1, -1), (2, 1)],
                [(0, 1), (1, -1), (2, 1), (1, 1), (2, -1)],
                [(0, 1), (1, -1), (2, 1), (1, 1), (2, -1), (0, -1)],
            ]
        else:
            irrep_seq = [
                [(0, 1)],
                [(0, 1), (1, -1)],
                [(0, 1), (1, -1), (1, 1)],
                [(0, 1), (1, -1), (1, 1), (0, -1)],
            ]

        def fill_mults(ns, nv, irs, is_in=False):
            irreps = [
                (ns, (l, p)) if (l == 0 and p == 1) else [nv, (l, p)]
                for l, p in irs  # noqa: E741
            ]

            return irreps

        conv_layers = []
        update_layers = []
        for i in range(num_layers):
            in_seq, out_seq = (
                irrep_seq[min(i, len(irrep_seq) - 1)],
                irrep_seq[min(i + 1, len(irrep_seq) - 1)],
            )
            in_irreps = fill_mults(ns, nv, in_seq, is_in=(i == 0))
            out_irreps = fill_mults(ns, nv, out_seq)
            in_tp_irreps = fill_mults(ntps, ntpv, in_seq)
            out_tp_irreps = fill_mults(ntps, ntpv, out_seq)
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                in_tp_irreps=in_tp_irreps,
                out_tp_irreps=out_tp_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                batch_norm=False,
                node_feature_dim=min(ns, lin_nf),
                fc_dim=fc_dim,
                lin_self=lin_self,
            )
            conv_layers.append(layer)
            update_layers.append(
                o3.FullyConnectedTensorProduct(
                    out_irreps,
                    out_irreps,
                    "1x1o + 1x1e" if parity else "1x1o",
                    internal_weights=True,
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.update_layers = nn.ModuleList(update_layers)
        self.node_embedding_out = nn.Sequential(
            nn.Linear(60, 60),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(60, node_dim),
        )

    def compute_edge_attr(self, edge_vec, edge_attr=None, edge_pos_emb=None):
        B, N, M, _ = edge_vec.shape

        # Distance embedding
        edges = self.distance_expansion(edge_vec.norm(dim=-1).flatten() ** 0.5).reshape(
            B, N, M, -1
        )

        # Add positional encoding
        if edge_pos_emb is not None:
            edges = torch.cat(
                [edges, edge_pos_emb.expand(B, -1, -1, -1)],
                -1,
            )

        # Spherical harmonics
        if edge_attr is not None:
            edges = torch.cat(
                [edges, edge_attr],
                -1,
            )

        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        ).float()

        # Edge features
        edges = self.edge_embedding(edges)
        return edges, edge_sh

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute pairwise distances in original coordinates
        B, N = embeddings.shape[:2]
        rel_coors = coords.unsqueeze(2) - coords.unsqueeze(1)
        rel_dist = (rel_coors**2).sum(dim=-1) ** 0.5
        dists = self.distance_expansion(rearrange(rel_dist, "b i j -> (b i j)"))
        dists = rearrange(dists, "(b i j) d -> b i j d", b=B, i=N, j=N)

        # Create neighbors list, with max_neighbors neighbors
        neighbor_ids = None
        padding_mask = padding_mask.float()
        if self.max_neighbors is not None:
            assert self.max_neighbors < N
            # set padding to max distance so they are always last
            pair_mask = padding_mask.unsqueeze(2) * padding_mask.unsqueeze(1)
            rel_dist = rel_dist + (1 - pair_mask) * rel_dist.max()
            neighbor_ids = torch.argsort(rel_dist, dim=-1)
            # Increment by 1 to ignore self
            neighbor_ids = neighbor_ids[:, :, 1 : self.max_neighbors + 1]
            edges_in = dists[
                torch.arange(B)[:, None, None],
                torch.arange(N)[None, :, None],
                neighbor_ids,
            ]
        else:
            # Add to edges so we always keep the original distances
            edges_in = dists

        embeddings = self.node_embedding_in(embeddings)

        # Relative positional embedding
        # Consider adding this back in
        # idx = torch.arange(N, device=coords.device)
        # idx = idx.unsqueeze(0) - idx.unsqueeze(1)
        # edge_pos_emb = sinusoidal_embedding(idx.flatten(), self.pos_emb_dim)
        # edge_pos_emb = edge_pos_emb.reshape(1, N, N, -1)

        for i in range(self.num_layers):
            if neighbor_ids is not None:
                M = neighbor_ids.shape[-1]
                n_embeddings = embeddings[
                    torch.arange(B)[:, None], neighbor_ids.reshape(B, -1)
                ].reshape(B, N, M, -1)
                n_coords = coords[
                    torch.arange(B)[:, None], neighbor_ids.reshape(B, -1)
                ].reshape(B, N, M, -1)
                feats2 = n_embeddings
            else:
                M = N
                n_embeddings = embeddings
                n_coords = coords.unsqueeze(1)
                feats2 = n_embeddings.unsqueeze(1).expand(-1, N, -1, -1)

            rel_coors = coords.unsqueeze(2) - n_coords
            edges, edge_sh = self.compute_edge_attr(rel_coors, edge_attr=edges_in)

            # Compute edge features
            edge_attr_ = torch.cat(
                [
                    edges,
                    embeddings.unsqueeze(2).expand(-1, -1, M, -1)[..., : self.ns],
                    feats2[..., : self.ns],
                ],
                -1,
            )

            # Update node features
            layer = self.conv_layers[i]
            apply_mask = self.max_neighbors is None
            embeddings = layer(
                embeddings, edge_attr_, edge_sh, padding_mask, apply_mask=apply_mask
            )

            # Update node positions
            update = self.update_layers[i]
            dX = update(embeddings, embeddings)
            if self.parity:
                dX = dX.view(B, N, 2, 3).mean(-2)

            coords = coords + dX

        import pdb; pdb.set_trace()
        embeddings = self.node_embedding_out(embeddings)
        return embeddings
