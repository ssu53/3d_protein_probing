import math
import numpy as np
from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from e3nn.nn import BatchNorm


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


class FasterTensorProduct(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, **kwargs):
        super().__init__()
        for ir in in_irreps:
            m, (l, p) = ir
            assert l in [0, 1], "Higher order in irreps are not supported"
        for ir in out_irreps:
            m, (l, p) = ir
            assert l in [0, 1], "Higher order out irreps are not supported"
        assert o3.Irreps(sh_irreps) == o3.Irreps(
            "1x0e+1x1o"
        ), "sh_irreps don't look like 1st order spherical harmonics"
        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)

        in_muls = {"0e": 0, "1o": 0, "1e": 0, "0o": 0}
        out_muls = {"0e": 0, "1o": 0, "1e": 0, "0o": 0}
        for m, ir in self.in_irreps:
            in_muls[str(ir)] = m
        for m, ir in self.out_irreps:
            out_muls[str(ir)] = m

        self.weight_shapes = {
            "0e": (in_muls["0e"] + in_muls["1o"], out_muls["0e"]),
            "1o": (in_muls["0e"] + in_muls["1o"] + in_muls["1e"], out_muls["1o"]),
            "1e": (in_muls["1o"] + in_muls["1e"] + in_muls["0o"], out_muls["1e"]),
            "0o": (in_muls["1e"] + in_muls["0o"], out_muls["0o"]),
        }
        self.weight_numel = sum(a * b for (a, b) in self.weight_shapes.values())

    def forward(self, in_, sh, weight):
        in_dict, out_dict = {}, {"0e": [], "1o": [], "1e": [], "0o": []}
        for (m, ir), sl in zip(self.in_irreps, self.in_irreps.slices()):
            in_dict[str(ir)] = in_[..., sl]
            if ir[0] == 1:
                in_dict[str(ir)] = in_dict[str(ir)].reshape(
                    list(in_dict[str(ir)].shape)[:-1] + [-1, 3]
                )
        sh_0e, sh_1o = sh[..., 0], sh[..., 1:]
        if "0e" in in_dict:
            out_dict["0e"].append(in_dict["0e"] * sh_0e.unsqueeze(-1))
            out_dict["1o"].append(in_dict["0e"].unsqueeze(-1) * sh_1o.unsqueeze(-2))
        if "1o" in in_dict:
            out_dict["0e"].append(
                (in_dict["1o"] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3)
            )
            out_dict["1o"].append(in_dict["1o"] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict["1e"].append(
                torch.cross(
                    in_dict["1o"].expand(-1, in_dict["1o"].shape[2], -1, -1, -1),
                    sh_1o.unsqueeze(-2).expand(-1, -1, -1, in_dict["1o"].shape[-2], -1),
                    dim=-1,
                )
                / np.sqrt(2)
            )
        if "1e" in in_dict:
            out_dict["1o"].append(
                torch.cross(
                    in_dict["1e"].expand(-1, in_dict["1e"].shape[2], -1, -1, -1),
                    sh_1o.unsqueeze(-2).expand(-1, -1, -1, in_dict["1e"].shape[-2], -1),
                    dim=-1,
                )
                / np.sqrt(2)
            )
            out_dict["1e"].append(in_dict["1e"] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict["0o"].append(
                (in_dict["1e"] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3)
            )
        if "0o" in in_dict:
            out_dict["1e"].append(in_dict["0o"].unsqueeze(-1) * sh_1o.unsqueeze(-2))
            out_dict["0o"].append(in_dict["0o"] * sh_0e.unsqueeze(-1))

        weight_dict = {}
        start = 0
        for key in self.weight_shapes:
            in_, out = self.weight_shapes[key]
            weight_dict[key] = weight[..., start : start + in_ * out].reshape(
                list(weight.shape)[:-1] + [in_, out]
            ) / np.sqrt(in_)
            start += in_ * out

        if out_dict["0e"]:
            out_dict["0e"] = torch.cat(out_dict["0e"], dim=-1)
            out_dict["0e"] = torch.matmul(
                out_dict["0e"].unsqueeze(-2), weight_dict["0e"]
            ).squeeze(-2)

        if out_dict["1o"]:
            out_dict["1o"] = torch.cat(out_dict["1o"], dim=-2)
            out_dict["1o"] = (
                out_dict["1o"].unsqueeze(-2) * weight_dict["1o"].unsqueeze(-1)
            ).sum(-3)
            out_dict["1o"] = out_dict["1o"].reshape(
                list(out_dict["1o"].shape)[:-2] + [-1]
            )

        if out_dict["1e"]:
            out_dict["1e"] = torch.cat(out_dict["1e"], dim=-2)
            out_dict["1e"] = (
                out_dict["1e"].unsqueeze(-2) * weight_dict["1e"].unsqueeze(-1)
            ).sum(-3)
            out_dict["1e"] = out_dict["1e"].reshape(
                list(out_dict["1e"].shape)[:-2] + [-1]
            )

        if out_dict["0o"]:
            out_dict["0o"] = torch.cat(out_dict["0o"], dim=-1)
            # out_dict['0o'] = (out_dict['0o'].unsqueeze(-1) * weight_dict['0o']).sum(-2)
            out_dict["0o"] = torch.matmul(
                out_dict["0o"].unsqueeze(-2), weight_dict["0o"]
            ).squeeze(-2)

        out = []
        for _, ir in self.out_irreps:
            out.append(out_dict[str(ir)])
        return torch.cat(out, dim=-1)


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
        use_fast=False,
    ):
        super(TensorProductConvLayer, self).__init__()

        # consider attention...
        if attention:
            raise NotImplementedError

        self.nf = node_feature_dim
        self.lin_in = o3.Linear(in_irreps, in_tp_irreps, internal_weights=True)
        if use_fast:
            self.tp = FasterTensorProduct(in_tp_irreps, sh_irreps, out_tp_irreps)
        else:
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
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, self.tp.weight_numel),
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(
        self,
        node_attr,
        edge_attr,
        edge_sh,
        mask,
        residual=True,
    ):
        B, N = mask.shape
        node_attr_in = self.lin_in(node_attr)

        tp = self.tp(node_attr_in[:, None], edge_sh, self.fc(edge_attr))
        tp = tp * mask.view(B, 1, N, 1)
        self_mask = 1 - torch.eye(N, device=edge_sh.device).view(1, N, N, 1)
        tp = tp * self_mask
        out = tp.sum(dim=2) / (mask.sum(-1).view(B, 1, 1) - 1)

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
        sh_lmax: int,
        ns: int,
        nv: int,
        ntps: int,
        ntpv: int,
        fc_dim: int = 128,
        edge_dim: int = 0,
        pos_emb_dim: int = 16,
        radius_emb_dim: int = 50,
        radius_emb_type: str = "gaussian",
        radius_emb_max: int = 50,
        order: int = 1,
        lin_nf: int = 1,
        lin_self: bool = False,
        parity: int = 1,
        attention: bool = False,
        use_fast: bool = False,
    ):
        super(TFN, self).__init__()

        self.pos_emb_dim = pos_emb_dim
        self.parity = parity
        self.ns = ns
        self.num_layers = num_layers

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns),
            nn.ReLU(),
            nn.Linear(ns, ns),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(radius_emb_dim + pos_emb_dim + edge_dim, ns),
            nn.ReLU(),
            nn.Linear(ns, ns),
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
                attention=attention,
                use_fast=use_fast,
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

    def compute_edge_attr(self, pos, edge_pos_emb, edge_attr=None):
        B, N, _ = pos.shape

        # Distance embedding
        edge_vec = pos.unsqueeze(1) - pos.unsqueeze(2)
        edge_length_emb = self.distance_expansion(
            edge_vec.norm(dim=-1).flatten() ** 0.5
        ).reshape(B, N, N, -1)

        # Spherical harmonics
        if edge_attr is not None:
            edges = torch.cat(
                [edge_length_emb, edge_pos_emb.expand(B, -1, -1, -1), edge_attr],
                -1,
            )
        else:
            edges = torch.cat([edge_length_emb, edge_pos_emb.expand(B, -1, -1, -1)], -1)

        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        ).float()

        # Edge features
        edges = self.edge_embedding(edges)

        return edges, edge_sh

    def forward(
        self,
        node_attr,
        pos,
        mask,
        edge_attr=None,
    ):
        B, N, _ = pos.shape
        node_attr = self.node_embedding(node_attr)

        # Relative positional embedding
        idx = torch.arange(N, device=pos.device)
        idx = idx.unsqueeze(0) - idx.unsqueeze(1)
        edge_pos_emb = sinusoidal_embedding(idx.flatten(), self.pos_emb_dim)
        edge_pos_emb = edge_pos_emb.reshape(1, N, N, -1)

        for i in range(self.num_layers):
            edges, edge_sh = self.compute_edge_attr(pos, edge_pos_emb, edge_attr)

            # Compute edge features
            edge_attr_ = torch.cat(
                [
                    edges,
                    node_attr[..., None, : self.ns].expand(-1, -1, N, -1),
                    node_attr[..., None, :, : self.ns].expand(-1, N, -1, -1),
                ],
                -1,
            )

            # Update node features
            layer = self.conv_layers[i]
            node_attr = layer(node_attr, edge_attr_, edge_sh, mask)

            # Update node positions
            update = self.update_layers[i]
            dX = update(node_attr, node_attr)
            if self.parity:
                dX = dX.view(B, N, 2, 3).mean(-2)

            pos = pos + dX

        return node_attr
