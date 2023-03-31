import math

import torch
import torch.nn as nn

from minifold.utils.linear import Linear


def is_fp16_enabled():
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()
    return fp16_enabled


def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t, no_dims):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def rot_vec_mul(
    r: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
        Applies a rotation to a vector. Written out by hand to avoid transfer
        to avoid AMP downcasting.
        Args:
            r: [*, 3, 3] rotation matrices
            t: [*, 3] coordinate tensors
        Returns:
            [*, 3] rotated coordinates
    """
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


class IPA(nn.Module):
    def __init__(
        self,
        c_s: int = 128,
        c_hidden: int = 16,
        no_heads: int = 12,
        no_qk_points: int = 4,
        no_v_points: int = 8,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.c_s = c_s
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        with torch.no_grad():
            softplus_inverse_1 = 0.541324854612918
            self.head_weights.fill_(softplus_inverse_1)

        concat_out_dim = self.no_heads * (self.c_hidden + self.no_v_points * 4)
        self.linear_out = Linear(concat_out_dim, self.c_s)
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(self, s, R, t, mask) -> torch.Tensor:

        # Compute query, key, value
        q = self.linear_q(s)
        kv = self.linear_kv(s)
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # Compute query, key, value points
        q_pts = self.linear_q_points(s)
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = rot_vec_mul(R[:, :, None], q_pts) + t[:, :, None]
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        kv_pts = self.linear_kv_points(s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = rot_vec_mul(R[:, :, None], kv_pts) + t[:, :, None]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        # [*, H, N_res, N_res]
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    permute_final_dims(q.float(), (1, 0, 2)),  # [*, H, N_res, C_hidden]
                    permute_final_dims(k.float(), (1, 2, 0)),  # [*, H, C_hidden, N_res]
                )
        else:
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
                permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
            )

        a = a * math.sqrt(1.0 / (3 * self.c_hidden))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = 1e5 * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        o = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)
        o_pt = torch.sum(
            (
                a[..., None, :, :, None]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = rot_vec_mul(R[:, :, None, None].transpose(-1, -2), o_pt) - t[:, :, None, None]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1).to(
                dtype=s.dtype
            )
        )
        return s


class BackboneUpdate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = Linear(dim, 6, init="final")

    def forward(self, s):
        # Compute update
        update = self.linear(s)

        # Compute norm
        ones = torch.ones_like(update[:, :, :1])
        norm = torch.cat((ones, update[:, :, :3]), dim=-1)
        norm = torch.norm(norm, dim=-1)

        # Compute quaternion
        a = 1 / norm
        b = update[:, :, 0] / norm
        c = update[:, :, 1] / norm
        d = update[:, :, 2] / norm
        t = update[:, :, 3:]

        # Compute rotation matrix
        a2, b2, c2, d2 = a * a, b * b, c * c, d * d
        ab, ac, ad, bc, bd, cd = a * b, a * c, a * d, b * c, b * d, c * d

        row1 = torch.stack([a2 + b2 - c2 - d2, 2 * (bc - ad), 2 * (bd + ac)], dim=2)
        row2 = torch.stack([2 * (bc + ad), a2 - b2 + c2 - d2, 2 * (cd - ab)], dim=2)
        row3 = torch.stack([2 * (bd - ac), 2 * (cd + ab), a2 - b2 - c2 + d2], dim=2)

        R = torch.stack([row1, row2, row3], dim=2)
        return R, t


class StructureModule(nn.Module):
    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.dim = dim
        self.layers = layers

        self.input_fc = Linear(dim, dim)
        self.input_ln = nn.LayerNorm(dim)
        self.ipa = IPA(dim, dim)
        self.ipa_ln = nn.LayerNorm(dim)
        self.transition_fc = nn.Sequential(
            Linear(dim, dim, init="relu"),
            nn.ReLU(),
            Linear(dim, dim, init="relu"),
            nn.ReLU(),
            Linear(dim, dim, init="final"),
        )
        self.transition_ln = nn.LayerNorm(dim)
        self.backbone_update = BackboneUpdate(dim)

    def forward(self, data, transform, mask):
        # Initialize frames
        R = torch.eye(3, device=data.device)
        t = torch.zeros(3, device=data.device)

        # Expand batch & sequence dimensions
        B, N = data.shape[:2]
        R = R[None, None, :, :].repeat(B, N, 1, 1)
        t = t[None, None, :].repeat(B, N, 1)

        # Compute input fc
        data = self.input_ln(data)
        data = self.input_fc(data)

        for i in range(self.layers):
            # Compute IPA
            data = data + self.ipa(data, R, t, mask=mask)
            data = self.ipa_ln(data)

            # Compute transition
            data = data + self.transition_fc(data)
            data = self.transition_ln(data)

            # Update backbone
            R_u, t_u = self.backbone_update(data)

            # Compute new frames
            R = R.matmul(R_u)
            t = t + rot_vec_mul(R, t_u)

            if i < self.layers - 1:
                R = R.detach()

        positions = rot_vec_mul(R[:, :, None], transform) + t[:, :, None]
        return positions, R, t
