"""
Adapted from perceiver-io
"""
# %%

from typing import List, Optional, Tuple

import torch
from torch import nn as nn
from einops import rearrange

from pp3.models_prot.position import RotaryPositionEmbedding
from pp3.models_prot.utils import ModuleOutput, Residual, init_parameters


KVCache = Tuple[torch.Tensor, torch.Tensor]



class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Multi-head attention as specified in https://arxiv.org/abs/2107.14795 Appendix E plus support for rotary
        position embeddings (https://arxiv.org/abs/2104.09864) and causal attention. Causal attention requires
        queries and keys to be right-aligned, if they have different length.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of query and key channels. Default is number `num_q_input_channels`
        :param num_v_channels: Number of value channels. Default is `num_qk_channels`.
        :param num_output_channels: Number of output channels. Default is `num_q_input_channels`
        :param max_heads_parallel: Maximum number of heads to be processed in parallel. Default is `num_heads`.
        :param causal_attention: Whether to apply a causal attention mask. Default is `False`.
        :param dropout: Dropout probability for attention matrix values. Default is `0.0`
        :param qkv_bias: Whether to use a bias term for query, key and value projections. Default is `True`.
        :param qkv_bias: Whether to use a bias term for output projection. Default is `True`.
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head**-0.5
        self.num_heads = num_heads
        self.num_qk_channels = num_qk_channels
        self.num_v_channels = num_v_channels
        self.causal_attention = causal_attention

        if max_heads_parallel is None:
            self.max_heads_parallel = num_heads
        else:
            self.max_heads_parallel = max_heads_parallel

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=qkv_bias)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels, bias=qkv_bias)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels, bias=qkv_bias)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels, bias=out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb_q: Optional[RotaryPositionEmbedding] = None,
        rot_pos_emb_k: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        """...

        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length and D the
                number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence length and C
                are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param rot_pos_emb_q: Applies a rotary position embedding to query i.e. if defined, rotates the query.
        :param rot_pos_emb_k: Applies a rotary position embedding to key i.e. if defined, rotates the key.
        :param kv_cache: cache with past keys and values.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length and F the
                number of output channels (= `num_output_channels`)
        """

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
            kv_cache = (k, v)

        q, k, v = (rearrange(x, "b n (h c) -> b h n c", h=self.num_heads) for x in [q, k, v])
        q = q * self.dp_scale

        if rot_pos_emb_q is not None:
            q = rot_pos_emb_q.rotate(q)

        if rot_pos_emb_k is not None:
            k = rot_pos_emb_k.rotate(k)

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, "b j -> b 1 1 j")

        if self.causal_attention:
            i = q.shape[2]
            j = k.shape[2]

            # If q and k have different length, causal masking only works if they are right-aligned.
            causal_mask = torch.ones((i, j), device=x_q.device, dtype=torch.bool).triu(j - i + 1)

        o_chunks = []

        # Only process a given maximum number of heads in
        # parallel, using several iterations, if necessary.
        for q_chunk, k_chunk, v_chunk in zip(
            q.split(self.max_heads_parallel, dim=1),
            k.split(self.max_heads_parallel, dim=1),
            v.split(self.max_heads_parallel, dim=1),
        ):
            attn = torch.einsum("b h i c, b h j c -> b h i j", q_chunk, k_chunk)
            attn_max_neg = -torch.finfo(attn.dtype).max

            if pad_mask is not None:
                attn.masked_fill_(pad_mask, attn_max_neg)

            if self.causal_attention:
                attn.masked_fill_(causal_mask, attn_max_neg)

            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)

            o_chunk = torch.einsum("b h i j, b h j c -> b h i c", attn, v_chunk)
            o_chunks.append(o_chunk)

        o = torch.cat(o_chunks, dim=1)
        o = rearrange(o, "b h n c -> b n (h c)", h=self.num_heads)
        o = self.o_proj(o)

        return ModuleOutput(last_hidden_state=o, kv_cache=kv_cache)



class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ):
        """Pre-layer norm self-attention (see `MultiHeadAttention` and for attention details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        """Pre-layer-norm self-attention of input `x`."""
        x = self.norm(x)
        return self.attention(
            x,
            x,
            pad_mask=pad_mask,
            rot_pos_emb_q=rot_pos_emb,
            rot_pos_emb_k=rot_pos_emb,
            kv_cache=kv_cache,
        )



class AbstractAttentionLayer(nn.Sequential):
    def empty_kv_cache(self, x) -> KVCache:
        k_cache = torch.empty(x.shape[0], 0, self.num_qk_channels, dtype=x.dtype, device=x.device)
        v_cache = torch.empty(x.shape[0], 0, self.num_v_channels, dtype=x.dtype, device=x.device)
        return k_cache, v_cache

    def forward(self, *args, kv_cache: Optional[KVCache] = None, **kwargs):
        attn_output = self[0](*args, kv_cache=kv_cache, **kwargs)
        mlp_output = self[1](attn_output.last_hidden_state)
        return ModuleOutput(last_hidden_state=mlp_output.last_hidden_state, kv_cache=attn_output.kv_cache)



class SelfAttentionLayer(AbstractAttentionLayer):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        max_heads_parallel: Optional[int] = None,
        causal_attention: bool = False,
        widening_factor: int = 1,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            max_heads_parallel=max_heads_parallel,
            causal_attention=causal_attention,
            dropout=dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

        self.num_qk_channels = self_attn.attention.num_qk_channels
        self.num_v_channels = self_attn.attention.num_v_channels

        super().__init__(
            Residual(self_attn, residual_dropout),
            Residual(MLP(num_channels, widening_factor, bias=mlp_bias), residual_dropout),
        )



class SelfAttentionBlock(nn.Sequential):
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
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
    ):
        layers = [
            SelfAttentionLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                num_qk_channels=num_qk_channels,
                num_v_channels=num_v_channels,
                max_heads_parallel=max_heads_parallel,
                causal_attention=causal_attention,
                widening_factor=widening_factor,
                dropout=dropout,
                residual_dropout=residual_dropout,
                qkv_bias=qkv_bias,
                out_bias=out_bias,
                mlp_bias=mlp_bias,
            )
            for _ in range(num_layers)
        ]

        if activation_checkpointing:
            layers = [activation_checkpoint_wrapper(layer, offload_to_cpu=activation_offloading) for layer in layers]

        self.num_rotary_layers = num_rotary_layers
        super().__init__(*layers)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[List[KVCache]] = None,
    ):
        if kv_cache is None:
            kv_cache_updated = None
        else:
            if len(kv_cache) == 0:
                # initialize kv_cache for each self-attention layer
                kv_cache = [layer.empty_kv_cache(x) for layer in self]
            kv_cache_updated = []

        for i, layer in enumerate(self):
            rot_pos_emb_use = i < self.num_rotary_layers or self.num_rotary_layers == -1
            rot_pos_emb_i = rot_pos_emb if rot_pos_emb_use else None

            kv_cache_i = None if kv_cache is None else kv_cache[i]
            output = layer(x, pad_mask=pad_mask, rot_pos_emb=rot_pos_emb_i, kv_cache=kv_cache_i)

            x = output.last_hidden_state

            if kv_cache_updated is not None:
                kv_cache_updated.append(output.kv_cache)

        return ModuleOutput(last_hidden_state=x, kv_cache=kv_cache_updated)



class MLP(nn.Sequential):
    def __init__(self, num_channels: int, widening_factor: int, bias: bool = True):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels, bias=bias),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels, bias=bias),
        )

    def forward(self, x):
        return ModuleOutput(last_hidden_state=super().forward(x))



class MeanAggSelfAttentionBlock(nn.Module):
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
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
        init_scale: float = 0.02,
    ):
    
        super().__init__()

        self.self_attn_blk = SelfAttentionBlock(
            num_layers,
            num_heads,
            num_channels,
            num_qk_channels,
            num_v_channels,
            num_rotary_layers,
            max_heads_parallel,
            causal_attention,
            widening_factor,
            dropout,
            residual_dropout,
            activation_checkpointing,
            activation_offloading,
            qkv_bias,
            out_bias,
            mlp_bias,
        )

        self._init_parameters(init_scale)


    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        rot_pos_emb: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional[List[KVCache]] = None,
    ):

        out = self.self_attn_blk(x, pad_mask, rot_pos_emb, kv_cache)

        valid_positions = (~pad_mask).to(torch.int64) # (batch, num_res)
        latents_per_residue = out.last_hidden_state * rearrange(valid_positions, 'b s -> b s 1') # (batch, num_res, dim)
        residues_per_protein = valid_positions.sum(dim=-1) # (batch,)
        latents_per_protein = latents_per_residue.sum(dim=1) / rearrange(residues_per_protein, 'b -> b 1') # (batch, dim)

        # Mean aggregate, ignoring masked positions
        # latents_per_residue = out.last_hidden_state * \
                # rearrange(torch.where(pad_mask, torch.nan, 1.), 'b s -> b s 1') # (batch, num_res, dim)
        # latents_per_protein = latents_per_residue.nanmean(dim=1)
        
        return latents_per_protein


    def _init_parameters(self, init_scale: float):
        with torch.no_grad():
            init_parameters(self, init_scale)




def test():

    sab = SelfAttentionBlock(
        num_layers = 3,
        num_heads = 5,
        num_channels = 30, # number of dimensions of residue-levelembedding
        # num_qk_channels: Optional[int] = None,
        # num_v_channels: Optional[int] = None,
        # num_rotary_layers: int = 1,
        # max_heads_parallel: Optional[int] = None,
        # causal_attention: bool = False,
        # widening_factor: int = 1,
        # dropout: float = 0.0,
        # residual_dropout: float = 0.0,
        # activation_checkpointing: bool = False,
        # activation_offloading: bool = False,
        # qkv_bias: bool = True,
        # out_bias: bool = True,
        # mlp_bias: bool = True,
    )


    masab = MeanAggSelfAttentionBlock(
        num_layers = 3,
        num_heads = 5,
        num_channels = 30, # number of dimensions of residue-levelembedding
        # num_qk_channels: Optional[int] = None,
        # num_v_channels: Optional[int] = None,
        # num_rotary_layers: int = 1,
        # max_heads_parallel: Optional[int] = None,
        # causal_attention: bool = False,
        # widening_factor: int = 1,
        # dropout: float = 0.0,
        # residual_dropout: float = 0.0,
        # activation_checkpointing: bool = False,
        # activation_offloading: bool = False,
        # qkv_bias: bool = True,
        # out_bias: bool = True,
        # mlp_bias: bool = True,
    )
    masab.self_attn_blk.load_state_dict(sab.state_dict())

    total_params = sum(p.numel() for p in sab.parameters() if p.requires_grad)
    print(f"{total_params=}")

    total_params = sum(p.numel() for p in masab.parameters() if p.requires_grad)
    print(f"{total_params=}")

    # %%

    embed_dim = 30
    embeddings = [
        torch.rand(41, embed_dim),
        torch.rand(74, embed_dim),
        torch.rand(85, embed_dim),
        torch.rand(99, embed_dim),
        torch.rand(55, embed_dim),
        torch.rand(71, embed_dim),
        torch.rand(65, embed_dim),
        torch.rand(65, embed_dim),
    ]


    # Apply padding
    lengths = [embedding.shape[0] for embedding in embeddings]
    max_seq_len = max(lengths)
    valid_positions = torch.tensor([[1] * length + [0] * (max_seq_len - length) for length in lengths])
    padding_mask = ~valid_positions.bool() # True where padding token
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)

    print(f"{embeddings.shape=}")
    print(f"{padding_mask.shape=}")
    # %%

    out = sab(
        x=embeddings,
        pad_mask=padding_mask,
        # rot_pos_emb: Optional[RotaryPositionEmbedding] = None,
        # kv_cache: Optional[List[KVCache]] = None,
    )

    # Outputs
    latents_per_residue = out.last_hidden_state * rearrange(valid_positions, 'b s -> b s 1')
    print(f"{latents_per_residue.shape=}")

    # Mean aggregate
    latents_per_protein = latents_per_residue.sum(dim=1) / rearrange(valid_positions.sum(dim=-1), 'b -> b 1')
    print(f"{latents_per_protein.shape=}")

    # %%

    out_masab = masab(
        x=embeddings,
        pad_mask=padding_mask,
        # rot_pos_emb: Optional[RotaryPositionEmbedding] = None,
        # kv_cache: Optional[List[KVCache]] = None,
    )

    # %%

    print(latents_per_protein)
    print(out_masab)
    assert torch.allclose(latents_per_protein, out_masab)

    # %%
