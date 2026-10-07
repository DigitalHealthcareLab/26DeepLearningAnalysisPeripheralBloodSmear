import torch
import torch.nn as nn


class PMAPool(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_heads,
        attn_dropout=0.0,
        dropout=0.1,
    ):
        super().__init__()

        self.seeds = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        padding_mask=None,
    ):
        B = x.size(0)
        queries = self.seeds.expand(B, -1, -1)

        out, attn_weights = self.attn(
            queries, x, x,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        out = self.drop(out)

        pooled = out[:, 0, :]
        attn_out = attn_weights[:, 0, :]

        attn_out = attn_out.masked_fill(padding_mask, 0.0)
        denom = attn_out.sum(dim=1, keepdim=True)
        attn_out = attn_out / denom

        return pooled, attn_out

