import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_heads,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(
        self,
        x,
        key_padding_mask=None,
    ):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits.masked_fill(
            key_padding_mask[:, None, None, :], float("-inf")
        )

        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        out = out.masked_fill(key_padding_mask[:, :, None], 0.0)

        return out


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0,
        drop_path=0.0,
        qkv_bias=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            embed_dim=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qkv_bias=qkv_bias,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(embed_dim=d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x,
        key_padding_mask=None,
    ):
        attn_out = self.attn(self.norm1(x), key_padding_mask)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
