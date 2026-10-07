import torch
import torch.nn as nn

from models.tokenizer import PBSTokenizer
from models.blocks import TransformerEncoderBlock


class PBSEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = PBSTokenizer(
            patch_feat_dim=config.patch_feat_dim,
            d_model=config.d_model,
            type_vocab_size=config.type_vocab_size,
            type_dropout=config.type_dropout,
            use_type_embed=config.use_type_embed,
            use_patch_feat=config.use_patch_feat,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        dpr = torch.linspace(0, config.drop_path_rate, config.n_layers).tolist()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                drop_path=dpr[i],
                qkv_bias=config.qkv_bias,
            )
            for i in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        patch_features,
        type_ids,
        padding_mask=None,
    ):
        x = self.tokenizer(patch_features, type_ids)
        B = x.size(0)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_pad = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
        mask_with_cls = torch.cat([cls_pad, padding_mask], dim=1)

        for block in self.blocks:
            x = block(x, key_padding_mask=mask_with_cls)
        x = self.final_norm(x)

        return x
