import torch.nn as nn


class PBSTokenizer(nn.Module):
    def __init__(
        self,
        patch_feat_dim=768,
        d_model=512,
        type_vocab_size=10,
        type_dropout=0.1,
        use_type_embed=True,
        use_patch_feat=True,
    ):
        super().__init__()
        self.pad_token_id = type_vocab_size
        self.use_type_embed = use_type_embed
        self.use_patch_feat = use_patch_feat

        if use_patch_feat:
            self.patch_proj = nn.Linear(patch_feat_dim, d_model)
        else:
            self.patch_proj = None
        if use_type_embed:
            self.type_embed = nn.Embedding(
                num_embeddings=type_vocab_size + 1,
                embedding_dim=d_model,
                padding_idx=self.pad_token_id,
            )
            self.type_dropout = nn.Dropout(type_dropout)
        else:
            self.type_embed = None
            self.type_dropout = None

        self._init_weights()

    def _init_weights(self):
        if self.use_patch_feat:
            nn.init.xavier_uniform_(self.patch_proj.weight)
            nn.init.zeros_(self.patch_proj.bias)
        if self.use_type_embed:
            nn.init.normal_(self.type_embed.weight)
            self.type_embed.weight.data[self.pad_token_id].zero_()

    def forward(
        self,
        patch_features,
        type_ids,
    ):
        if self.use_type_embed:
            type_emb = self.type_embed(type_ids)
            type_emb = self.type_dropout(type_emb)
        if not self.use_patch_feat:
            return type_emb
        value_emb = self.patch_proj(patch_features)
        if not self.use_type_embed:
            return value_emb
        return value_emb + type_emb
