import torch.nn as nn

class PBSModel(nn.Module):
    def __init__(
        self,
        encoder,
        pool,
        head,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.head = head

    def forward(
        self,
        patch_features,
        type_ids,
        padding_mask=None,
    ):
        tokens = self.encoder(patch_features, type_ids, padding_mask)
        tokens = tokens[:, 1:, :]
        pooled, _ = self.pool(tokens, padding_mask=padding_mask)

        logits = self.head(pooled)
        return {"logits": logits, "features": pooled}

    def get_param_groups(
        self,
        lr,
        encoder_lr_scale,
        pma_lr_scale,
        weight_decay,
    ):
        return [
            {
                "params": list(self.encoder.parameters()),
                "lr": lr * encoder_lr_scale,
                "weight_decay": weight_decay,
                "name": "encoder",
            },
            {
                "params": list(self.pool.parameters()),
                "lr": lr * pma_lr_scale,
                "weight_decay": weight_decay,
                "name": "pma",
            },
            {
                "params": list(self.head.parameters()),
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]

    def state_dict_for_save(self):
        return {
            "head_state_dict": self.head.state_dict(),
            "pool_state_dict": self.pool.state_dict(),
            "encoder_state_dict": self.encoder.state_dict(),
        }

    def load_state_dict_for_resume(self, ckpt):
        self.head.load_state_dict(ckpt["head_state_dict"])
        self.pool.load_state_dict(ckpt["pool_state_dict"])
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])

