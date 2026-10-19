import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_ckpt(ckpt_path, device):
    model = MLP(in_dim=768, hidden_dim=512, n_classes=9, dropout=0.1)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    return model.to(device).eval()
