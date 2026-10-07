import torch.nn as nn


class DownstreamHead(nn.Module):

    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, pooled):
        return self.linear(self.drop(self.norm(pooled)))
