"""Per-residue MLP classifier on top of frozen language-model embeddings."""
import torch
import torch.nn as nn


class ResidueMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, in_dim] -> logits [N]."""
        return self.net(x).squeeze(-1)
