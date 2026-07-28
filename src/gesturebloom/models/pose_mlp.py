"""Static pose classifier: canonical landmark vector -> pose class.

Small on purpose. After canonicalization the problem is nearly linearly
separable for distinct poses, so a 3-layer MLP with ~40k parameters reaches
high accuracy on a few thousand hand-recorded frames. Reaching for a large model
here would be the wrong instinct -- the win came from the geometry, not capacity,
and a small model is what lets the whole pipeline hit its latency budget.

Report per-class accuracy and a confusion matrix, not just top-1. Confusable
pairs (fist vs. curled, point vs. gun) are where the interesting failures live
and where you learn which poses to redesign.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

try:
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for gesturebloom.models. Install with: pip install 'gesturebloom[train]'"
    ) from exc


@dataclass
class PoseMLPConfig:
    input_dim: int = 60  # 20 landmarks x 3 (wrist dropped -- always zero)
    hidden: tuple[int, ...] = (128, 128)
    n_classes: int = 8
    dropout: float = 0.15
    label_smoothing: float = 0.05
    """Small label smoothing matters here: hand-recorded data has genuinely
    ambiguous transition frames, and a confidently wrong model produces visible
    flicker in the render."""


class PoseMLP(nn.Module):
    """MLP over a canonical landmark vector.

    BatchNorm before the nonlinearity, dropout after. BatchNorm is doing real
    work despite the input already being scale-normalized, because individual
    coordinate channels still have very different variances (the y channel is
    dominated by finger extension, z by palm curvature).
    """

    def __init__(self, config: PoseMLPConfig | None = None) -> None:
        super().__init__()
        self.config = config or PoseMLPConfig()
        dims = [self.config.input_dim, *self.config.hidden]
        layers: list[nn.Module] = []
        for a, b in pairwise(dims):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.GELU(), nn.Dropout(self.config.dropout)]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], self.config.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, input_dim)`` -> ``(B, n_classes)`` logits."""
        return self.head(self.trunk(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(class_index, probability)`` for a batch."""
        self.eval()
        probs = torch_functional.softmax(self.forward(x), dim=-1)
        conf, idx = probs.max(dim=-1)
        return idx, conf

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch_functional.cross_entropy(logits, targets, label_smoothing=self.config.label_smoothing)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
