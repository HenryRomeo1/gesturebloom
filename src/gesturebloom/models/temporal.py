"""Temporal gesture spotter: a causal dilated TCN over landmark windows.

This is the part of the project that is actually a machine learning problem
rather than a geometry problem, and it is worth being precise about *which*
problem it is.

A static classifier can tell you the hand is currently in a pinch. It cannot
tell you the user just *performed a pinch gesture*, because that is an event with
an onset, and onset is a property of a trajectory, not a frame. Detecting events
in an unsegmented stream is **gesture spotting**, and the hard parts are:

- **No segmentation at inference time.** Frames arrive one at a time and most of
  them belong to no gesture at all. The negative class dominates heavily.
- **Latency is part of the objective.** A detector that fires 400 ms after the
  gesture is useless for interaction no matter how accurate it is.
- **Label ambiguity at boundaries.** Where exactly does a swipe begin? Human
  annotators disagree by several frames, so the loss should not be punished for
  boundary disagreement.

**Why a dilated TCN and not an LSTM or CTC.** Three reasons, all practical.
Dilated convolutions give an exponentially growing receptive field with a fixed
per-frame cost, so a 4-block stack with dilations 1/2/4/8 sees ~31 frames of
history at ~0.5 ms of compute. Causal padding means the model never peeks
forward, so offline accuracy equals online accuracy -- with a bidirectional model
your eval numbers are a lie. And frame-wise labels plus a hysteresis state
machine is dramatically easier to debug than CTC alignment: when it misfires you
can look at the probability trace and see exactly why.

**Soft boundary labels.** Rather than a hard 0/1 frame label, the target ramps
up over a few frames around the annotated onset. This directly addresses
annotator disagreement and empirically reduces jitter at the decision boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for gesturebloom.models. Install with: pip install 'gesturebloom[train]'"
    ) from exc


@dataclass
class TCNConfig:
    input_dim: int = 60
    channels: int = 64
    n_blocks: int = 4
    kernel_size: int = 3
    dropout: float = 0.1
    n_classes: int = 5
    """Number of gesture classes *including* index 0 = "no gesture"."""

    @property
    def receptive_field(self) -> int:
        """Frames of history visible to the last output. Report this in the README --
        it is the model's memory, and it should comfortably exceed your longest
        gesture."""
        rf = 1
        for i in range(self.n_blocks):
            rf += 2 * (self.kernel_size - 1) * (2**i)
        return rf


class CausalConv1d(nn.Module):
    """Dilated 1D convolution padded only on the left.

    Left-only padding is the whole point: output ``t`` depends on inputs
    ``<= t``, so the model can run frame-by-frame at inference with the same
    weights and produce the same numbers as an offline pass.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(torch_functional.pad(x, (self.pad, 0)))


class ResidualBlock(nn.Module):
    """Two causal convolutions, weight-normalized, with a residual connection."""

    def __init__(self, ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(ch, ch, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, ch)
        self.norm2 = nn.GroupNorm(1, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(torch_functional.gelu(self.norm1(self.conv1(x))))
        h = self.drop(torch_functional.gelu(self.norm2(self.conv2(h))))
        return x + h


class GestureTCN(nn.Module):
    """Frame-wise gesture logits from a landmark feature sequence.

    Input ``(B, T, input_dim)``, output ``(B, T, n_classes)`` -- one prediction
    per frame, which is what makes onset detection possible.
    """

    def __init__(self, config: TCNConfig | None = None) -> None:
        super().__init__()
        self.config = config or TCNConfig()
        c = self.config
        self.stem = nn.Conv1d(c.input_dim, c.channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(c.channels, c.kernel_size, 2**i, c.dropout) for i in range(c.n_blocks)]
        )
        self.head = nn.Conv1d(c.channels, c.n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)  # (B, C, T) for conv1d
        h = self.stem(h)
        for block in self.blocks:
            h = block(h)
        return self.head(h).transpose(1, 2)  # back to (B, T, n_classes)

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Frame-wise cross-entropy against soft targets.

        Parameters
        ----------
        logits:
            ``(B, T, n_classes)``.
        targets:
            ``(B, T, n_classes)`` soft target distribution from
            :func:`soft_onset_targets`.
        class_weights:
            ``(n_classes,)``. Strongly recommended -- the "no gesture" class is
            typically 90%+ of frames, and unweighted training yields a model that
            predicts "nothing" forever with excellent accuracy.
        """
        logp = torch_functional.log_softmax(logits, dim=-1)
        per_frame = -(targets * logp).sum(dim=-1)
        if class_weights is not None:
            w = (targets * class_weights.view(1, 1, -1)).sum(dim=-1)
            return (per_frame * w).sum() / w.sum().clamp_min(1e-6)
        return per_frame.mean()

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def soft_onset_targets(
    labels: np.ndarray,
    n_classes: int,
    ramp: int = 3,
) -> np.ndarray:
    """Convert hard frame labels into soft targets with ramped boundaries.

    Annotators disagree about onset frames by a few frames. Hard labels force the
    model to commit to a boundary it cannot see, producing high-variance
    predictions right where you need stability. Ramping the target over
    ``2 * ramp + 1`` frames around each transition makes the loss tolerant of
    exactly the ambiguity the data actually contains.

    Parameters
    ----------
    labels:
        ``(T,)`` integer class per frame, ``0`` = no gesture.
    n_classes:
        Total classes including background.
    ramp:
        Half-width of the transition ramp, in frames.

    Returns
    -------
    np.ndarray
        ``(T, n_classes)`` float32 rows summing to 1.

    Examples
    --------
    >>> import numpy as np
    >>> lab = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    >>> t = soft_onset_targets(lab, n_classes=2, ramp=1)
    >>> t.shape
    (8, 2)
    >>> bool(np.allclose(t.sum(axis=1), 1.0))
    True
    >>> bool(0.0 < t[2, 1] < 1.0)
    True
    """
    labels = np.asarray(labels, dtype=np.int64)
    T = labels.shape[0]
    onehot = np.zeros((T, n_classes), dtype=np.float32)
    onehot[np.arange(T), labels] = 1.0
    if ramp <= 0:
        return onehot

    kernel = np.ones(2 * ramp + 1, dtype=np.float32)
    kernel /= kernel.sum()
    smoothed = np.empty_like(onehot)
    for k in range(n_classes):
        padded = np.pad(onehot[:, k], (ramp, ramp), mode="edge")
        smoothed[:, k] = np.convolve(padded, kernel, mode="valid")
    smoothed /= np.maximum(smoothed.sum(axis=1, keepdims=True), 1e-8)
    return smoothed.astype(np.float32)


def class_weights_from_labels(labels: np.ndarray, n_classes: int, power: float = 0.5) -> np.ndarray:
    """Inverse-frequency class weights, softened by ``power``.

    Full inverse frequency (``power=1``) over-corrects badly when background is
    95% of frames -- the model starts firing constantly. The square root is a
    reliable middle ground.

    Examples
    --------
    >>> import numpy as np
    >>> w = class_weights_from_labels(np.array([0]*90 + [1]*10), n_classes=2)
    >>> bool(w[1] > w[0])
    True
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = (counts.sum() / counts) ** power
    return (w / w.mean()).astype(np.float32)
