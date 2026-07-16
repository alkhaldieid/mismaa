"""1D convolutional network over raw CWRU acceleration segments (Phase 4).

A raw-signal classifier as a foil to the hand-engineered feature model. The
network learns its own filters directly from the 2048-sample waveform:

* a wide first kernel (length 7) gives an early receptive field large enough to
  catch the ~1 ms impact transients that mark a bearing defect;
* stacked conv + batch-norm + max-pool blocks progressively halve/quarter the
  time axis while widening channels, building from local transients to segment-
  level structure;
* global average pooling makes the head length-agnostic and reduces overfitting
  versus a large flattened dense layer — useful given only 30 training files.

Everything sizeable (channel width, dropout, epochs, lr) is set from
``configs/vibration.yaml``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Cnn1D(nn.Module):
    """1D-CNN mapping ``(B, 1, segment_length)`` to ``n_classes`` logits.

    Args:
        n_classes: Number of output classes.
        base_channels: Channels in the first conv block; doubles each block.
        dropout: Dropout probability before the linear classifier head.
    """

    def __init__(
        self, *, n_classes: int = 4, base_channels: int = 16, dropout: float = 0.3
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = (base_channels * m for m in (1, 2, 4, 8))

        def block(cin: int, cout: int, kernel: int, pool: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool),
            )

        self.features = nn.Sequential(
            block(1, c1, kernel=7, pool=4),   # wide first kernel for impact transients
            block(c1, c2, kernel=5, pool=4),
            block(c2, c3, kernel=3, pool=4),
            block(c3, c4, kernel=3, pool=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pool -> length agnostic
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of ``(B, 1, L)`` segments."""
        z = self.features(x)
        z = self.pool(z)
        return self.head(z)


def build_cnn(cnn_cfg: dict[str, Any], *, n_classes: int) -> Cnn1D:
    """Construct a :class:`Cnn1D` from the ``cnn`` block of vibration.yaml."""
    return Cnn1D(
        n_classes=n_classes,
        base_channels=int(cnn_cfg["base_channels"]),
        dropout=float(cnn_cfg["dropout"]),
    )
