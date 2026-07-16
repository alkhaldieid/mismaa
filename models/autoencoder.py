"""Convolutional autoencoder for MIMII acoustic anomaly detection (Phase 2).

Design choices and why:

* **Convolutional, not dense.** The DCASE 2020 Task 2 baseline flattens a
  5-frame context window into a vector and feeds a fully-connected autoencoder.
  We keep each window as a 2-D ``(n_mels, n_frames)`` single-channel image so 2-D
  convolutions can exploit *spectral locality* — neighbouring mel bands and
  neighbouring frames are correlated, and weight-sharing across the mel axis is
  far more parameter-efficient than a dense layer over 320 inputs.
* **Downsample the mel axis only.** The time axis of a context window is tiny
  (5 frames), so all striding happens on the mel (frequency) axis; the frame
  axis is preserved throughout. Three stride-2 blocks reduce 64 mel bands to 8.
* **Linear output.** Targets are log-Mel values in decibels (unbounded, mostly
  negative), so the final layer is linear — no sigmoid/tanh squashing.
* **Reconstruction error is the anomaly score.** Trained only on normal audio,
  the AE reconstructs normal windows well and anomalous windows poorly; the MSE
  gap is the detection signal (see :mod:`models.evaluate`).
"""

from __future__ import annotations

import torch
from torch import nn

# Number of stride-2 (mel-axis) down/up-sampling blocks. Fixed at 3: 64 -> 8.
_N_BLOCKS = 3


class ConvAutoencoder(nn.Module):
    """Symmetric conv autoencoder over ``(1, n_mels, n_frames)`` context windows.

    Args:
        n_mels: Mel bands per window (must be divisible by ``2**3 = 8``).
        n_frames: Frames per context window (preserved through the network).
        bottleneck: Latent dimension of the dense code between encoder and decoder.
        base_channels: Channel width of the first conv block; doubles each block.
    """

    def __init__(
        self,
        *,
        n_mels: int,
        n_frames: int,
        bottleneck: int,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        if n_mels % (2**_N_BLOCKS) != 0:
            raise ValueError(
                f"n_mels ({n_mels}) must be divisible by {2**_N_BLOCKS} for "
                f"{_N_BLOCKS} stride-2 blocks"
            )
        self.n_mels = n_mels
        self.n_frames = n_frames

        chans = [1] + [base_channels * (2**i) for i in range(_N_BLOCKS)]
        enc_layers: list[nn.Module] = []
        for cin, cout in zip(chans[:-1], chans[1:], strict=True):
            # stride (2, 1): halve the mel axis, keep the frame axis.
            enc_layers += [
                nn.Conv2d(cin, cout, kernel_size=3, stride=(2, 1), padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
        self.encoder_conv = nn.Sequential(*enc_layers)

        # Discover the flattened encoder-output shape with a dummy pass so the
        # dense bottleneck adapts to n_mels/n_frames without hard-coded sizes.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, n_frames)
            enc_out = self.encoder_conv(dummy)
        self._enc_shape = tuple(enc_out.shape[1:])  # (C, n_mels/8, n_frames)
        flat = int(enc_out.numel())

        self.to_code = nn.Linear(flat, bottleneck)
        self.from_code = nn.Linear(bottleneck, flat)

        dec_layers: list[nn.Module] = []
        rev = chans[::-1]  # [4C, 2C, C, 1]
        for i, (cin, cout) in enumerate(zip(rev[:-1], rev[1:], strict=True)):
            last = i == _N_BLOCKS - 1
            # Mirror the encoder: double the mel axis (output_padding on height),
            # keep the frame axis. The last block emits the single-channel image
            # with no activation (linear reconstruction of dB values).
            dec_layers.append(
                nn.ConvTranspose2d(
                    cin,
                    cout,
                    kernel_size=3,
                    stride=(2, 1),
                    padding=1,
                    output_padding=(1, 0),
                )
            )
            if not last:
                dec_layers += [nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        self.decoder_conv = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to the bottleneck and decode back to the input shape."""
        z = self.encoder_conv(x)
        z = self.to_code(z.flatten(1))
        z = self.from_code(z).view(-1, *self._enc_shape)
        return self.decoder_conv(z)


def build_autoencoder(model_cfg: dict, *, n_mels: int, n_frames: int) -> ConvAutoencoder:
    """Construct a :class:`ConvAutoencoder` from the ``model`` block of train.yaml."""
    return ConvAutoencoder(
        n_mels=n_mels,
        n_frames=n_frames,
        bottleneck=int(model_cfg["bottleneck"]),
        base_channels=int(model_cfg["base_channels"]),
    )
