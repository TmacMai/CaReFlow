"""Velocity vector-field network V_{m1,m2} for CaReFlow (Eqs. (9)-(10)).

The network is a small MLP that takes the current feature concatenated with a
sinusoidal time embedding. It is intentionally lightweight (this is the only
trainable module CaReFlow adds on top of a standard multimodal backbone).
"""

import torch
import torch.nn as nn


class Prediction(nn.Module):
    def __init__(self, base_channels=16, time_emb_dim=None):
        super().__init__()
        # Time embedding width equals the feature width by default.
        self.time_emb_dim = time_emb_dim if time_emb_dim is not None else base_channels
        self.d_l = base_channels
        self.prediction_v = nn.Sequential(
            nn.Linear(self.d_l * 2, self.d_l),
            nn.ReLU(),
            nn.Linear(self.d_l, self.d_l),
            nn.ReLU(),
            nn.Linear(self.d_l, self.d_l),
            nn.ReLU(),
            nn.Linear(self.d_l, self.d_l),
        )

    @staticmethod
    def _sinusoidal_emb(values, dim):
        """Sinusoidal positional embedding (Eq. (9)): sin/cos at several freqs."""
        # Build the frequency table directly on the target device to avoid
        # silent CPU/GPU mismatches; values are scaled by 1000 (Eq. (9)).
        freqs = torch.pow(
            10000.0, torch.linspace(0.0, 1.0, dim // 2, device=values.device)
        )
        scaled = values * 1000.0
        sin_emb = torch.sin(scaled[:, None] / freqs)
        cos_emb = torch.cos(scaled[:, None] / freqs)
        return torch.cat([sin_emb, cos_emb], dim=-1)

    def time_emb(self, t, dim):
        return self._sinusoidal_emb(t, dim)

    def label_emb(self, y, dim):
        """Optional label embedding (reserved for classifier-free guidance).

        NOTE: CaReFlow's released models do NOT use guidance (y is always
        None at train/eval time). If you re-enable it for a regression task,
        do not use ``-1`` as the "unconditional" sentinel because -1 is a
        legitimate sentiment score on CMU-MOSI/MOSEI; pass an explicit mask
        instead.
        """
        return self._sinusoidal_emb(y, dim)

    def forward(self, x, t, y=None, label_mask=None):
        """
        Args:
            x: features (N, d_l).
            t: time, shape (N,) or (1,); a single shared t is broadcast to the
               whole batch (this is the inference case).
            y: optional labels (N,), see note above.
            label_mask: optional bool tensor (N,); True where the label
               embedding should be added. Replaces the unsafe ``y == -1`` rule.
        """
        temb = self.time_emb(t, x.shape[-1])
        if y is not None:
            if y.dim() == 1:
                yemb = self.label_emb(y, x.shape[-1])
                if label_mask is None:
                    # Backward-compatible behaviour (guidance is disabled in the
                    # released pipeline, so this path is never exercised).
                    label_mask = y != -1
                yemb = yemb * label_mask.to(yemb.dtype)[:, None]
                temb = temb + yemb
        # Broadcast a single time value to every sample in the batch.
        if temb.shape[0] == 1 and x.shape[0] != 1:
            temb = temb.repeat(x.shape[0], 1)
        return self.prediction_v(torch.cat([x, temb], dim=-1))
