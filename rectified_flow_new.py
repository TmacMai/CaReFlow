"""Rectified flow primitives used by CaReFlow.

Direction convention (consistent with Eqs. (1)(2)(7) of the paper):
    * ``x_0`` is the SOURCE feature  (acoustic / visual, distribution p_{m1});
    * ``x_1`` is the TARGET feature  (language,           distribution p_{m2});
    * interpolation  x_t = t * x_1 + (1 - t) * x_0,  so that
        t = 0 -> source x_0,  t = 1 -> target x_1;
    * the regression target of the velocity field is ``x_1 - x_0``.

Inference integrates the learned ODE from t = 0 to t = 1 with Euler steps,
i.e. it starts from a source feature and ends at the language distribution.
"""

import torch
import torch.nn as nn


class RectifiedFlow:
    def __init__(self):
        # element-wise squared error; reduction is done manually in mse_loss
        # so that a per-sample margin eta can be applied (Eq. (7)/(8)).
        self.loss = nn.MSELoss(reduction="none")

    @staticmethod
    def euler(x_t, v, dt):
        """One Euler update  x_{t+dt} = x_t + v * dt  (Eq. (1))."""
        return x_t + v * dt

    @staticmethod
    def create_flow(x_1, t, x_0=None):
        """Build the linear interpolation x_t = t * x_1 + (1 - t) * x_0.

        Args:
            x_1: target features, shape (N, D).
            t:   time in [0, 1], shape (N,) or (N, 1).
            x_0: source features, shape (N, D). If None, sample a standard
                 Gaussian (used only in vanilla generative rectified flow,
                 not in CaReFlow where x_0 is always a real modality feature).
        Returns:
            (x_t, x_0).
        """
        if x_0 is None:
            x_0 = torch.randn_like(x_1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return t * x_1 + (1.0 - t) * x_0, x_0

    def mse_loss(self, v, x_1, x_0, eta=0.0):
        """Adaptive-relaxed rectified-flow objective (Eq. (7)).

        per_sample = relu( mean_d ||v - (x_1 - x_0)||^2 - eta )
        then average over the batch.

        ``eta`` is either a scalar or a per-sample tensor of shape (N,):
        eta = 0 for a same-sample pair, eta = eps + (y_i - y_j)^2 for a pair
        drawn from different samples (Eq. (8)); the backward flow (Eq. (11))
        calls this with the default eta = 0.

        Note: the paper writes the squared L2 *norm* (sum over D), whereas the
        code averages over D. The two differ only by a constant factor D, and
        all released hyper-parameters (eps, alpha_f, alpha_b) were tuned for
        this averaged version, so it is kept unchanged for reproducibility.
        """
        per_sample = self.loss(x_1 - x_0, v).mean(dim=-1)  # (N,)
        if torch.is_tensor(eta):
            # Guard against the (now removed) CFG doubling that silently left
            # eta with a mismatched batch size.
            if eta.shape[0] != per_sample.shape[0]:
                raise ValueError(
                    f"eta batch size {eta.shape[0]} != velocity batch size "
                    f"{per_sample.shape[0]}. Every interpolation sample needs "
                    f"one margin value."
                )
        return torch.relu(per_sample - eta).mean()
