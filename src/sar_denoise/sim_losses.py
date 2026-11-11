"""Supervised loss for the simulation experiment: KL divergence between CN(0, C_true) and CN(0, C_hat).

Per pixel, for 2x2 Hermitian positive-definite matrices,
    KL = tr(C_hat^-1 C) - log det(C_hat^-1 C) - 2  >= 0,   = 0 iff C_hat = C.
In terms of the head output (a, b, z = u + iv), with gamma_hat = tanh|z| z/|z|:
    det C_hat = e^{a+b} (1 - |gamma_hat|^2) = e^{a+b} / cosh^2|z|
    tr(C_hat^-1 C) = cosh^2|z| * [ C11/C11_hat + C22/C22_hat - 2 Re(gamma_hat conj(C12)) / sqrt(C11_hat C22_hat) ]
    log det C = log C11 + log C22 + log(1 - |gamma|^2)
so the loss is closed form, no matrix inverse, and log cosh is evaluated stably.

Why this loss for coherence:
  * C12 enters on equal footing with the powers: the minimum is only reached with the right complex C12.
  * It is invariant under C -> D C D for any positive diagonal D (both matrices scaled alike), so per-pixel it does
    not care about absolute brightness: bright point scatterers do not dominate the batch, dark fields are not ignored,
    and the per-patch normalisation does not change it.
  * It grows like 1 / (1 - |gamma|^2) for errors on highly coherent pixels (that is where errors cost information).
"""
from __future__ import annotations

import math

import torch

from .models import LOG_CLAMP

Z_MAX = 20.0
GAMMA_TRUE_MAX = 0.999


def _log_cosh(m: torch.Tensor) -> torch.Tensor:
    return m + torch.nn.functional.softplus(-2.0 * m) - math.log(2.0)


def kl_map(raw: torch.Tensor, c_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-pixel KL(CN(0, C_true) || CN(0, C_hat)), (B,H,W). raw: (B,4,H,W) head; c_true: (B,4,H,W) [C11,C22,ReC12,ImC12]."""
    a = raw[:, 0].clamp(-LOG_CLAMP, LOG_CLAMP)
    b = raw[:, 1].clamp(-LOG_CLAMP, LOG_CLAMP)
    u, v = raw[:, 2], raw[:, 3]
    m = torch.sqrt(u * u + v * v + 1e-8).clamp(max=Z_MAX)
    t = torch.tanh(m)
    g_re, g_im = t * u / m, t * v / m

    c11, c22 = c_true[:, 0].clamp_min(eps), c_true[:, 1].clamp_min(eps)
    re, im = c_true[:, 2], c_true[:, 3]
    lc = _log_cosh(m)
    cross = (g_re * re + g_im * im) * torch.exp(-0.5 * (a + b))           # Re(gamma_hat conj(C12)) / sqrt(C11_hat C22_hat)
    tr = torch.exp(2.0 * lc) * (c11 * torch.exp(-a) + c22 * torch.exp(-b) - 2.0 * cross)
    g2 = ((re * re + im * im) / (c11 * c22)).clamp(max=GAMMA_TRUE_MAX ** 2)
    logdet_true = torch.log(c11) + torch.log(c22) + torch.log1p(-g2)
    logdet_hat = a + b - 2.0 * lc
    return tr - logdet_true + logdet_hat - 2.0


def kl_loss(raw: torch.Tensor, c_true: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    k = kl_map(raw, c_true)
    if valid is None:
        return k.mean()
    return (k * valid).sum() / valid.sum().clamp_min(1)
