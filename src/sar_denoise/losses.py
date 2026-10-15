"""Noise-aware complex-Gaussian likelihood on the full 2x2 covariance.

Model. A single-look pixel k = [S_VV, S_VH]^T of the TARGET look is circular complex Gaussian,
    k ~ CN(0, Sigma),   Sigma = C_hat + diag(n_VV, n_VH),
where C_hat is the network's noise-free covariance estimate and n_* is the (known) thermal-noise power of
the target look, on the same scale as the data. Noise is modelled, never subtracted, and it is independent
between channels, so it only touches the diagonal.

Per-pixel negative log-likelihood (dropping the constant 2 log pi):
    L = log det Sigma + tr(Sigma^-1 C_obs),      C_obs = k k^H  (the rank-one single-look C2 of the target)
For 2 x 2 Hermitian matrices both terms are closed form, so no matrix inverse is taken:
    s11 = C11 + n_VV,   s22 = C22 + n_VH,   s12 = C12
    det = s11 s22 - |s12|^2 = C11 C22 (1 - |gamma|^2) + C11 n_VH + n_VV C22 + n_VV n_VH
    tr(Sigma^-1 C_obs) = (s22 c11 + s11 c22 - 2 Re(s12 conj(c12))) / det
with 1 - |gamma|^2 = sech^2(|z|) evaluated directly (no cancellation when |gamma| -> 1).

Why this preserves coherence: the C12 term enters the likelihood on equal footing with the powers, so the
loss is minimised only by the true complex C12 (sign and phase included), and the noise term stops the
network from lowering |gamma| to explain noise power that is already accounted for.

Noise2Noise: Look A is the input and Look B the target, and vice versa; the two looks share the scene
covariance but have independent speckle and independent noise.
"""
from __future__ import annotations

import torch

from .models import LOG_CLAMP, make_input, raw_to_covariance

DET_FLOOR = 1e-12
SECH_X_MAX = 20.0           # tanh/sech saturate long before this; avoids exp overflow in float32


def _sech2(m: torch.Tensor) -> torch.Tensor:
    x = m.clamp(max=SECH_X_MAX)
    return 4.0 / (torch.exp(x) + torch.exp(-x)) ** 2


def covariance_nll_map(raw: torch.Tensor, target_c2: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    """Per-pixel NLL (B,H,W). raw: (B,4,H,W) head output; target_c2: (B,4,H,W); target_noise: (B,2,H,W)."""
    a = raw[:, 0].clamp(-LOG_CLAMP, LOG_CLAMP)
    b = raw[:, 1].clamp(-LOG_CLAMP, LOG_CLAMP)
    u, v = raw[:, 2], raw[:, 3]
    c11, c22 = torch.exp(a), torch.exp(b)
    m = torch.sqrt(u * u + v * v + 1e-8)
    g = torch.tanh(m)
    s = torch.sqrt(c11 * c22)
    c12_re, c12_im = g * u / m * s, g * v / m * s

    n11, n22 = target_noise[:, 0], target_noise[:, 1]
    o11, o22, o_re, o_im = target_c2[:, 0], target_c2[:, 1], target_c2[:, 2], target_c2[:, 3]

    s11, s22 = c11 + n11, c22 + n22
    det = c11 * c22 * _sech2(m) + c11 * n22 + n11 * c22 + n11 * n22
    det = det.clamp_min(DET_FLOOR)
    cross = c12_re * o_re + c12_im * o_im                 # Re(s12 conj(c12_obs))
    return torch.log(det) + (s22 * o11 + s11 * o22 - 2.0 * cross) / det


def _sanitise(x: torch.Tensor, valid: torch.Tensor, fill: float) -> torch.Tensor:
    return torch.where(valid.unsqueeze(1), x, torch.full_like(x, fill))


def covariance_nll(raw, target_c2, target_noise, valid=None, reduction: str = "mean"):
    """Masked NLL. valid: (B,H,W) bool. Invalid pixels (NaN or not) contribute nothing, not even gradient."""
    if valid is None:
        valid = torch.ones(raw.shape[0], *raw.shape[2:], dtype=torch.bool, device=raw.device)
    raw = _sanitise(raw, valid, 0.0)
    target_c2 = _sanitise(target_c2, valid, 1.0)
    target_noise = _sanitise(target_noise, valid, 1.0)
    nll = covariance_nll_map(raw, target_c2, target_noise)
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    if reduction == "map":
        return nll
    total = nll.sum()
    if reduction == "sum":
        return total
    return total / valid.sum().clamp_min(1)


def n2n_loss(model, batch: dict, use_noise_input: bool = True):
    """Symmetric Noise2Noise loss. batch tensors: c2_a, c2_b (B,4,H,W), noise_a, noise_b (B,2,H,W), valid (B,H,W).

    Returns (loss, parts) with parts = {'a2b': ..., 'b2a': ...} detached, for logging.
    """
    valid = batch["valid"]
    in_a = make_input(batch["c2_a"], batch["noise_a"] if use_noise_input else None)
    in_b = make_input(batch["c2_b"], batch["noise_b"] if use_noise_input else None)
    a2b = covariance_nll(model(in_a), batch["c2_b"], batch["noise_b"], valid)
    b2a = covariance_nll(model(in_b), batch["c2_a"], batch["noise_a"], valid)
    return 0.5 * (a2b + b2a), {"a2b": a2b.detach(), "b2a": b2a.detach()}


@torch.no_grad()
def coherence_metrics(raw: torch.Tensor, ref_c2: torch.Tensor, valid: torch.Tensor | None = None) -> dict:
    """Coherence-preservation metrics of a prediction against a reference covariance map (B,4,H,W).

    The reference should be an ESTIMATE or a simulated truth, never a single-look C2 (that has |gamma| = 1).
    phase_err is angle(gamma_hat * conj(gamma_ref)), so values near +-pi do not look like 2 pi errors.
    """
    c11, c22, re, im, g_re, g_im, g_abs = raw_to_covariance(raw)
    r11, r22, rre, rim = ref_c2[:, 0], ref_c2[:, 1], ref_c2[:, 2], ref_c2[:, 3]
    r_abs = torch.sqrt(rre ** 2 + rim ** 2) / torch.sqrt(r11 * r22).clamp_min(1e-12)
    gr_re = rre / torch.sqrt(r11 * r22).clamp_min(1e-12)
    gr_im = rim / torch.sqrt(r11 * r22).clamp_min(1e-12)
    d_re = g_re * gr_re + g_im * gr_im                   # gamma_hat * conj(gamma_ref)
    d_im = g_im * gr_re - g_re * gr_im
    phase_err = torch.atan2(d_im, d_re)
    if valid is None:
        valid = torch.ones_like(g_abs, dtype=torch.bool)
    w = valid
    n = w.sum().clamp_min(1)
    return {
        "gamma_abs_mae": ((g_abs - r_abs).abs() * w).sum() / n,
        "gamma_complex_rmse": torch.sqrt((((g_re - gr_re) ** 2 + (g_im - gr_im) ** 2) * w).sum() / n),
        "phase_abs_err_rad": (phase_err.abs() * w).sum() / n,
        "log_c11_mae": ((torch.log(c11) - torch.log(r11.clamp_min(1e-12))).abs() * w).sum() / n,
        "log_c22_mae": ((torch.log(c22) - torch.log(r22.clamp_min(1e-12))).abs() * w).sum() / n,
    }