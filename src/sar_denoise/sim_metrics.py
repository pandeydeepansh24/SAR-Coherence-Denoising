"""Metrics for the simulation experiment (numpy, single images). Estimates and truth are (4, H, W) C2 stacks
[C11, C22, Re C12, Im C12] in the same (physical) units.

  enl            mu^2 / var of VV intensity over a mask (homogeneous, texture-free field): 1 for single-look speckle
  psnr_db        on VV intensity in dB: 10 log10(R^2 / MSE), R = max - min of the TRUTH in dB
  complex_ssim   on the complex coherence gamma, local Gaussian window w:
                   (2 |sum w x y*| + K) / (sum w |x|^2 + sum w |y|^2 + K), averaged over the image
                 (sensitive to |gamma| and to phase VARIATIONS in the window; NOT to a constant phase offset,
                 since |sum x y*| removes a common rotation, so the phase error is reported separately)
  gamma_bins     |gamma| bias and RMSE binned by the true |gamma|
  phase_error    median |angle(gamma_hat gamma*)| where the true |gamma| > 0.3
  point_metrics  per point scatterer: VV intensity at the point vs truth, and the number of "bright" pixels
                 (> 3 x mean truth background) in a ring 2..8 px around it (single-look speckle: P(I > 3 mu) = e^-3 = 5%)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

DB_FLOOR = 1e-6
SSIM_K = 0.01
SSIM_SIGMA = 2.0
GAMMA_EDGES = np.round(np.arange(0.0, 1.0001, 0.1), 2)


def gamma_of(c2: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    return (c2[2] + 1j * c2[3]) / np.sqrt(np.maximum(c2[0] * c2[1], eps))


def db(x):
    return 10 * np.log10(np.maximum(x, DB_FLOOR))


def enl(intensity: np.ndarray, mask: np.ndarray) -> float:
    v = intensity[mask]
    return float(v.mean() ** 2 / v.var()) if v.size > 1 and v.var() > 0 else float("inf")


def psnr_db(est_int: np.ndarray, true_int: np.ndarray, mask: np.ndarray | None = None) -> float:
    t, e = db(true_int), db(est_int)
    if mask is not None:
        t, e = t[mask], e[mask]
    rng = t.max() - t.min()
    return float(10 * np.log10(rng ** 2 / np.mean((e - t) ** 2)))


def complex_ssim_map(x: np.ndarray, y: np.ndarray, sigma: float = SSIM_SIGMA, k: float = SSIM_K) -> np.ndarray:
    f = lambda a: gaussian_filter(a, sigma, mode="reflect")
    xy = x * np.conj(y)
    num = 2 * np.abs(f(xy.real) + 1j * f(xy.imag)) + k
    den = f(np.abs(x) ** 2) + f(np.abs(y) ** 2) + k
    return num / den


def complex_ssim(x, y, mask=None, **kw) -> float:
    m = complex_ssim_map(x, y, **kw)
    return float(m[mask].mean() if mask is not None else m.mean())


def gamma_bins(g_est: np.ndarray, g_true: np.ndarray, edges=GAMMA_EDGES) -> list[dict]:
    """Per true-|gamma| bin: n, mean true, mean est, bias (est - true), RMSE of |gamma|. Inputs flattened arrays."""
    a_t, a_e = np.abs(g_true).ravel(), np.abs(g_est).ravel()
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (a_t >= lo) & (a_t < hi)
        if not sel.any():
            continue
        d = a_e[sel] - a_t[sel]
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()), "true": float(a_t[sel].mean()),
                     "est": float(a_e[sel].mean()), "bias": float(d.mean()), "rmse": float(np.sqrt(np.mean(d ** 2)))})
    return rows


def phase_error_deg(g_est, g_true, min_true: float = 0.3) -> float:
    sel = np.abs(g_true) > min_true
    return float(np.degrees(np.median(np.abs(np.angle(g_est[sel] * np.conj(g_true[sel])))))) if sel.any() else float("nan")


def ring_mask(shape, r, c, r_in: float = 2, r_out: float = 8) -> np.ndarray:
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    d = np.hypot(rr - r, cc - c)
    return (d >= r_in) & (d <= r_out)


def point_metrics(est_int: np.ndarray, true_int: np.ndarray, points, r_in=2, r_out=8, factor=3.0) -> list[dict]:
    rows = []
    for r, c in points:
        ring = ring_mask(true_int.shape, r, c, r_in, r_out)
        bg = true_int[ring].mean()
        rows.append({"row": r, "col": c, "peak_db": float(db(est_int[r, c])), "true_peak_db": float(db(true_int[r, c])),
                     "bg_db": float(db(bg)), "bright_px": int((est_int[ring] > factor * bg).sum()),
                     "ring_px": int(ring.sum())})
    return rows
