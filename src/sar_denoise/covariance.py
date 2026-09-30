"""Dual-pol (VV/VH) covariance helpers.

Channel order used everywhere in the project:
    [C11, C22, Re C12, Im C12]
with
    C11 = |S_vv|^2,   C22 = |S_vh|^2,   C12 = S_vv * conj(S_vh).

A single-look C2 is rank one (|coherence| == 1 exactly). Coherence and phase
only become meaningful for a covariance ESTIMATE (multilooked, or the output
of the network).
"""
from __future__ import annotations

import numpy as np

CHANNELS = ("C11", "C22", "ReC12", "ImC12")


def form_c2(s_vv: np.ndarray, s_vh: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Single-look C2 from complex VV and VH; returns array of shape (4, ...)."""
    s_vv = np.asarray(s_vv)
    s_vh = np.asarray(s_vh)
    if s_vv.shape != s_vh.shape:
        raise ValueError(f"shape mismatch: {s_vv.shape} vs {s_vh.shape}")
    if not (np.iscomplexobj(s_vv) and np.iscomplexobj(s_vh)):
        raise ValueError("form_c2 needs complex (SLC) input")

    vv_r, vv_i = s_vv.real, s_vv.imag
    vh_r, vh_i = s_vh.real, s_vh.imag
    out = np.empty((4,) + s_vv.shape, dtype=dtype)
    out[0] = vv_r * vv_r + vv_i * vv_i
    out[1] = vh_r * vh_r + vh_i * vh_i
    out[2] = vv_r * vh_r + vv_i * vh_i          # Re{S_vv S_vh*}
    out[3] = vv_i * vh_r - vv_r * vh_i          # Im{S_vv S_vh*}
    return out


def multilook(c2: np.ndarray, size: int | tuple[int, int] = 5) -> np.ndarray:
    """Boxcar average of a (4, H, W) C2 stack over the two spatial axes.

    size is (azimuth, range) window, or one int for a square window.
    """
    from scipy.ndimage import uniform_filter

    if isinstance(size, int):
        size = (size, size)
    return uniform_filter(c2, size=(1, size[0], size[1]), mode="reflect")


def coherence(c2: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return (|gamma|, phase) from a C2 ESTIMATE of shape (4, ...).

    |gamma| = |C12| / sqrt(C11 C22),   phase = arg(C12) = arctan2(Im, Re).
    """
    c11, c22, re, im = c2[0], c2[1], c2[2], c2[3]
    gamma = np.hypot(re, im) / np.sqrt(np.maximum(c11 * c22, eps))
    phase = np.arctan2(im, re)
    return gamma, phase


def psd_violation(c2: np.ndarray) -> np.ndarray:
    """|C12|^2 - C11*C22. Must be <= 0 for a valid (PSD) 2x2 covariance."""
    c11, c22, re, im = c2[0], c2[1], c2[2], c2[3]
    return re * re + im * im - c11 * c22