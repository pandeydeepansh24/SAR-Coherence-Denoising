"""Synthetic dual-pol SLC data with a KNOWN covariance, for testing.

For fully developed speckle the vector k = [S_vv, S_vh]^T is circular complex
Gaussian, k ~ CN(0, C) with C = [[C11, C12], [C12*, C22]].

Construction (z1, z2 independent CN(0, 1)):
    S_vv = sqrt(C11) * z1
    S_vh = conj(C12)/sqrt(C11) * z1 + sqrt(C22 - |C12|^2 / C11) * z2
which gives E|S_vv|^2 = C11, E|S_vh|^2 = C22, E[S_vv S_vh*] = C12.
"""
from __future__ import annotations

import numpy as np


def _cn(rng: np.random.Generator, shape) -> np.ndarray:
    """Standard circular complex Gaussian, E|z|^2 = 1."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def simulate_dualpol_slc(c11, c22, c12, shape=None, rng=None):
    """Return (s_vv, s_vh) complex128 arrays with the given covariance.

    c11, c22: real, >= 0.  c12: complex with |c12|^2 <= c11*c22.
    Each may be a scalar or an array (broadcast to `shape`), so a spatially
    varying covariance map is allowed.
    """
    rng = np.random.default_rng() if rng is None else rng
    c11 = np.asarray(c11, dtype=float)
    c22 = np.asarray(c22, dtype=float)
    c12 = np.asarray(c12, dtype=complex)
    if shape is None:
        shape = np.broadcast(c11, c22, c12).shape
    c11, c22, c12 = (np.broadcast_to(a, shape) for a in (c11, c22, c12))

    resid = c22 - (np.abs(c12) ** 2) / c11
    if np.any(resid < -1e-12 * np.maximum(c22, 1e-30)):
        raise ValueError("covariance not positive semi-definite: |C12|^2 > C11*C22")
    resid = np.maximum(resid, 0.0)

    z1 = _cn(rng, shape)
    z2 = _cn(rng, shape)
    s_vv = np.sqrt(c11) * z1
    s_vh = np.conj(c12) / np.sqrt(c11) * z1 + np.sqrt(resid) * z2
    return s_vv, s_vh


def simulate_look_pair(c11, c22, c12, n11=0.0, n22=0.0, shape=None, rng=None) -> dict:
    """Two independent single-look observations of the SAME noise-free covariance, plus their noise maps.

    Each look is k = signal + thermal noise, with signal ~ CN(0, C) and noise ~ CN(0, diag(n11, n22)), so the
    observed covariance is Sigma = C + diag(n). Returns float32 arrays in the layout the training code uses:
        c2_a, c2_b       (4, H, W)  [C11, C22, Re C12, Im C12] of the rank-one single-look C2
        noise_a, noise_b (2, H, W)  [n_VV, n_VH]
    """
    rng = np.random.default_rng() if rng is None else rng
    n11 = np.asarray(n11, dtype=float)
    n22 = np.asarray(n22, dtype=float)
    out = {}
    for tag in ("a", "b"):
        s_vv, s_vh = simulate_dualpol_slc(c11, c22, c12, shape=shape, rng=rng)
        shp = s_vv.shape
        s_vv = s_vv + np.sqrt(np.broadcast_to(n11, shp)) * _cn(rng, shp)
        s_vh = s_vh + np.sqrt(np.broadcast_to(n22, shp)) * _cn(rng, shp)
        c12_obs = s_vv * np.conj(s_vh)
        out[f"c2_{tag}"] = np.stack([np.abs(s_vv) ** 2, np.abs(s_vh) ** 2, c12_obs.real, c12_obs.imag]).astype(np.float32)
        out[f"noise_{tag}"] = np.stack([np.broadcast_to(n11, shp), np.broadcast_to(n22, shp)]).astype(np.float32)
    return out