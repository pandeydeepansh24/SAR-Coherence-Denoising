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