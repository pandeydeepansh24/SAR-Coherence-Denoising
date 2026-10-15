"""Run a trained network over a whole region archive.

The network was trained on 64 x 512 patches, each normalised on its own (patches.normalise_pair), so a
region is processed the same way: overlapping patches with the training size, each normalised exactly like
in training, and the overlapping predictions are averaged. Edge rows/columns that the regular grid would
miss get one extra patch flush with the border.

The prediction comes from ONE look (default A). The reference coherence is computed from the OTHER look, so
the two are independent estimates of the same scene.
"""
from __future__ import annotations

import numpy as np
import torch

from .evaluate import REF_WINDOW, boxcar
from .models import make_input, raw_to_covariance
from .patches import PatchConfig, extract_patch


def _axis(n: int, size: int, stride: int) -> list[int]:
    if n < size:
        raise ValueError(f"region side {n} is smaller than the patch size {size}")
    o = list(range(0, n - size + 1, stride))
    if o[-1] != n - size:
        o.append(n - size)
    return o


def tile_origins(shape, cfg: PatchConfig) -> list[tuple[int, int]]:
    h, w = shape
    return [(r, c) for r in _axis(h, cfg.rows, cfg.stride_rows) for c in _axis(w, cfg.cols, cfg.stride_cols)]


@torch.no_grad()
def predict_region(model, arrays: dict, cfg: PatchConfig | None = None, device="cpu", use_noise_input=True,
                   look: str = "a", batch_size: int = 8) -> dict:
    """Returns maps of shape (H, W): c11, c22 (denormalised powers, same units as the archive), g_re, g_im, g_abs."""
    cfg = cfg or PatchConfig()
    h, w = arrays["valid"].shape
    acc = np.zeros((4, h, w))
    cnt = np.zeros((h, w))
    origins = tile_origins((h, w), cfg)
    if hasattr(model, "eval"):
        model.eval()
    for i in range(0, len(origins), batch_size):
        chunk = origins[i:i + batch_size]
        items = [extract_patch(arrays, r, c, cfg, normalise=True) for r, c in chunk]
        clean = lambda k: torch.from_numpy(np.stack([np.nan_to_num(it[k], nan=0.0, posinf=0.0, neginf=0.0) for it in items])).float()
        x = make_input(clean(f"c2_{look}"), clean(f"noise_{look}") if use_noise_input else None).to(device)
        c11, c22, _, _, g_re, g_im, _ = (t.cpu().numpy() for t in raw_to_covariance(model(x)))
        for j, ((r, c), it) in enumerate(zip(chunk, items)):
            s11, s22 = it["scales"]
            sl = (slice(r, r + cfg.rows), slice(c, c + cfg.cols))
            acc[0][sl] += c11[j] * s11
            acc[1][sl] += c22[j] * s22
            acc[2][sl] += g_re[j]
            acc[3][sl] += g_im[j]
            cnt[sl] += 1
    out = acc / np.maximum(cnt, 1)
    return {"c11": out[0], "c22": out[1], "g_re": out[2], "g_im": out[3], "g_abs": np.hypot(out[2], out[3]), "count": cnt}


def reference_gamma(arrays: dict, look: str = "b", window=REF_WINDOW) -> dict:
    """Boxcar coherence of one look: gamma_re, gamma_im, |gamma| maps (H, W). Noisy and noise-biased: a reference, not truth."""
    c2 = torch.from_numpy(np.nan_to_num(arrays[f"c2_{look}"], nan=0.0)).float()[None]
    ml = boxcar(c2, window)[0].numpy()
    s = np.sqrt(np.maximum(ml[0] * ml[1], 1e-12))
    re, im = ml[2] / s, ml[3] / s
    return {"g_re": re, "g_im": im, "g_abs": np.hypot(re, im)}