"""Look at one region archive made by make_patches.py: is it what the quick-look said it is?

Six panels (geometric aspect ratio, so the picture is not squashed):
    sigma0 VV [dB]            mean of the two looks
    sigma0 VH [dB]
    local |coherence|         9 x 65 boxcar on the mean of both looks (no noise correction)
    VH power / noise [dB]     0 dB = pure noise
    histogram of |coherence|  with the null floor (uncorrelated speckle) and the 0.3 threshold
    phase of C12 [rad]        only where |coherence| > 0.2, grey elsewhere

Usage:
    python scripts/inspect_region.py ~/data/sar/regions/r1_iw1_b2.npz [--out r1.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sar_denoise.covariance import coherence, multilook
from sar_denoise.patches import load_region


def block_mean(img: np.ndarray, fr: int, fc: int) -> np.ndarray:
    h, w = (img.shape[0] // fr) * fr, (img.shape[1] // fc) * fc
    return img[:h, :w].reshape(h // fr, fr, w // fc, fc).mean(axis=(1, 3))


def db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-12))


def make_figure(arrays: dict, meta: dict, max_cols: int = 1400, phase_min_gamma: float = 0.2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = arrays["valid"]
    c2 = 0.5 * (arrays["c2_a"] + arrays["c2_b"])
    rows, cols = valid.shape
    fc = max(1, int(np.ceil(cols / max_cols)))
    az = float(meta.get("azimuth_pixel_spacing_m", 27.9))
    rg = float(meta.get("range_pixel_spacing_m", 2.33))
    aspect = az / (rg * fc)

    ml = multilook(c2, (9, 65))
    gamma, phase = coherence(ml)
    def show(a: np.ndarray) -> np.ndarray:
        """Average fc neighbouring columns; invalid pixels become NaN first."""
        return block_mean(np.where(valid, a, np.nan), 1, fc)

    vv, vh = db(show(c2[0])), db(show(c2[1]))
    pnr = show(arrays["pnr_db"][1])
    g_img = show(gamma)
    ph = np.where(gamma > phase_min_gamma, phase, np.nan)[:, : (cols // fc) * fc: fc]

    fig, ax = plt.subplots(3, 2, figsize=(15, 13))
    spec = [(vv, "sigma0 VV [dB]", -25, 0, "gray"), (vh, "sigma0 VH [dB]", -35, -10, "gray"),
            (g_img, "local |coherence| (9 x 65)", 0, 0.6, "magma"), (pnr, "VH power / noise [dB]", -3, 18, "viridis")]
    for a, (img, title, lo, hi, cmap) in zip(ax.flat[:4], spec):
        im = a.imshow(img, vmin=lo, vmax=hi, cmap=cmap, aspect=aspect, interpolation="nearest")
        a.set_title(title)
        a.set_xticks([])
        a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.04, pad=0.02)

    g = gamma[valid]
    ax[2, 0].hist(g, bins=80, range=(0, 1), color="#444")
    ax[2, 0].axvline(0.3, color="r", lw=1)
    ax[2, 0].axvline(0.064, color="c", lw=1)
    ax[2, 0].set_yscale("log")
    ax[2, 0].set_title(f"|coherence| over valid pixels (median {np.median(g):.2f}, >0.3: {np.mean(g > 0.3):.3f})\n"
                       "cyan = null floor of uncorrelated speckle (~0.06), red = 0.3")

    cmap = plt.get_cmap("hsv").copy()
    cmap.set_bad("#bbbbbb")
    im = ax[2, 1].imshow(ph, vmin=-np.pi, vmax=np.pi, cmap=cmap, aspect=aspect, interpolation="nearest")
    ax[2, 1].set_title(f"phase of C12 [rad], |coherence| > {phase_min_gamma}")
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    fig.colorbar(im, ax=ax[2, 1], fraction=0.04, pad=0.02)

    reg = meta.get("region", {})
    fig.suptitle(f"{reg.get('name', '?')} ({reg.get('split', '?')}, {reg.get('kind', '?')}): "
                 f"{rows} x {cols} looks, lines {meta.get('region_lines_used')}, samples {meta.get('region_samples_used')}")
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot one region archive")
    ap.add_argument("archive")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-cols", type=int, default=1400)
    args = ap.parse_args()

    path = Path(args.archive).expanduser()
    arrays, meta = load_region(path)
    out = Path(args.out).expanduser() if args.out else path.with_suffix(".png")
    fig = make_figure(arrays, meta, args.max_cols)
    fig.savefig(out, dpi=90)
    print(f"saved {out}")


if __name__ == "__main__":
    main()