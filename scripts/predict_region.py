"""Run a trained checkpoint over a whole region archive and compare with the boxcar reference.

    python scripts/predict_region.py ~/data/sar/regions/r2_iw1_b4.npz --run ~/runs/r1 [--out r2_pred.png]

Six panels, same geometric aspect as inspect_region.py:
    reference |gamma|   9x65 boxcar of Look B                 network |gamma_hat|   from Look A
    reference phase     where |gamma| > 0.2                   network phase        where |gamma_hat| > 0.2
    predicted vs reference |gamma| (2-D histogram, log counts)   network VV sigma0 [dB] (despeckled)
The prediction uses only Look A and the reference only Look B, so they are independent. Also prints how much
of the region each calls coherent and how often the sign of C12 agrees.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sar_denoise.evaluate import load_model
from sar_denoise.patches import PatchConfig, load_region
from sar_denoise.predict import predict_region, reference_gamma
from sar_denoise.train import pick_device


def summary(pred: dict, ref: dict, valid: np.ndarray, thr: float = 0.3) -> dict:
    v = valid
    coh = v & (ref["g_abs"] > thr)
    d_re = pred["g_re"] * ref["g_re"] + pred["g_im"] * ref["g_im"]
    d_im = pred["g_im"] * ref["g_re"] - pred["g_re"] * ref["g_im"]
    perr = np.arctan2(d_im, d_re)
    return {
        "frac_ref_gt": float(np.mean(ref["g_abs"][v] > thr)), "frac_pred_gt": float(np.mean(pred["g_abs"][v] > thr)),
        "mean_ref": float(ref["g_abs"][v].mean()), "mean_pred": float(pred["g_abs"][v].mean()),
        "sign_agreement": float(np.mean(np.cos(perr[coh]) > 0)) if coh.any() else float("nan"),
        "phase_err_median": float(np.median(np.abs(perr[coh]))) if coh.any() else float("nan"),
        "ref_coherent_px": int(coh.sum()),
    }


def make_figure(pred: dict, ref: dict, valid: np.ndarray, meta: dict, name: str, fc: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    az = float(meta.get("azimuth_pixel_spacing_m", 27.9))
    rg = float(meta.get("range_pixel_spacing_m", 2.33))
    aspect = az / (rg * fc)

    def sub(a):                       # average fc columns so a 6000-wide region fits the screen
        n = (a.shape[1] // fc) * fc
        return a[:, :n].reshape(a.shape[0], n // fc, fc).mean(axis=2)

    def sub_phase(re, im, mag, lo):
        ph = np.arctan2(im, re)
        ph = np.where(valid & (mag > lo), ph, np.nan)
        return ph[:, : (ph.shape[1] // fc) * fc: fc]

    fig, ax = plt.subplots(3, 2, figsize=(15, 13))
    kw = dict(aspect=aspect, interpolation="nearest")
    for a, img, title, lo, hi, cm in (
        (ax[0, 0], sub(np.where(valid, ref["g_abs"], np.nan)), "reference |gamma| (Look B, 9x65)", 0, 0.8, "magma"),
        (ax[0, 1], sub(np.where(valid, pred["g_abs"], np.nan)), "network |gamma_hat| (from Look A)", 0, 0.8, "magma"),
    ):
        im = a.imshow(img, vmin=lo, vmax=hi, cmap=cm, **kw)
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=0.04, pad=0.02)
    cmap = plt.get_cmap("hsv").with_extremes(bad="#bbbbbb")
    for a, img, title in (
        (ax[1, 0], sub_phase(ref["g_re"], ref["g_im"], ref["g_abs"], 0.2), "reference phase of C12 [rad], |gamma| > 0.2"),
        (ax[1, 1], sub_phase(pred["g_re"], pred["g_im"], pred["g_abs"], 0.2), "network phase of C12 [rad], |gamma_hat| > 0.2"),
    ):
        im = a.imshow(img, vmin=-np.pi, vmax=np.pi, cmap=cmap, **kw)
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=0.04, pad=0.02)
    v = valid
    h = ax[2, 0].hist2d(ref["g_abs"][v], pred["g_abs"][v], bins=60, range=[[0, 1], [0, 1]], norm=matplotlib.colors.LogNorm(), cmap="viridis")
    ax[2, 0].plot([0, 1], [0, 1], "w--", lw=1)
    ax[2, 0].set_xlabel("reference |gamma|")
    ax[2, 0].set_ylabel("network |gamma_hat|")
    ax[2, 0].set_title("predicted vs reference (dashed = equal)")
    fig.colorbar(h[3], ax=ax[2, 0], fraction=0.04, pad=0.02)
    vv = 10 * np.log10(np.maximum(sub(np.where(valid, pred["c11"], np.nan)), 1e-12))
    im = ax[2, 1].imshow(vv, vmin=-25, vmax=0, cmap="gray", **kw)
    ax[2, 1].set_title("network VV sigma0 [dB]")
    fig.colorbar(im, ax=ax[2, 1], fraction=0.04, pad=0.02)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[2, 1]):
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(name)
    fig.tight_layout()
    return fig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patch-rows", type=int, default=64)
    ap.add_argument("--patch-cols", type=int, default=512)
    a = ap.parse_args(argv)

    run = Path(a.run).expanduser()
    ckpt = Path(a.ckpt).expanduser() if a.ckpt.endswith(".pt") else run / f"{a.ckpt}.pt"
    dev = pick_device(a.device)
    model, meta, _ = load_model(ckpt, dev)
    path = Path(a.archive).expanduser()
    arrays, rmeta = load_region(path)
    cfg = PatchConfig(rows=a.patch_rows, cols=a.patch_cols)
    pred = predict_region(model, arrays, cfg, dev, meta.get("use_noise_input", True), look="a")
    ref = reference_gamma(arrays, look="b")
    valid = arrays["valid"].astype(bool) & (pred["count"] > 0)
    s = summary(pred, ref, valid)
    print(f"{path.name}: |gamma|>0.3 in {100 * s['frac_ref_gt']:.1f}% of pixels (reference) vs {100 * s['frac_pred_gt']:.1f}% (network)")
    print(f"mean |gamma|: reference {s['mean_ref']:.3f}, network {s['mean_pred']:.3f}")
    print(f"where the reference is coherent ({s['ref_coherent_px']} px): sign agreement {s['sign_agreement']:.3f}, median phase error {s['phase_err_median']:.3f} rad")
    out = Path(a.out).expanduser() if a.out else path.with_name(path.stem + "_pred.png")
    fig = make_figure(pred, ref, valid, rmeta, f"{path.stem}: {ckpt.name} of {run.name}")
    fig.savefig(out, dpi=90)
    print(f"saved {out}")


if __name__ == "__main__":
    main()