"""One figure to show: single look vs standard multilook vs the network, for powers AND coherence.

    python scripts/make_showcase.py ~/data/sar/regions/r2_iw1_b4.npz --run ~/runs/r1 [--crop R0 R1 C0 C1] [--out showcase.png]

Columns (all from Look A except the last coherence column, which is the independent Look B):
    1  input: one look (A)            the speckled data the network sees
    2  standard: 5x33 boxcar          the usual multilook (blurs detail, but gives a coherence)
    3  network                        despeckled powers + coherence, one pass
Rows: sigma0 VV [dB], sigma0 VH [dB], |gamma|, phase of C12 where |gamma| > 0.2.
Row 3/4, column 1 is empty on purpose: one look has |gamma| = 1 at every pixel by construction (rank-one C2),
so there is no meaningful coherence "before". Column 4 of rows 3/4 is a 9x65 boxcar of Look B, an independent
reference for both estimates.
--crop selects looks (rows R0..R1, columns C0..C1 of the region archive); default is the whole region.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sar_denoise.evaluate import boxcar, load_model
from sar_denoise.patches import PatchConfig, load_region
from sar_denoise.predict import predict_region, reference_gamma
from sar_denoise.train import pick_device

BOX = (5, 33)


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-12))


def gamma_phase(c2: np.ndarray):
    s = np.sqrt(np.maximum(c2[0] * c2[1], 1e-12))
    re, im = c2[2] / s, c2[3] / s
    return np.hypot(re, im), np.arctan2(im, re)


def build_maps(arrays: dict, pred: dict, ref: dict) -> dict:
    """Everything the figure shows, as (H, W) arrays (powers in linear units)."""
    c2a = np.nan_to_num(arrays["c2_a"], nan=0.0)
    box = boxcar(torch.from_numpy(c2a).float()[None], BOX)[0].numpy()
    g_box, ph_box = gamma_phase(box)
    return {
        "in_vv": c2a[0], "in_vh": c2a[1], "box_vv": box[0], "box_vh": box[1], "net_vv": pred["c11"], "net_vh": pred["c22"],
        "g_box": g_box, "ph_box": ph_box, "g_net": pred["g_abs"], "ph_net": np.arctan2(pred["g_im"], pred["g_re"]),
        "g_ref": ref["g_abs"], "ph_ref": np.arctan2(ref["g_im"], ref["g_re"]),
    }


def make_figure(maps: dict, valid: np.ndarray, meta: dict, title: str, phase_min: float = 0.2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = valid.shape
    fc = max(1, int(np.ceil(cols / 600)))
    az = float(meta.get("azimuth_pixel_spacing_m", 27.9))
    rg = float(meta.get("range_pixel_spacing_m", 2.33))
    aspect = az / (rg * fc)

    def sub(a):
        n = (a.shape[1] // fc) * fc
        return a[:, :n].reshape(a.shape[0], n // fc, fc).mean(axis=2)

    def sub_ph(ph, g):
        out = np.where(valid & (g > phase_min), ph, np.nan)
        return out[:, : (out.shape[1] // fc) * fc: fc]

    def blank(a):
        a.set_xticks([]); a.set_yticks([])
        a.text(0.5, 0.5, "single look: |gamma| = 1\nat every pixel\n(no meaningful coherence)", ha="center", va="center",
               transform=a.transAxes, fontsize=10, color="#555")

    cmap_ph = plt.get_cmap("hsv").with_extremes(bad="#bbbbbb")
    fig, ax = plt.subplots(4, 4, figsize=(17, 20), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    kw = dict(aspect=aspect, interpolation="nearest")
    heads = ["input: one look (A)", f"standard: {BOX[0]}x{BOX[1]} boxcar (A)", "network (from A)", "reference: 9x65 boxcar (B)"]
    for j, h in enumerate(heads):
        ax[0, j].set_title(h, fontsize=12)
    for r, (key, lo, hi, name) in enumerate((("vv", -25, 0, "sigma0 VV [dB]"), ("vh", -35, -10, "sigma0 VH [dB]"))):
        for j, src in enumerate(("in", "box", "net")):
            im = ax[r, j].imshow(db(sub(np.where(valid, maps[f"{src}_{key}"], np.nan))), vmin=lo, vmax=hi, cmap="gray", **kw)
        ax[r, 0].set_ylabel(name, fontsize=12)
        ax[r, 3].axis("off")
        fig.colorbar(im, ax=ax[r, 2], fraction=0.04, pad=0.02)
    blank(ax[2, 0]); blank(ax[3, 0])
    for j, src in ((1, "box"), (2, "net"), (3, "ref")):
        im = ax[2, j].imshow(sub(np.where(valid, maps[f"g_{src}"], np.nan)), vmin=0, vmax=0.8, cmap="magma", **kw)
        imp = ax[3, j].imshow(sub_ph(maps[f"ph_{src}"], maps[f"g_{src}"]), vmin=-np.pi, vmax=np.pi, cmap=cmap_ph, **kw)
    fig.colorbar(im, ax=ax[2, 3], fraction=0.04, pad=0.02)
    fig.colorbar(imp, ax=ax[3, 3], fraction=0.04, pad=0.02)
    ax[2, 0].set_ylabel("|gamma|", fontsize=12)
    ax[3, 0].set_ylabel(f"phase of C12 [rad], |gamma| > {phase_min}", fontsize=12)
    for a in ax.flat:
        if a.axison:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("R0", "R1", "C0", "C1"), default=None)
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
    maps = build_maps(arrays, pred, ref)
    valid = arrays["valid"].astype(bool) & (pred["count"] > 0)
    if a.crop:
        r0, r1, c0, c1 = a.crop
        maps = {k: v[r0:r1, c0:c1] for k, v in maps.items()}
        valid = valid[r0:r1, c0:c1]
    out = Path(a.out).expanduser() if a.out else path.with_name(path.stem + "_showcase.png")
    fig = make_figure(maps, valid, rmeta, f"{path.stem}: single look vs standard multilook vs network")
    fig.savefig(out, dpi=80)
    print(f"saved {out}")


if __name__ == "__main__":
    main()