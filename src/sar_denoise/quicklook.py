"""Coarse quick-look of whole sub-swaths: where is there VV-VH coherence, and is VH above the noise?

Why this is cheap: C11 = |S_vv|^2, C22 = |S_vh|^2 and C12 = S_vv S_vh* do not change when
both channels are multiplied by the same phase, so deramping is not needed here. Each burst
is read, block-averaged (default 8 lines x 64 samples, about 110 m x 150 m on the ground),
calibrated at the block centres, and summarised:

    sigma0 VV / VH [dB]    brightness
    VH power/noise [dB]    measured VH power over the thermal-noise floor (0 dB = pure noise)
    |coherence|            |<C12>| / sqrt(<C11><C22>) of each block, no noise correction

Pure speckle already gives |coherence| of about 0.05 for a 512-sample block, so only values
clearly above that mean real VV-VH correlation. VH noise biases coherence low.

Outputs per sub-swath: a PNG mosaic (all bursts stacked) and a CSV with per-tile statistics,
including the absolute TIFF line / sample window of every tile, so patches can later be
taken from the interesting places.

Usage:
    python -m sar_denoise.quicklook /path/to/product.SAFE [--swath iw2 ...] [--out-dir /tmp/quicklook]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sar_denoise.calibration import find_calibration_files, load_calibration_lut, load_noise_lut
from sar_denoise.safe_io import find_swath_files, parse_annotation, read_burst


def coarse_c2(vv: np.ndarray, vh: np.ndarray, block=(8, 64), chunk_rows: int = 64):
    """Block means of |vv|^2, |vh|^2 and vv*conj(vh) (DN units).

    Partial blocks at the bottom/right are dropped. Returns (c11, c22, c12), each of shape
    (n // block[0], m // block[1]); c12 is complex.
    """
    ba, br = block
    n, m = vv.shape
    if vh.shape != (n, m):
        raise ValueError("vv and vh must have the same shape")
    na, nr = n // ba, m // br
    c11 = np.empty((na, nr), dtype=np.float32)
    c22 = np.empty((na, nr), dtype=np.float32)
    c12 = np.empty((na, nr), dtype=np.complex64)
    step = max(1, chunk_rows // ba) * ba

    def bm(x, rows):
        return x.reshape(rows // ba, ba, nr, br).mean(axis=(1, 3), dtype=np.float64)

    for i0 in range(0, na * ba, step):
        i1 = min(i0 + step, na * ba)
        a, b = vv[i0:i1, : nr * br], vh[i0:i1, : nr * br]
        k0, k1 = i0 // ba, i1 // ba
        c11[k0:k1] = bm(a.real ** 2 + a.imag ** 2, i1 - i0)
        c22[k0:k1] = bm(b.real ** 2 + b.imag ** 2, i1 - i0)
        prod = a * np.conj(b)
        c12[k0:k1] = bm(prod.real, i1 - i0) + 1j * bm(prod.imag, i1 - i0)
    return c11, c22, c12


def coherence_map(c11, c22, c12, eps: float = 1e-30) -> np.ndarray:
    return np.abs(c12) / np.sqrt(np.maximum(c11 * c22, eps))


def tile_stats(gamma, pnr_vh_db, sigma_vv_db, tile=(16, 16), threshold: float = 0.3) -> list[dict]:
    """Statistics of non-overlapping tiles (in block units) of one burst's maps."""
    th, tw = tile
    na, nr = gamma.shape
    out = []
    for i in range(na // th):
        for j in range(nr // tw):
            sl = (slice(i * th, (i + 1) * th), slice(j * tw, (j + 1) * tw))
            g = gamma[sl]
            out.append({
                "row": i * th, "col": j * tw,
                "gamma_mean": float(g.mean()), "gamma_p90": float(np.percentile(g, 90)),
                "frac_gamma_above": float(np.mean(g > threshold)),
                "pnr_vh_db": float(pnr_vh_db[sl].mean()), "sigma_vv_db": float(sigma_vv_db[sl].mean()),
            })
    return out


def scan_swath(safe_dir, swath: str, block=(8, 64), tile=(16, 16), bursts=None, threshold: float = 0.3,
               progress=print) -> dict:
    """Scan the bursts of one sub-swath. Returns stacked maps and a list of tile rows."""
    tiff_vv, xml_vv = find_swath_files(safe_dir, swath, "vv")
    tiff_vh, xml_vh = find_swath_files(safe_dir, swath, "vh")
    info = parse_annotation(xml_vv)
    info_vh = parse_annotation(xml_vh)
    luts = {}
    for pol in ("vv", "vh"):
        cal_xml, noise_xml = find_calibration_files(safe_dir, swath, pol)
        luts[pol] = (load_calibration_lut(cal_xml, "sigmaNought"), load_noise_lut(noise_xml))

    ba, br = block
    per_burst, tiles = [], []
    for k in (range(len(info.bursts)) if bursts is None else bursts):
        b = info.bursts[k]
        l0, l1 = b.valid_line_range
        s0, s1 = b.valid_sample_range
        vv = read_burst(tiff_vv, info, k)[l0:l1, s0:s1]
        vh = read_burst(tiff_vh, info_vh, k)[l0:l1, s0:s1]
        c11, c22, c12 = coarse_c2(vv, vh, block)
        del vv, vh

        na, nr = c11.shape
        lines = b.line_start + l0 + (np.arange(na) + 0.5) * ba
        samples = s0 + (np.arange(nr) + 0.5) * br
        a_vv = luts["vv"][0].evaluate(lines, samples)
        a_vh = luts["vh"][0].evaluate(lines, samples)
        noise_vh = luts["vh"][1].evaluate(lines, samples) / a_vh ** 2
        sig_vv = c11 / a_vv ** 2
        sig_vh = c22 / a_vh ** 2
        tiny = 1e-30
        maps = {
            "sigma_vv_db": 10 * np.log10(np.maximum(sig_vv, tiny)),
            "sigma_vh_db": 10 * np.log10(np.maximum(sig_vh, tiny)),
            "pnr_vh_db": 10 * np.log10(np.maximum(sig_vh, tiny) / noise_vh),
            "gamma": coherence_map(c11, c22, c12),
        }
        per_burst.append(maps)
        for t in tile_stats(maps["gamma"], maps["pnr_vh_db"], maps["sigma_vv_db"], tile, threshold):
            t.update({"swath": swath, "burst": k,
                      "line": int(b.line_start + l0 + t["row"] * ba), "lines": tile[0] * ba,
                      "sample": int(s0 + t["col"] * br), "samples": tile[1] * br})
            tiles.append(t)
        progress(f"{swath} burst {k}: |coherence| median {np.median(maps['gamma']):.3f}, "
                 f"VH power/noise median {np.median(maps['pnr_vh_db']):+.1f} dB")

    rows = min(m["gamma"].shape[0] for m in per_burst)
    cols = min(m["gamma"].shape[1] for m in per_burst)
    stacked = {key: np.concatenate([m[key][:rows, :cols] for m in per_burst], axis=0) for key in per_burst[0]}
    return {"maps": stacked, "rows_per_burst": rows, "tiles": tiles, "swath": swath, "block": block}


def save_mosaic(result: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = result["maps"]
    spec = [("sigma_vv_db", "sigma0 VV [dB]", -25, 0, "gray"),
            ("sigma_vh_db", "sigma0 VH [dB]", -35, -10, "gray"),
            ("pnr_vh_db", "VH power / noise [dB]", -3, 15, "viridis"),
            ("gamma", "|VV-VH coherence| (blocks)", 0, 0.4, "magma")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 12))
    for ax, (key, title, lo, hi, cmap) in zip(axes, spec):
        im = ax.imshow(maps[key], vmin=lo, vmax=hi, cmap=cmap, aspect="equal", interpolation="nearest")
        for r in range(result["rows_per_burst"], maps[key].shape[0], result["rows_per_burst"]):
            ax.axhline(r - 0.5, color="white", lw=0.4, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    fig.suptitle(f"{result['swath']}: bursts stacked top to bottom (white lines), block {result['block']}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_tiles_csv(tiles: list[dict], path) -> None:
    fields = ["swath", "burst", "line", "lines", "sample", "samples", "gamma_mean", "gamma_p90",
              "frac_gamma_above", "pnr_vh_db", "sigma_vv_db"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for t in tiles:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items() if k in fields})


def main() -> None:
    ap = argparse.ArgumentParser(description="Coarse quick-look of sub-swaths (coherence, VH power/noise)")
    ap.add_argument("safe_dir")
    ap.add_argument("--swath", nargs="+", default=["iw1", "iw2", "iw3"])
    ap.add_argument("--block", nargs=2, type=int, default=[8, 64], metavar=("AZ", "RG"))
    ap.add_argument("--tile", nargs=2, type=int, default=[16, 16], metavar=("AZ", "RG"),
                    help="tile size in blocks (default 16 x 16, about 1.8 km x 2.4 km)")
    ap.add_argument("--bursts", nargs="+", type=int, default=None)
    ap.add_argument("--out-dir", default="/tmp/quicklook")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_tiles = []
    for swath in args.swath:
        res = scan_swath(args.safe_dir, swath, tuple(args.block), tuple(args.tile), args.bursts)
        g, p = res["maps"]["gamma"], res["maps"]["pnr_vh_db"]
        print(f"\n{swath}: |coherence| of blocks: median {np.median(g):.3f}, 90% {np.percentile(g, 90):.3f}, "
              f"99% {np.percentile(g, 99):.3f}, >0.2 {np.mean(g > 0.2):.3f}, >0.3 {np.mean(g > 0.3):.3f}")
        print(f"{swath}: VH power/noise: 1% {np.percentile(p, 1):+.1f}, median {np.median(p):+.1f}, "
              f"99% {np.percentile(p, 99):+.1f} dB, blocks above 3 dB: {np.mean(p > 3):.3f}")
        save_mosaic(res, out / f"{swath}_quicklook.png")
        write_tiles_csv(res["tiles"], out / f"{swath}_tiles.csv")
        all_tiles += res["tiles"]
        print(f"saved {out / (swath + '_quicklook.png')} and {out / (swath + '_tiles.csv')}")

    ranked = sorted(all_tiles, key=lambda t: t["gamma_p90"], reverse=True)[: args.top]
    print(f"\ntop {len(ranked)} tiles by 90th-percentile |coherence| (tile = {args.tile[0] * args.block[0]} lines "
          f"x {args.tile[1] * args.block[1]} samples):")
    print("swath burst   line  sample  gamma_p90 frac>thr  VH p/n [dB]  VV [dB]")
    for t in ranked:
        print(f"{t['swath']:5s} {t['burst']:5d} {t['line']:6d} {t['sample']:7d}  {t['gamma_p90']:8.3f} "
              f"{t['frac_gamma_above']:8.3f}  {t['pnr_vh_db']:10.1f}  {t['sigma_vv_db']:7.1f}")


if __name__ == "__main__":
    main()