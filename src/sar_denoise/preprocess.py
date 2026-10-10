"""One burst through the whole preprocessing chain, ready for patch extraction.

    read VV, VH burst  ->  deramp + demodulate (same function for both)
      ->  crop to the valid region  ->  radiometric calibration  (S / A_sigma)
      ->  Doppler sub-look split (Look A, Look B)  ->  C2 per look (single look)
      ->  thermal-noise map per look  ->  validity mask  ->  power-to-noise map

All operations act identically on VV and VH, so the VV-VH phase is untouched.
Nothing is subtracted from the data: the noise map is kept separately, so the
training loss can ADD the known noise to the model's covariance instead.

Units: calibrated sigma0 (linear). Channel order of the C2 stacks:
[C11, C22, Re C12, Im C12]; noise maps: [VV, VH].

Memory: a full burst (1459 x 24441) needs several GB. Process range strips with
`cols=(first_sample, stop_sample)`; columns are independent, azimuth is not split.

Usage:
    python -m sar_denoise.preprocess /path/to/product.SAFE iw2 [--burst K] [--cols 4096] [--out file.npz]
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from sar_denoise.calibration import (
    calibrate_complex,
    find_calibration_files,
    load_calibration_lut,
    load_noise_lut,
    noise_sigma0,
)
from sar_denoise.covariance import coherence, form_c2, multilook
from sar_denoise.deramp import deramp_burst, deramp_params
from sar_denoise.safe_io import find_swath_files, parse_annotation, read_burst
from sar_denoise.sublooks import (
    SubLookConfig,
    c12_speckle_correlation,
    noise_factors,
    split_sublooks,
)


@dataclass
class BurstProducts:
    c2_a: np.ndarray       # (4, rows, cols) float32, Look A
    c2_b: np.ndarray       # (4, rows, cols) float32, Look B
    noise_a: np.ndarray    # (2, rows, cols) float32, thermal noise power in Look A [VV, VH]
    noise_b: np.ndarray    # (2, rows, cols) float32, thermal noise power in Look B
    pnr_db: np.ndarray     # (2, rows, cols) float32, smoothed measured power / noise power [VV, VH]
    valid: np.ndarray      # (rows, cols) bool
    dt_out: float          # azimuth time step of the looks [s]
    meta: dict
    looks: object = None   # the SubLooks object (complex looks), for diagnostics


def edge_deviation_rows(profile: np.ndarray, tol: float = 0.1) -> tuple[int, int]:
    """Rows at the start and at the end of a power profile that deviate more than tol
    (relative) from its central median, counted up to the first row that is within tol."""
    n = len(profile)
    ref = float(np.median(profile[int(0.3 * n): int(0.7 * n)]))
    off = np.abs(profile / ref - 1.0) > tol
    start = int(np.argmin(off)) if not off.all() else n
    end = int(np.argmin(off[::-1])) if not off.all() else n
    return start, end


def process_arrays(vv, vh, a_vv, a_vh, noise_vv, noise_vh, dt: float, cfg: SubLookConfig | None = None,
                   trim_rows: int = 32, noise_model: str = "white",
                   pnr_window: tuple[int, int] = (9, 65)) -> BurstProducts:
    """Core of the chain on arrays that are already deramped, demodulated and cropped.

    vv, vh           complex DN, shape (lines, samples)
    a_vv, a_vh       sigmaNought LUT evaluated on the same grid
    noise_vv/vh      noise LUT (DN^2) evaluated on the same grid
    trim_rows        rows (in look rows) masked at both ends: filter wrap-around and burst edges
    """
    cfg = SubLookConfig() if cfg is None else cfg
    shapes = {x.shape for x in (vv, vh, a_vv, a_vh, noise_vv, noise_vh)}
    if len(shapes) != 1:
        raise ValueError(f"all inputs must have the same shape, got {shapes}")

    cal_vv = calibrate_complex(vv.astype(np.complex64, copy=False), a_vv.astype(np.float32, copy=False))
    cal_vh = calibrate_complex(vh.astype(np.complex64, copy=False), a_vh.astype(np.float32, copy=False))
    ns_vv = noise_sigma0(noise_vv.astype(np.float32, copy=False), a_vv.astype(np.float32, copy=False))
    ns_vh = noise_sigma0(noise_vh.astype(np.float32, copy=False), a_vh.astype(np.float32, copy=False))
    zero_both = (vv == 0) & (vh == 0)

    looks = split_sublooks(cal_vv, cal_vh, dt, cfg)
    del cal_vv, cal_vh

    n_used, step, rows = looks.n_lines_used, cfg.decimate, looks.a_vv.shape[0]
    pick = lambda arr: arr[:n_used:step][:rows]          # output row j <-> input row j * step
    fa, fb = noise_factors(n_used, dt, cfg, noise_model)
    g = looks.gains

    # noise in a look = LUT noise x (what the band filter does to white noise) x (equalisation gain)^2
    noise_a = np.stack([pick(ns_vv) * (fa * g["a_vv"] ** 2),
                        pick(ns_vh) * (fa * g["a_vh"] ** 2)]).astype(np.float32)
    noise_b = np.stack([pick(ns_vv) * (fb * g["b_vv"] ** 2),
                        pick(ns_vh) * (fb * g["b_vh"] ** 2)]).astype(np.float32)

    c2_a = form_c2(looks.a_vv, looks.a_vh)
    c2_b = form_c2(looks.b_vv, looks.b_vh)

    valid = ~pick(zero_both)
    if trim_rows > 0:
        valid[:trim_rows] = False
        valid[-trim_rows:] = False

    meas = 0.5 * (c2_a[:2] + c2_b[:2])
    noi = 0.5 * (noise_a + noise_b)
    size = (1,) + tuple(pnr_window)
    sm_meas = uniform_filter(meas, size=size, mode="reflect")
    sm_noi = uniform_filter(noi, size=size, mode="reflect")
    pnr_db = (10.0 * np.log10(np.maximum(sm_meas, 1e-30) / np.maximum(sm_noi, 1e-30))).astype(np.float32)

    meta = {
        "dt_out": looks.dt_out, "decimation": step, "n_lines_used": n_used, "trim_rows": trim_rows,
        "noise_model": noise_model, "noise_factor_a": fa, "noise_factor_b": fb,
        "gains": {k: float(v) for k, v in g.items()}, "cfg": asdict(cfg), "pnr_window": list(pnr_window),
    }
    return BurstProducts(c2_a=c2_a, c2_b=c2_b, noise_a=noise_a, noise_b=noise_b, pnr_db=pnr_db,
                         valid=valid, dt_out=looks.dt_out, meta=meta, looks=looks)


def process_burst(safe_dir, swath: str, burst: int, cols: tuple[int, int] | None = None,
                  cfg: SubLookConfig | None = None, **kwargs) -> BurstProducts:
    """Read one burst of a SAFE product and run the whole chain.

    cols = (first, stop) restricts the range samples (absolute sample indices).
    """
    tiff_vv, xml_vv = find_swath_files(safe_dir, swath, "vv")
    tiff_vh, xml_vh = find_swath_files(safe_dir, swath, "vh")
    info = parse_annotation(xml_vv)
    info_vh = parse_annotation(xml_vh)
    if (len(info.bursts) != len(info_vh.bursts) or info.lines_per_burst != info_vh.lines_per_burst
            or info.samples_per_burst != info_vh.samples_per_burst):
        raise ValueError("VV and VH annotations describe different burst layouts")

    b = info.bursts[burst]
    l0, l1 = b.valid_line_range
    s0, s1 = b.valid_sample_range
    if cols is not None:
        s0, s1 = max(s0, cols[0]), min(s1, cols[1])
        if s1 <= s0:
            raise ValueError(f"cols {cols} lie outside the valid samples {b.valid_sample_range}")

    # VV and VH are deramped with the SAME (VV) parameters; report how far the VH annotation differs
    p_vv, p_vh = deramp_params(info, burst), deramp_params(info_vh, burst)
    mismatch = {"f_dc_hz": float(np.max(np.abs(p_vv.f_dc - p_vh.f_dc))),
                "k_t_hz_per_s": float(np.max(np.abs(p_vv.k_t - p_vh.k_t)))}

    vv = deramp_burst(read_burst(tiff_vv, info, burst), info, burst, demodulate=True)[l0:l1, s0:s1].copy()
    vh = deramp_burst(read_burst(tiff_vh, info_vh, burst), info, burst, demodulate=True)[l0:l1, s0:s1].copy()

    lines = b.line_start + np.arange(l0, l1)
    samples = np.arange(s0, s1)
    luts = {}
    for pol in ("vv", "vh"):
        cal_xml, noise_xml = find_calibration_files(safe_dir, swath, pol)
        luts[pol] = (load_calibration_lut(cal_xml, "sigmaNought").evaluate(lines, samples),
                     load_noise_lut(noise_xml).evaluate(lines, samples))

    cfg = SubLookConfig.from_info(info) if cfg is None else cfg
    prod = process_arrays(vv, vh, luts["vv"][0], luts["vh"][0], luts["vv"][1], luts["vh"][1],
                          info.azimuth_time_interval, cfg, **kwargs)
    prod.meta.update({
        "product": Path(str(safe_dir)).name, "swath": info.swath, "burst": burst,
        "valid_lines": [l0, l1], "samples": [s0, s1], "line_start": b.line_start,
        "azimuth_pixel_spacing_m": info.azimuth_pixel_spacing * cfg.decimate,
        "range_pixel_spacing_m": info.range_pixel_spacing, "deramp_vv_vs_vh_mismatch": mismatch,
    })
    return prod


def _db(x: float) -> float:
    return float(10.0 * np.log10(x))


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the preprocessing chain on one burst and summarise it")
    ap.add_argument("safe_dir")
    ap.add_argument("swath")
    ap.add_argument("--burst", type=int, default=None)
    ap.add_argument("--cols", type=int, default=4096, help="number of range samples (centre of the valid area)")
    ap.add_argument("--trim", type=int, default=32, help="look rows masked at each end")
    ap.add_argument("--out", default=None, help="save the products to this .npz file")
    args = ap.parse_args()

    _, xml = find_swath_files(args.safe_dir, args.swath, "vv")
    info = parse_annotation(xml)
    k = args.burst if args.burst is not None else len(info.bursts) // 2
    s0, s1 = info.bursts[k].valid_sample_range
    mid = (s0 + s1) // 2
    c0 = max(s0, mid - args.cols // 2)
    prod = process_burst(args.safe_dir, args.swath, k, cols=(c0, c0 + args.cols), trim_rows=args.trim)

    m = prod.meta
    rows, ncols = prod.c2_a.shape[1:]
    print(f"burst {k}, samples {m['samples']}, looks {rows} x {ncols}, valid {prod.valid.mean():.3f} of pixels")
    print(f"pixel size {m['azimuth_pixel_spacing_m']:.1f} m (azimuth) x {m['range_pixel_spacing_m']:.2f} m (range), "
          f"dt_out {prod.dt_out * 1e3:.3f} ms")
    print("equalisation gains (dB): " + ", ".join(f"{key} {20 * np.log10(v):+.2f}" for key, v in m["gains"].items()))
    print(f"noise factors ({m['noise_model']}): A x{m['noise_factor_a']:.3f}, B x{m['noise_factor_b']:.3f}")
    mm = m["deramp_vv_vs_vh_mismatch"]
    print(f"VV vs VH deramp parameters: max |df_dc| {mm['f_dc_hz']:.3f} Hz, max |dk_t| {mm['k_t_hz_per_s']:.3f} Hz/s")

    v = prod.valid
    for i, pol in enumerate(("VV", "VH")):
        meas = 0.5 * (prod.c2_a[i][v].mean() + prod.c2_b[i][v].mean())
        noi = 0.5 * (prod.noise_a[i][v].mean() + prod.noise_b[i][v].mean())
        print(f"{pol}: mean sigma0 (looks) {_db(meas):.1f} dB, noise floor {_db(noi):.1f} dB, "
              f"median power/noise {np.median(prod.pnr_db[i][v]):.1f} dB, "
              f"pixels with power/noise < 3 dB: {np.mean(prod.pnr_db[i][v] < 3.0):.2f}")

    profile = prod.c2_a[0].mean(axis=1)
    start, end = edge_deviation_rows(profile)
    print(f"VV power profile along azimuth: first {start} and last {end} look rows deviate >10% "
          f"from the central level (masked: {args.trim} at each end)")

    c2 = multilook(prod.c2_a, (9, 65))
    gamma, _ = coherence(c2)
    g = gamma[v]
    print(f"local |coherence| (9 x 65 boxcar, Look A, no noise correction): "
          f"median {np.median(g):.2f}, 90% {np.percentile(g, 90):.2f}, 99% {np.percentile(g, 99):.2f}, "
          f">0.3: {np.mean(g > 0.3):.3f}")
    lk = prod.looks
    print(f"C12 speckle correlation between looks: "
          f"{c12_speckle_correlation(lk.a_vv, lk.a_vh, lk.b_vv, lk.b_vh):.3f}")

    if args.out:
        np.savez_compressed(args.out, c2_a=prod.c2_a, c2_b=prod.c2_b, noise_a=prod.noise_a,
                            noise_b=prod.noise_b, pnr_db=prod.pnr_db, valid=prod.valid,
                            dt_out=prod.dt_out, meta=str(prod.meta))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()