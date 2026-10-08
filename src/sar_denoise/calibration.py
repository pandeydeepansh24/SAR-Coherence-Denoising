"""Sentinel-1 calibration and thermal-noise look-up tables (LUTs).

Both files (annotation/calibration/calibration-*.xml and noise-*.xml) store a
coarse grid of vectors: one vector per `line`, holding values at a list of
`pixel` positions. Line and pixel numbers refer to the whole subswath image
(all bursts stacked, as in the measurement TIFF).

Conventions (ESA product specification):
    sigma0 (linear) = |DN|^2 / A_sigma^2                 A_sigma = sigmaNought LUT
    noise floor in sigma0 units = noiseLut / A_sigma^2   (noiseLut is in DN^2)
For complex data the amplitude calibration is  S_cal = S / A_sigma,  so that
|S_cal|^2 is sigma0. VV and VH each use their own LUT.

The sigmaNought LUT is used as is; do not multiply by
absoluteCalibrationConstant again.

Usage:
    python -m sar_denoise.calibration /path/to/product.SAFE iw2 vv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lxml import etree

CAL_KINDS = ("sigmaNought", "betaNought", "gamma", "dn")


@dataclass
class Lut:
    lines: np.ndarray            # (n_vec,) sorted line numbers
    pixels: list                 # per vector: pixel positions
    values: list                 # per vector: LUT values at those pixels

    def evaluate(self, lines, pixels, dtype=np.float32) -> np.ndarray:
        """Bilinear interpolation onto lines x pixels; clamped outside the grid.

        Returns an array of shape (len(lines), len(pixels)).
        """
        lines = np.asarray(lines, dtype=float)
        pixels = np.asarray(pixels, dtype=float)
        if len(self.lines) < 2:
            raise ValueError("need at least two LUT vectors to interpolate")

        # 1) along range: bring every vector onto the requested pixel grid
        rows = np.stack([np.interp(pixels, p, v) for p, v in zip(self.pixels, self.values)])
        rows = rows.astype(dtype)

        # 2) along azimuth: linear between the two bracketing vectors
        idx = np.clip(np.searchsorted(self.lines, lines, side="right") - 1, 0, len(self.lines) - 2)
        l0, l1 = self.lines[idx], self.lines[idx + 1]
        w = np.clip((lines - l0) / (l1 - l0), 0.0, 1.0)

        out = np.empty((len(lines), len(pixels)), dtype=dtype)
        for k in np.unique(idx):
            sel = idx == k
            wk = w[sel].astype(dtype)[:, None]
            out[sel] = (1 - wk) * rows[k] + wk * rows[k + 1]
        return out


def _load(xml_path, list_path: str, value_tag: str) -> Lut:
    root = etree.parse(str(xml_path)).getroot()
    vectors = root.findall(list_path)
    if not vectors:
        raise ValueError(f"no <{list_path}> entries in {xml_path}")
    items = []
    for v in vectors:
        line = int(v.findtext("line"))
        pix = np.array(v.findtext("pixel").split(), dtype=float)
        val = np.array(v.findtext(value_tag).split(), dtype=float)
        if pix.size != val.size:
            raise ValueError(f"pixel/{value_tag} length mismatch at line {line}")
        items.append((line, pix, val))
    items.sort(key=lambda t: t[0])
    return Lut(
        lines=np.array([t[0] for t in items], dtype=float),
        pixels=[t[1] for t in items],
        values=[t[2] for t in items],
    )


def load_calibration_lut(xml_path, kind: str = "sigmaNought") -> Lut:
    if kind not in CAL_KINDS:
        raise ValueError(f"kind must be one of {CAL_KINDS}")
    return _load(xml_path, "calibrationVectorList/calibrationVector", kind)


def load_noise_lut(xml_path) -> Lut:
    """Range noise LUT (older layout: noiseVectorList/noiseVector/noiseLut).

    Newer processor versions use noiseRangeVectorList (handled here) and add an
    azimuth noise LUT, which is NOT handled by this function.
    """
    root = etree.parse(str(xml_path)).getroot()
    if root.find("noiseVectorList/noiseVector") is not None:
        return _load(xml_path, "noiseVectorList/noiseVector", "noiseLut")
    if root.find("noiseRangeVectorList/noiseRangeVector") is not None:
        return _load(xml_path, "noiseRangeVectorList/noiseRangeVector", "noiseRangeLut")
    raise ValueError(f"no recognised noise vectors in {xml_path}")


def find_calibration_files(safe_dir, swath: str, pol: str) -> tuple[Path, Path]:
    """Return (calibration xml, noise xml) for one subswath and polarisation."""
    d = Path(safe_dir).expanduser() / "annotation" / "calibration"
    swath, pol = swath.lower(), pol.lower()
    cal = sorted(d.glob(f"calibration-s1?-{swath}-slc-{pol}-*.xml"))
    noi = sorted(d.glob(f"noise-s1?-{swath}-slc-{pol}-*.xml"))
    if len(cal) != 1 or len(noi) != 1:
        raise FileNotFoundError(
            f"Expected one calibration and one noise xml for {swath}/{pol} in {d}, "
            f"found {len(cal)} and {len(noi)}."
        )
    return cal[0], noi[0]


def evaluate_for_burst(lut: Lut, burst, sample_start: int = 0, sample_stop: int | None = None,
                       dtype=np.float32) -> np.ndarray:
    """Evaluate a LUT on the pixel grid of one burst (see safe_io.Burst).

    Optionally restrict to samples [sample_start, sample_stop) to save memory.
    """
    lines = burst.line_start + np.arange(burst.n_lines)
    stop = burst.n_samples if sample_stop is None else sample_stop
    return lut.evaluate(lines, np.arange(sample_start, stop), dtype=dtype)


def calibrate_complex(s: np.ndarray, a_sigma: np.ndarray) -> np.ndarray:
    """S_cal = S / A_sigma  (|S_cal|^2 is sigma0, linear)."""
    if s.shape != a_sigma.shape:
        raise ValueError(f"shape mismatch: {s.shape} vs {a_sigma.shape}")
    return s / a_sigma


def noise_sigma0(noise_dn2: np.ndarray, a_sigma: np.ndarray) -> np.ndarray:
    """Thermal noise floor in sigma0 (linear) units."""
    return noise_dn2 / (a_sigma * a_sigma)


def _db(x: float) -> float:
    return 10.0 * np.log10(x)


def main() -> None:
    from sar_denoise.safe_io import find_swath_files, parse_annotation, read_burst

    p = argparse.ArgumentParser(description="Sanity-check calibration and noise LUTs on one burst")
    p.add_argument("safe_dir")
    p.add_argument("swath")
    p.add_argument("pol")
    p.add_argument("--burst", type=int, default=None)
    args = p.parse_args()

    tiff, xml = find_swath_files(args.safe_dir, args.swath, args.pol)
    info = parse_annotation(xml)
    cal_xml, noise_xml = find_calibration_files(args.safe_dir, args.swath, args.pol)
    cal = load_calibration_lut(cal_xml, "sigmaNought")
    noi = load_noise_lut(noise_xml)

    print(f"calibration vectors: {len(cal.lines)}, lines {int(cal.lines[0])}..{int(cal.lines[-1])}, "
          f"{len(cal.pixels[0])} pixels each")
    print(f"noise vectors      : {len(noi.lines)}, lines {int(noi.lines[0])}..{int(noi.lines[-1])}, "
          f"{len(noi.pixels[0])} pixels each")
    print(f"sigmaNought LUT range: {min(v.min() for v in cal.values):.1f} .. "
          f"{max(v.max() for v in cal.values):.1f}")

    k = args.burst if args.burst is not None else len(info.bursts) // 2
    b = info.bursts[k]
    l0, l1 = b.valid_line_range
    s0, s1 = b.valid_sample_range

    data = read_burst(tiff, info, k)[l0:l1, s0:s1]
    lines = b.line_start + np.arange(l0, l1)
    samples = np.arange(s0, s1)
    a = cal.evaluate(lines, samples)
    n = noi.evaluate(lines, samples)

    sig0 = (np.abs(data) ** 2) / (a * a)
    nz = noise_sigma0(n, a)
    print(f"\nburst {k}, valid region {data.shape}")
    print(f"mean sigma0        : {_db(sig0.mean()):.1f} dB")
    print(f"mean noise floor   : {_db(nz.mean()):.1f} dB")
    print(f"mean SNR           : {_db(sig0.mean() / nz.mean()):.1f} dB")
    print(f"pixels below noise : {np.mean(sig0 < nz):.3f}")


if __name__ == "__main__":
    main()