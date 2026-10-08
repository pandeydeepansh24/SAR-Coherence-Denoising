"""Doppler sub-look splitting of deramped + demodulated Sentinel-1 bursts.

The azimuth spectrum of a burst occupies [-B/2, B/2] (B = processing bandwidth,
313 Hz for IW2) with a Hamming weighting. Two non-overlapping bands, separated by
a guard gap, give two looks of the same scene whose speckle is (nearly)
independent: Look A = lower band, Look B = upper band.

For each look (identical operation on VV and VH, which keeps C12 intact):
  1. FFT along azimuth
  2. multiply by the band filter (optionally divided by the Hamming weighting,
     so the look has a flat spectrum), with a short cosine ramp at the inner edge
  3. shift the band centre to 0 Hz (integer FFT-bin shift)
  4. keep only the bins within +-fs/(2M) and inverse FFT  ->  decimation by M
  5. scale the amplitude so that the look has the same mean POWER as the
     full-band product (the sigma0 level of a distributed target does not depend
     on resolution)

Assumption to verify on real data: thermal noise is shaped by the same azimuth
weighting as the signal. If so, step 5 also leaves the noise floor unchanged, so
the noise LUT applies to the looks as is.

Input rows must be azimuth lines (axis 0) and must already be deramped and
demodulated (deramp.deramp_burst) and cropped to the valid region. Output row j
corresponds to input row j * decimate. The first and last ~1/edge_hz seconds of
each output are affected by filter ringing/wrap-around; trim them when cutting
patches.

Usage:
    python -m sar_denoise.sublooks /path/to/product.SAFE iw2 [--burst K] [--cols 4096]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft


@dataclass
class SubLookConfig:
    bandwidth_hz: float = 313.0        # processed azimuth bandwidth (annotation)
    window_coefficient: float = 0.75   # Hamming coefficient (annotation)
    guard_hz: float = 10.0             # gap between the two bands
    edge_hz: float = 4.0               # cosine ramp at the inner band edges (inside each band)
    split_hz: float = 0.0              # centre of the gap (after demodulation)
    deweight: bool = True              # divide by the Hamming weighting inside each band
    decimate: int = 2                  # azimuth decimation factor

    @classmethod
    def from_info(cls, info, **overrides) -> "SubLookConfig":
        """Take bandwidth and window from the SwathInfo of safe_io.parse_annotation."""
        kw = {}
        if info.processing_bandwidth:
            kw["bandwidth_hz"] = float(info.processing_bandwidth)
        if (info.window_type or "").lower().startswith("hamming") and info.window_coefficient is not None:
            kw["window_coefficient"] = float(info.window_coefficient)
        else:
            kw["deweight"] = False      # unknown weighting: do not try to undo it
        kw.update(overrides)
        return cls(**kw)


@dataclass
class SubLooks:
    a_vv: np.ndarray
    a_vh: np.ndarray
    b_vv: np.ndarray
    b_vh: np.ndarray
    dt_out: float           # azimuth time step of the looks [s]
    n_lines_used: int       # input lines used (input length rounded down to a multiple of decimate)
    decimation: int
    cfg: SubLookConfig


def hamming_weight(freq: np.ndarray, bandwidth_hz: float, alpha: float) -> np.ndarray:
    """Amplitude weighting alpha + (1 - alpha) cos(2 pi f / B) on |f| <= B/2, else 0."""
    w = alpha + (1.0 - alpha) * np.cos(2.0 * np.pi * freq / bandwidth_hz)
    return np.where(np.abs(freq) <= bandwidth_hz / 2.0, w, 0.0)


def _ramp(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def band_edges(cfg: SubLookConfig) -> tuple[tuple[float, float], tuple[float, float]]:
    """((lo_A, hi_A), (lo_B, hi_B)) in Hz."""
    half = cfg.bandwidth_hz / 2.0
    return ((-half, cfg.split_hz - cfg.guard_hz / 2.0),
            (cfg.split_hz + cfg.guard_hz / 2.0, half))


def look_filters(freq: np.ndarray, cfg: SubLookConfig):
    """[(amplitude filter, centre_hz)] for look A and look B, on the given frequency grid.

    The filters include the radiometric gain described in the module docstring.
    """
    half = cfg.bandwidth_hz / 2.0
    (lo_a, hi_a), (lo_b, hi_b) = band_edges(cfg)
    e = max(cfg.edge_hz, 1e-9)
    inside = (np.abs(freq) <= half).astype(float)
    m_a = inside * (1.0 - _ramp((freq - (hi_a - e)) / e))
    m_b = inside * _ramp((freq - lo_b) / e)

    w = hamming_weight(freq, cfg.bandwidth_hz, cfg.window_coefficient)
    if cfg.deweight:
        inv = np.zeros_like(w)
        np.divide(1.0, w, out=inv, where=w > 0)
        m_a, m_b = m_a * inv, m_b * inv

    full_power = np.sum(w ** 2)
    out = []
    for m, centre in ((m_a, (lo_a + hi_a) / 2.0), (m_b, (lo_b + hi_b) / 2.0)):
        gain = np.sqrt(full_power / np.sum((m * w) ** 2))
        out.append((m * gain, centre))
    return out


def _check_aliasing(cfg: SubLookConfig, dt: float) -> None:
    fs_out_half = 1.0 / (2.0 * cfg.decimate * dt)
    for lo, hi in band_edges(cfg):
        if (hi - lo) / 2.0 > fs_out_half:
            raise ValueError(
                f"decimate={cfg.decimate} would alias: a look is {hi - lo:.1f} Hz wide but the "
                f"decimated sampling rate only supports {2 * fs_out_half:.1f} Hz"
            )


def _extract_look(spectrum: np.ndarray, filt: np.ndarray, centre_hz: float, df: float, n_out: int):
    """spectrum: (N, cols) FFT along axis 0. Returns (n_out, cols) complex64 baseband look."""
    n = spectrum.shape[0]
    k_c = int(round(centre_hz / df))
    y = spectrum * filt.astype(np.float32)[:, None]
    y = np.roll(y, -k_c, axis=0)
    h = (n_out + 1) // 2
    z = np.concatenate([y[:h], y[n - (n_out - h):]], axis=0)
    return sfft.ifft(z, axis=0, workers=-1) * (n_out / n)


def split_sublooks(s_vv: np.ndarray, s_vh: np.ndarray, dt: float,
                   cfg: SubLookConfig | None = None, col_chunk: int = 2048) -> SubLooks:
    """Split deramped, demodulated complex VV and VH into Look A and Look B."""
    cfg = SubLookConfig() if cfg is None else cfg
    s_vv, s_vh = np.asarray(s_vv), np.asarray(s_vh)
    if s_vv.shape != s_vh.shape or s_vv.ndim != 2:
        raise ValueError(f"need two 2-D arrays of equal shape, got {s_vv.shape} and {s_vh.shape}")
    if not (np.iscomplexobj(s_vv) and np.iscomplexobj(s_vh)):
        raise ValueError("sub-look splitting needs complex (SLC) input")
    if cfg.decimate < 1:
        raise ValueError("decimate must be >= 1")
    _check_aliasing(cfg, dt)

    n_lines, n_cols = s_vv.shape
    n = n_lines - (n_lines % cfg.decimate)
    if n < 64:
        raise ValueError("too few azimuth lines")
    n_out = n // cfg.decimate
    freq = np.fft.fftfreq(n, d=dt)
    df = 1.0 / (n * dt)
    filters = look_filters(freq, cfg)

    out = {k: np.empty((n_out, n_cols), dtype=np.complex64) for k in ("a_vv", "a_vh", "b_vv", "b_vh")}
    for c0 in range(0, n_cols, col_chunk):
        c1 = min(c0 + col_chunk, n_cols)
        for pol, arr in (("vv", s_vv), ("vh", s_vh)):
            spec = sfft.fft(arr[:n, c0:c1].astype(np.complex64), axis=0, workers=-1)
            for name, (filt, centre) in zip(("a", "b"), filters):
                out[f"{name}_{pol}"][:, c0:c1] = _extract_look(spec, filt, centre, df, n_out)

    return SubLooks(dt_out=dt * cfg.decimate, n_lines_used=n, decimation=cfg.decimate, cfg=cfg, **out)


# ---------------------------------------------------------------- diagnostics
def complex_correlation(a: np.ndarray, b: np.ndarray) -> complex:
    """sum(a b*) / sqrt(sum|a|^2 sum|b|^2). About 0 for independent looks."""
    a = a.astype(np.complex128, copy=False)
    b = b.astype(np.complex128, copy=False)
    den = np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
    return complex(np.sum(a * np.conj(b)) / den)


def speckle_intensity_correlation(a: np.ndarray, b: np.ndarray, win: tuple[int, int] = (9, 65)) -> float:
    """Correlation of texture-normalised intensities |a|^2/<|a|^2> and |b|^2/<|b|^2>.

    Dividing by a local mean removes the shared scene texture, so this is about 0
    for independent speckle (and |complex correlation|^2 for partially coherent looks).
    """
    from scipy.ndimage import uniform_filter

    ia = (np.abs(a) ** 2).astype(np.float32)
    ib = (np.abs(b) ** 2).astype(np.float32)
    ma = uniform_filter(ia, size=win, mode="reflect")
    mb = uniform_filter(ib, size=win, mode="reflect")
    ok = (ma > 0) & (mb > 0)
    na, nb = ia[ok] / ma[ok], ib[ok] / mb[ok]
    return float(np.corrcoef(na, nb)[0, 1])


def noise_factors(n_lines: int, dt: float, cfg: "SubLookConfig | None" = None,
                  noise_psd="window") -> tuple[float, float]:
    """Factor by which the LUT noise power is multiplied in (look A, look B).

    noise_psd: "window" = noise spectrum shaped like the signal (factors are 1),
               "white"  = flat inside the processed band,
               or an array of length n_lines (fftfreq order) with the measured noise spectrum.
    """
    cfg = SubLookConfig() if cfg is None else cfg
    freq = np.fft.fftfreq(n_lines, d=dt)
    w = hamming_weight(freq, cfg.bandwidth_hz, cfg.window_coefficient)
    if isinstance(noise_psd, str):
        if noise_psd == "window":
            psd = w ** 2
        elif noise_psd == "white":
            psd = (np.abs(freq) <= cfg.bandwidth_hz / 2.0).astype(float)
        else:
            raise ValueError("noise_psd must be 'window', 'white' or an array")
    else:
        psd = np.asarray(noise_psd, dtype=float)
        if psd.shape != freq.shape:
            raise ValueError("noise_psd array must have length n_lines")
    fa, fb = (f for f, _ in look_filters(freq, cfg))
    total = np.sum(psd)
    return float(np.sum(fa ** 2 * psd) / total), float(np.sum(fb ** 2 * psd) / total)


def dark_column_spectrum(block: np.ndarray, dt: float, fraction: float = 0.1, smooth: int = 33):
    """Mean azimuth power spectrum over the darkest range columns of a (lines, cols) block.

    Columns are ranked by their mean power over all lines, smoothed over `smooth`
    columns; the lowest `fraction` are kept. In VH these columns are dominated by
    thermal noise, so the spectrum shows the noise spectral shape.
    Returns (freq, power), fftshifted, power normalised to a maximum of 1.
    """
    from scipy.ndimage import uniform_filter1d

    col_power = np.mean(np.abs(block) ** 2, axis=0)
    col_power = uniform_filter1d(col_power, size=min(smooth, len(col_power)), mode="nearest")
    k = max(1, int(round(fraction * block.shape[1])))
    cols = np.argsort(col_power)[:k]
    x = block[:, cols]
    win = np.hanning(x.shape[0]).astype(np.float32)[:, None]
    power = np.mean(np.abs(sfft.fft(x * win, axis=0)) ** 2, axis=1)
    freq = np.fft.fftfreq(x.shape[0], d=dt)
    power = np.fft.fftshift(power)
    return np.fft.fftshift(freq), power / power.max()


def edge_centre_db(freq: np.ndarray, power: np.ndarray, bandwidth_hz: float) -> float:
    """Power near the band edges (0.85-0.95 of the half-band) relative to the centre (<0.15), in dB."""
    half = bandwidth_hz / 2.0
    a = np.abs(freq)
    edge = power[(a > 0.85 * half) & (a < 0.95 * half)].mean()
    centre = power[a < 0.15 * half].mean()
    return float(10.0 * np.log10(edge / centre))


def _db(x: float) -> float:
    return 10.0 * np.log10(x)


def main() -> None:
    from sar_denoise.deramp import deramp_burst
    from sar_denoise.safe_io import find_swath_files, parse_annotation, read_burst

    ap = argparse.ArgumentParser(description="Split one burst into Doppler sub-looks and report diagnostics")
    ap.add_argument("safe_dir")
    ap.add_argument("swath")
    ap.add_argument("--burst", type=int, default=None)
    ap.add_argument("--cols", type=int, default=4096, help="number of range samples to use (centre of the valid area)")
    ap.add_argument("--guard", type=float, default=None)
    ap.add_argument("--no-deweight", action="store_true")
    args = ap.parse_args()

    blocks = {}
    info = None
    for pol in ("vv", "vh"):
        tiff, xml = find_swath_files(args.safe_dir, args.swath, pol)
        info = parse_annotation(xml)
        k = args.burst if args.burst is not None else len(info.bursts) // 2
        b = info.bursts[k]
        l0, l1 = b.valid_line_range
        s0, s1 = b.valid_sample_range
        mid = (s0 + s1) // 2
        c0 = max(s0, mid - args.cols // 2)
        c1 = min(s1, c0 + args.cols)
        data = deramp_burst(read_burst(tiff, info, k), info, k, demodulate=True)
        blocks[pol] = data[l0:l1, c0:c1].copy()
        del data

    overrides = {}
    if args.guard is not None:
        overrides["guard_hz"] = args.guard
    if args.no_deweight:
        overrides["deweight"] = False
    cfg = SubLookConfig.from_info(info, **overrides)
    (lo_a, hi_a), (lo_b, hi_b) = band_edges(cfg)
    print(f"burst {k}: block {blocks['vv'].shape} (lines x samples), dt {info.azimuth_time_interval * 1e3:.4f} ms")
    print(f"look A band [{lo_a:.1f}, {hi_a:.1f}] Hz, look B band [{lo_b:.1f}, {hi_b:.1f}] Hz, "
          f"deweight={cfg.deweight}, decimate={cfg.decimate}")

    sl = split_sublooks(blocks["vv"], blocks["vh"], info.azimuth_time_interval, cfg)
    print(f"output looks: {sl.a_vv.shape}, azimuth step {sl.dt_out * 1e3:.3f} ms "
          f"(~{info.azimuth_pixel_spacing * sl.decimation:.1f} m), using {sl.n_lines_used} input lines")

    n = sl.n_lines_used
    for pol, a, b_ in (("VV", sl.a_vv, sl.b_vv), ("VH", sl.a_vh, sl.b_vh)):
        full = np.mean(np.abs(blocks[pol.lower()][:n]) ** 2)
        pa, pb = np.mean(np.abs(a) ** 2), np.mean(np.abs(b_) ** 2)
        rho = complex_correlation(a, b_)
        r_int = speckle_intensity_correlation(a, b_)
        print(f"{pol}: mean power full {_db(full):.2f} dB, A {_db(pa):.2f} dB, B {_db(pb):.2f} dB, "
              f"A/B {_db(pa / pb):+.2f} dB | |complex corr| {abs(rho):.3f}, speckle intensity corr {r_int:+.3f}")
        print(f"    mean of the two looks vs full product: {_db((pa + pb) / 2 / full):+.2f} dB")

    # shape of the noise spectrum on the darkest VH columns (Hamming-shaped: about -5.7 dB, white: about 0 dB)
    f, pw = dark_column_spectrum(blocks["vh"], info.azimuth_time_interval)
    half = cfg.bandwidth_hz / 2.0
    model = hamming_weight(f, cfg.bandwidth_hz, cfg.window_coefficient) ** 2
    print(f"\ndarkest 10% of VH columns: edge/centre {edge_centre_db(f, pw, cfg.bandwidth_hz):+.2f} dB "
          f"(signal-shaped model {edge_centre_db(f, np.maximum(model, 1e-12), cfg.bandwidth_hz):+.2f} dB, white 0 dB)")
    for label in ("window", "white"):
        na, nb = noise_factors(sl.n_lines_used, info.azimuth_time_interval, cfg, label)
        print(f"noise power in looks if noise is {label:6s}: A x{na:.2f} ({_db(na):+.2f} dB), "
              f"B x{nb:.2f} ({_db(nb):+.2f} dB) relative to the noise LUT")


if __name__ == "__main__":
    main()