"""Parse Sentinel-1 IW SLC (.SAFE) products and read single bursts.

Usage (from the repo root, venv active, after `pip install -e .`):
    python -m sar_denoise.safe_io /path/to/product.SAFE iw2 vv

Tag names follow the Sentinel-1 product annotation layout. Anything that cannot
be found is stored as None and reported by summary(), so a mismatch is visible
instead of silent.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lxml import etree


@dataclass
class Burst:
    index: int
    line_start: int            # first line of this burst inside the TIFF
    n_lines: int
    n_samples: int
    azimuth_time: np.datetime64
    first_valid: np.ndarray    # per line, -1 where the line is invalid
    last_valid: np.ndarray

    @property
    def valid_lines(self) -> np.ndarray:
        return (self.first_valid >= 0) & (self.last_valid >= 0)

    @property
    def valid_line_range(self) -> tuple[int, int]:
        idx = np.flatnonzero(self.valid_lines)
        return int(idx[0]), int(idx[-1]) + 1

    @property
    def valid_sample_range(self) -> tuple[int, int]:
        """Common valid sample interval [start, stop) over all valid lines."""
        ok = self.valid_lines
        return int(self.first_valid[ok].max()), int(self.last_valid[ok].min()) + 1


@dataclass
class TimedPoly:
    azimuth_time: np.datetime64
    t0: float
    coeffs: np.ndarray


@dataclass
class SwathInfo:
    swath: str
    polarisation: str
    n_lines: int
    n_samples: int
    lines_per_burst: int
    samples_per_burst: int
    azimuth_time_interval: float | None      # s
    range_pixel_spacing: float | None        # m
    azimuth_pixel_spacing: float | None      # m
    radar_frequency: float | None            # Hz
    azimuth_steering_rate: float | None      # deg/s
    processing_bandwidth: float | None       # Hz (azimuth)
    window_type: str | None
    window_coefficient: float | None
    bursts: list[Burst] = field(default_factory=list)
    fm_rates: list[TimedPoly] = field(default_factory=list)
    dc_estimates: list[TimedPoly] = field(default_factory=list)
    slant_range_time: float | None = None        # s, two-way time of sample 0
    range_sampling_rate: float | None = None     # Hz
    orbit_times: np.ndarray = field(default_factory=lambda: np.array([], dtype='datetime64[us]'))
    orbit_speeds: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))  # m/s

    def summary(self) -> str:
        rows = [
            f"swath / pol            : {self.swath} / {self.polarisation}",
            f"image (lines x samples): {self.n_lines} x {self.n_samples}",
            f"bursts                 : {len(self.bursts)} "
            f"({self.lines_per_burst} lines x {self.samples_per_burst} samples each)",
            f"azimuth time interval  : {self.azimuth_time_interval} s",
            f"pixel spacing (rg, az) : {self.range_pixel_spacing}, {self.azimuth_pixel_spacing} m",
            f"radar frequency        : {self.radar_frequency} Hz",
            f"azimuth steering rate  : {self.azimuth_steering_rate} deg/s",
            f"processing bandwidth   : {self.processing_bandwidth} Hz",
            f"azimuth window         : {self.window_type} (coefficient {self.window_coefficient})",
            f"FM-rate records        : {len(self.fm_rates)}",
            f"Doppler-centroid recs  : {len(self.dc_estimates)}",
            f"slant range time (s0) : {self.slant_range_time} s",
            f"range sampling rate   : {self.range_sampling_rate} Hz",
            f"orbit state vectors   : {len(self.orbit_times)}",
        ]
        return "\n".join(rows)


def _get(root, path: str, cast=str):
    node = root.find(path)
    if node is None or node.text is None or not node.text.strip():
        return None
    return cast(node.text.strip())


def _poly_list(root, list_path: str, poly_tags: tuple[str, ...]) -> list[TimedPoly]:
    out = []
    for node in root.findall(list_path):
        coeffs = None
        for tag in poly_tags:
            text = node.findtext(tag)
            if text and text.strip():
                coeffs = np.array(text.split(), dtype=float)
                break
        if coeffs is None:  # older layout: c0, c1, c2 as separate tags
            parts = [node.findtext(c) for c in ("c0", "c1", "c2")]
            if all(parts):
                coeffs = np.array(parts, dtype=float)
        if coeffs is None:
            continue
        out.append(TimedPoly(
            azimuth_time=np.datetime64(node.findtext("azimuthTime").strip()),
            t0=float(node.findtext("t0")),
            coeffs=coeffs,
        ))
    return out


def find_swath_files(safe_dir, swath: str, pol: str) -> tuple[Path, Path]:
    """Return (measurement tiff, annotation xml) for one subswath and polarisation."""
    safe_dir = Path(safe_dir).expanduser()
    swath, pol = swath.lower(), pol.lower()
    tiffs = sorted((safe_dir / "measurement").glob(f"s1?-{swath}-slc-{pol}-*.tiff"))
    xmls = sorted((safe_dir / "annotation").glob(f"s1?-{swath}-slc-{pol}-*.xml"))
    if len(tiffs) != 1 or len(xmls) != 1:
        raise FileNotFoundError(
            f"Expected exactly one tiff and one xml for {swath}/{pol} in {safe_dir}, "
            f"found {len(tiffs)} tiff and {len(xmls)} xml."
        )
    return tiffs[0], xmls[0]


def parse_annotation(xml_path) -> SwathInfo:
    root = etree.parse(str(xml_path)).getroot()

    n_lines = _get(root, "imageAnnotation/imageInformation/numberOfLines", int)
    n_samples = _get(root, "imageAnnotation/imageInformation/numberOfSamples", int)
    lines_per_burst = _get(root, "swathTiming/linesPerBurst", int)
    samples_per_burst = _get(root, "swathTiming/samplesPerBurst", int)

    bursts = []
    for i, b in enumerate(root.findall("swathTiming/burstList/burst")):
        bursts.append(Burst(
            index=i,
            line_start=i * lines_per_burst,
            n_lines=lines_per_burst,
            n_samples=samples_per_burst,
            azimuth_time=np.datetime64(b.findtext("azimuthTime").strip()),
            first_valid=np.array(b.findtext("firstValidSample").split(), dtype=int),
            last_valid=np.array(b.findtext("lastValidSample").split(), dtype=int),
        ))

    orbit = []
    for o in root.findall("generalAnnotation/orbitList/orbit"):
        v = o.find("velocity")
        vx, vy, vz = (float(v.findtext(k)) for k in ("x", "y", "z"))
        orbit.append((np.datetime64(o.findtext("time").strip()), float(np.sqrt(vx**2 + vy**2 + vz**2))))
    orbit.sort(key=lambda t: t[0])

    return SwathInfo(
        swath=_get(root, "adsHeader/swath"),
        polarisation=_get(root, "adsHeader/polarisation"),
        n_lines=n_lines,
        n_samples=n_samples,
        lines_per_burst=lines_per_burst,
        samples_per_burst=samples_per_burst,
        azimuth_time_interval=_get(root, "imageAnnotation/imageInformation/azimuthTimeInterval", float),
        range_pixel_spacing=_get(root, "imageAnnotation/imageInformation/rangePixelSpacing", float),
        azimuth_pixel_spacing=_get(root, "imageAnnotation/imageInformation/azimuthPixelSpacing", float),
        radar_frequency=_get(root, "generalAnnotation/productInformation/radarFrequency", float),
        azimuth_steering_rate=_get(root, "generalAnnotation/productInformation/azimuthSteeringRate", float),
        processing_bandwidth=_get(root, ".//azimuthProcessing/processingBandwidth", float),
        window_type=_get(root, ".//azimuthProcessing/windowType"),
        window_coefficient=_get(root, ".//azimuthProcessing/windowCoefficient", float),
        bursts=bursts,
        fm_rates=_poly_list(root, "generalAnnotation/azimuthFmRateList/azimuthFmRate",
                            ("azimuthFmRatePolynomial",)),
        dc_estimates=_poly_list(root, "dopplerCentroid/dcEstimateList/dcEstimate",
                                ("dataDcPolynomial",)),
        slant_range_time=_get(root, "imageAnnotation/imageInformation/slantRangeTime", float),
        range_sampling_rate=_get(root, "generalAnnotation/productInformation/rangeSamplingRate", float),
        orbit_times=np.array([t for t, _ in orbit], dtype="datetime64[us]"),
        orbit_speeds=np.array([v for _, v in orbit], dtype=float),
    )


def read_burst(tiff_path, info: SwathInfo, burst_index: int) -> np.ndarray:
    """Read one full burst as complex64, shape (lines_per_burst, samples_per_burst)."""
    import rasterio
    from rasterio.windows import Window

    b = info.bursts[burst_index]
    window = Window(0, b.line_start, b.n_samples, b.n_lines)
    with rasterio.open(tiff_path) as src:
        if src.width != info.n_samples or src.height != info.n_lines:
            raise ValueError(
                f"TIFF size {src.height}x{src.width} does not match annotation "
                f"{info.n_lines}x{info.n_samples}"
            )
        data = src.read(1, window=window)
    return data.astype(np.complex64)


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect one subswath of an S1 IW SLC product")
    p.add_argument("safe_dir")
    p.add_argument("swath", help="iw1, iw2 or iw3")
    p.add_argument("pol", help="vv or vh")
    p.add_argument("--burst", type=int, default=None, help="burst index to read (default: middle)")
    args = p.parse_args()

    tiff, xml = find_swath_files(args.safe_dir, args.swath, args.pol)
    info = parse_annotation(xml)
    print(info.summary())

    k = args.burst if args.burst is not None else len(info.bursts) // 2
    b = info.bursts[k]
    print(f"\nburst {k}: valid lines {b.valid_line_range}, valid samples {b.valid_sample_range}")

    data = read_burst(tiff, info, k)
    power = np.abs(data) ** 2
    print(f"read shape {data.shape}, dtype {data.dtype}")
    print(f"mean I {data.real.mean():.2f}, mean Q {data.imag.mean():.2f}")
    print(f"mean power {power.mean():.3e}, fraction exactly zero {np.mean(data == 0):.3f}")


if __name__ == "__main__":
    main()