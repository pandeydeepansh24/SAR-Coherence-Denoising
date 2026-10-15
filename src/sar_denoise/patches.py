"""From preprocessed bursts to training regions and patches.

Why regions first, patches second
---------------------------------
Coherent land is scarce: the quick-look scan found only a few connected areas. Patches cut from
the same area are neighbours and share speckle-independent but structurally identical content, so a
random patch-level split would leak. The unit of the train / val / test split is therefore a REGION
(a rectangle of the TIFF), and patches are cut only inside one region.

A region is stored once as an archive of full-resolution look products (physical sigma0 units):

    c2_a, c2_b   (4, rows, cols)   Look A / Look B, channels [C11, C22, Re C12, Im C12]
    noise_a/b    (2, rows, cols)   thermal-noise power in each look [VV, VH]
    pnr_db       (2, rows, cols)   smoothed power / noise [dB]
    gamma        (rows, cols)      local |coherence| (9 x 65 boxcar on the mean of both looks)
    valid        (rows, cols)      bool

Patches are then cropped on the fly (random offsets are free augmentation) and normalised per patch
with `normalise_pair`, which keeps |gamma| and the phase of C12 exactly.

Leakage between splits
----------------------
Neighbouring bursts overlap on the ground, so the same place can appear at different TIFF lines in
two bursts. `find_conflicts` therefore compares regions in (azimuth TIME, range sample) space, not in
line numbers, and reports pairs of different splits closer than the buffer.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from sar_denoise.covariance import coherence, multilook

SPLITS = ("train", "val", "test")
CLASSES = ("coherent", "weak", "incoherent")


# --------------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class Region:
    name: str
    split: str                  # train / val / test
    swath: str                  # iw1 / iw2 / iw3
    burst: int
    lines: tuple[int, int]      # absolute TIFF lines [start, stop)
    samples: tuple[int, int]    # absolute range samples [start, stop)
    kind: str = "positive"      # positive = chosen for coherence, negative = sea / ships / dark areas
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Region":
        r = cls(name=str(d["name"]), split=str(d["split"]), swath=str(d["swath"]).lower(),
                burst=int(d["burst"]), lines=tuple(int(v) for v in d["lines"]),
                samples=tuple(int(v) for v in d["samples"]), kind=str(d.get("kind", "positive")),
                note=str(d.get("note", "")))
        if r.split not in SPLITS:
            raise ValueError(f"region {r.name}: split must be one of {SPLITS}, got {r.split!r}")
        for what, (a, b) in (("lines", r.lines), ("samples", r.samples)):
            if len(getattr(r, what)) != 2 or b <= a:
                raise ValueError(f"region {r.name}: {what} must be [start, stop) with stop > start")
        return r


@dataclass
class PatchConfig:
    rows: int = 64                      # look rows (azimuth, already decimated by 2)
    cols: int = 512                     # range samples
    stride_rows: int = 32
    stride_cols: int = 256
    min_valid: float = 0.99             # minimum fraction of valid pixels in a patch
    gamma_threshold: float = 0.3        # a pixel counts as coherent above this local |coherence|
    coherent_frac: float = 0.15         # patch is "coherent" if at least this fraction is above threshold
    weak_frac: float = 0.03             # ... "weak" between weak_frac and coherent_frac, else "incoherent"
    ml_window: tuple[int, int] = (9, 65)

    @classmethod
    def from_dict(cls, d: dict | None) -> "PatchConfig":
        d = dict(d or {})
        if "ml_window" in d:
            d["ml_window"] = tuple(d["ml_window"])
        return cls(**d)


def regions_from_config(cfg: dict) -> list[Region]:
    regions = [Region.from_dict(d) for d in cfg.get("regions", [])]
    names = [r.name for r in regions]
    if len(set(names)) != len(names):
        raise ValueError("region names must be unique")
    return regions


# --------------------------------------------------------------------------- split hygiene

def region_time_span(info, burst: int, lines) -> tuple[float, float]:
    """Azimuth time span (seconds, relative to the first burst) of absolute TIFF lines of one burst."""
    b = info.bursts[burst]
    t_burst = float((b.azimuth_time - info.bursts[0].azimuth_time) / np.timedelta64(1, "s"))
    dt = info.azimuth_time_interval
    return t_burst + (lines[0] - b.line_start) * dt, t_burst + (lines[1] - b.line_start) * dt


def _gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(a[0] - b[1], b[0] - a[1])


def conflicts_with(region: Region, others, infos: dict, buffer_lines: int = 128, buffer_samples: int = 1024,
                   ignore_same_split: bool = True) -> list[Region]:
    """Regions among `others` that lie closer than the buffer to `region` (same swath only).

    Closeness is measured in azimuth time and range sample, so burst overlaps are caught.
    Different swaths are not compared (their small range overlap is ignored).
    """
    info = infos[region.swath]
    pad_t = buffer_lines * info.azimuth_time_interval
    ta = region_time_span(info, region.burst, region.lines)
    out = []
    for o in others:
        if o is region or o.swath != region.swath:       # identity, NOT name: a duplicate of a tile must conflict
            continue
        if ignore_same_split and o.split == region.split:
            continue
        tb = region_time_span(infos[o.swath], o.burst, o.lines)
        if _gap(ta, tb) < pad_t and _gap(region.samples, o.samples) < buffer_samples:
            out.append(o)
    return out


def find_conflicts(regions, infos: dict, buffer_lines: int = 128, buffer_samples: int = 1024,
                   ignore_same_split: bool = True) -> list[tuple[Region, Region]]:
    pairs = []
    for i, a in enumerate(regions):
        for b in conflicts_with(a, regions[i + 1:], infos, buffer_lines, buffer_samples, ignore_same_split):
            pairs.append((a, b))
    return pairs


def assign_splits(regions, fractions=(0.7, 0.15, 0.15), rng=None) -> list[Region]:
    """Randomly give each region a split (used for negative tiles, which are interchangeable)."""
    rng = np.random.default_rng() if rng is None else rng
    p = np.asarray(fractions, float)
    p = p / p.sum()
    return [replace(r, split=str(rng.choice(SPLITS, p=p))) for r in regions]


NEGATIVE_STRATA = {
    "dark": lambda t: t["pnr_vh_db"] < 3.0,                       # VH close to the noise floor
    "mid": lambda t: 3.0 <= t["pnr_vh_db"] < 8.0,
    "clean": lambda t: t["pnr_vh_db"] >= 8.0,                     # VH well above the noise, but not coherent
    "bright": lambda t: t["sigma_vv_db"] >= -8.0,                 # bright targets: ships, rough land
}


def pick_negatives(tiles, n: int, rng, is_free, max_gamma_p90: float = 0.12,
                   swaths=("iw1", "iw2"), strata=NEGATIVE_STRATA) -> list[Region]:
    """Choose about n incoherent tiles from the quick-look tile table, spread over several strata.

    tiles    : rows with swath, burst, line, lines, sample, samples, gamma_p90, pnr_vh_db, sigma_vv_db
    is_free  : callable(region, chosen_so_far) -> bool, e.g. a buffer check against all other regions
    """
    pool = []
    for t in tiles:
        t = dict(t)
        for key in ("burst", "line", "lines", "sample", "samples"):
            t[key] = int(float(t[key]))
        for key in ("gamma_p90", "pnr_vh_db", "sigma_vv_db"):
            t[key] = float(t[key])
        if t["swath"] in swaths and t["gamma_p90"] <= max_gamma_p90:
            pool.append(t)

    per = math.ceil(n / len(strata))
    chosen: list[Region] = []
    for label, rule in strata.items():
        cand = [pool[i] for i in rng.permutation(len(pool)) if rule(pool[i])]
        taken = 0
        for t in cand:
            if taken >= per:
                break
            reg = Region(name=f"neg_{t['swath']}_b{t['burst']}_{t['line']}_{t['sample']}", split="train",
                         swath=t["swath"], burst=t["burst"], lines=(t["line"], t["line"] + t["lines"]),
                         samples=(t["sample"], t["sample"] + t["samples"]), kind="negative", note=label)
            if reg.name not in {c.name for c in chosen} and is_free(reg, chosen):
                chosen.append(reg)
                taken += 1
    return chosen[:n]


# --------------------------------------------------------------------------- cutting a region out of a burst

def local_coherence(c2_a: np.ndarray, c2_b: np.ndarray, window=(9, 65)) -> np.ndarray:
    """|coherence| of the boxcar-averaged mean of both looks (independent speckle, so more samples)."""
    gamma, _ = coherence(multilook(0.5 * (c2_a + c2_b), window))
    return gamma.astype(np.float32)


def window_to_slices(meta: dict, lines, samples, shape) -> tuple[slice, slice]:
    """Absolute TIFF lines / samples -> row and column slices of the look arrays of a BurstProducts.

    Look row j corresponds to input line  line_start + valid_lines[0] + j * decimation.
    """
    step = int(meta["decimation"])
    first_line = int(meta["line_start"]) + int(meta["valid_lines"][0])
    first_sample = int(meta["samples"][0])
    r0 = math.ceil((lines[0] - first_line) / step)
    r1 = math.ceil((lines[1] - first_line) / step)
    c0, c1 = samples[0] - first_sample, samples[1] - first_sample
    rows, cols = shape
    r0, r1, c0, c1 = max(r0, 0), min(r1, rows), max(c0, 0), min(c1, cols)
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"window lines {tuple(lines)}, samples {tuple(samples)} does not intersect the product "
                         f"(look rows {rows}, columns {cols}, first line {first_line}, first sample {first_sample})")
    return slice(r0, r1), slice(c0, c1)


def crop_products(prod, region: Region, window=(9, 65)) -> tuple[dict, dict]:
    """Cut a region out of a BurstProducts. Returns (arrays, meta); the coherence map is computed BEFORE
    cropping so the boxcar sees real neighbours at the region edges."""
    gamma = local_coherence(prod.c2_a, prod.c2_b, window)
    rs, cs = window_to_slices(prod.meta, region.lines, region.samples, prod.valid.shape)
    s3 = (slice(None), rs, cs)
    arrays = {
        "c2_a": prod.c2_a[s3].copy(), "c2_b": prod.c2_b[s3].copy(),
        "noise_a": prod.noise_a[s3].copy(), "noise_b": prod.noise_b[s3].copy(),
        "pnr_db": prod.pnr_db[s3].copy(), "gamma": gamma[rs, cs].copy(), "valid": prod.valid[rs, cs].copy(),
    }
    step = int(prod.meta["decimation"])
    first_line = int(prod.meta["line_start"]) + int(prod.meta["valid_lines"][0])
    first_sample = int(prod.meta["samples"][0])
    meta = dict(prod.meta)
    meta.update({
        "region": asdict(region),
        "region_lines_used": [first_line + rs.start * step, first_line + rs.stop * step],
        "region_samples_used": [first_sample + cs.start, first_sample + cs.stop],
        "region_row_offset": rs.start, "region_col_offset": cs.start,
    })
    return arrays, meta


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.datetime64, np.timedelta64)):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_region(path, arrays: dict, meta: dict) -> None:
    np.savez_compressed(path, **arrays, meta=json.dumps(meta, default=_json_default))


def load_region(path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    return arrays, meta


# --------------------------------------------------------------------------- patches

def patch_origins(shape, cfg: PatchConfig) -> list[tuple[int, int]]:
    h, w = shape
    return [(r, c) for r in range(0, h - cfg.rows + 1, cfg.stride_rows)
            for c in range(0, w - cfg.cols + 1, cfg.stride_cols)]


def classify(frac_coherent: float, cfg: PatchConfig) -> str:
    if frac_coherent >= cfg.coherent_frac:
        return "coherent"
    return "weak" if frac_coherent >= cfg.weak_frac else "incoherent"


def build_index(arrays: dict, region: Region, cfg: PatchConfig) -> list[dict]:
    """One row per usable patch of a region: where it is, how coherent, how far above the noise."""
    valid, gamma, pnr = arrays["valid"], arrays["gamma"], arrays["pnr_db"]
    rows = []
    for r, c in patch_origins(valid.shape, cfg):
        sl = (slice(r, r + cfg.rows), slice(c, c + cfg.cols))
        v = valid[sl]
        frac_valid = float(v.mean())
        if frac_valid < cfg.min_valid:
            continue
        g = gamma[sl][v]
        frac = float(np.mean(g > cfg.gamma_threshold))
        rows.append({
            "region": region.name, "split": region.split, "kind": region.kind, "row": r, "col": c,
            "frac_valid": round(frac_valid, 4), "frac_coherent": round(frac, 4),
            "gamma_median": round(float(np.median(g)), 4), "class": classify(frac, cfg),
            "pnr_vh_db": round(float(np.median(pnr[1][sl][v])), 2),
        })
    return rows


def normalise_pair(c2_a, c2_b, noise_a, noise_b, valid=None):
    """Joint per-patch normalisation that preserves |gamma| and the phase of C12 exactly.

    s11, s22 = mean C11 / C22 over the valid pixels of both looks. Then
        C11 / s11,  C22 / s22,  C12 / sqrt(s11 s22)     (Re and Im with the SAME factor)
        noise_VV / s11,  noise_VH / s22
    i.e. C -> D C D with D = diag(1/sqrt(s11), 1/sqrt(s22)). This keeps C positive semi-definite,
    leaves gamma = C12 / sqrt(C11 C22) unchanged, and puts the data and the noise on one scale, which
    the noise-aware likelihood needs. Look A and Look B share the factors. Nothing nonlinear is applied.
    """
    if valid is None:
        valid = np.ones(c2_a.shape[1:], dtype=bool)
    if not valid.any():
        raise ValueError("no valid pixels to normalise with")
    s11 = float(0.5 * (c2_a[0][valid].mean() + c2_b[0][valid].mean()))
    s22 = float(0.5 * (c2_a[1][valid].mean() + c2_b[1][valid].mean()))
    if not (np.isfinite(s11) and np.isfinite(s22) and s11 > 0 and s22 > 0):
        raise ValueError(f"invalid normalisation scales s11={s11}, s22={s22}")
    s12 = math.sqrt(s11 * s22)
    d = np.array([1 / s11, 1 / s22, 1 / s12, 1 / s12], dtype=np.float32)[:, None, None]
    n = np.array([1 / s11, 1 / s22], dtype=np.float32)[:, None, None]
    return c2_a * d, c2_b * d, noise_a * n, noise_b * n, (s11, s22)


def extract_patch(arrays: dict, r: int, c: int, cfg: PatchConfig, normalise: bool = True) -> dict:
    sl2 = (slice(r, r + cfg.rows), slice(c, c + cfg.cols))
    sl3 = (slice(None),) + sl2
    out = {"valid": arrays["valid"][sl2], "gamma": arrays["gamma"][sl2], "pnr_db": arrays["pnr_db"][sl3]}
    parts = (arrays["c2_a"][sl3], arrays["c2_b"][sl3], arrays["noise_a"][sl3], arrays["noise_b"][sl3])
    if normalise:
        *parts, scales = normalise_pair(*parts, valid=out["valid"])
        out["scales"] = scales
    out.update(dict(zip(("c2_a", "c2_b", "noise_a", "noise_b"), parts)))
    return out


def write_index_csv(rows: list[dict], path) -> None:
    if not rows:
        raise ValueError("empty patch index")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarise_index(rows: list[dict]) -> str:
    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        counts[(r["split"], r["class"])] = counts.get((r["split"], r["class"]), 0) + 1
    lines = ["split   " + "  ".join(f"{c:>11s}" for c in CLASSES) + "        total"]
    for s in SPLITS:
        vals = [counts.get((s, c), 0) for c in CLASSES]
        lines.append(f"{s:6s}  " + "  ".join(f"{v:11d}" for v in vals) + f"  {sum(vals):11d}")
    return "\n".join(lines)