"""Cut the training regions out of a Sentinel-1 SLC product and index their patches.

For every region in the config (plus optional randomly chosen incoherent "negative" tiles from the
quick-look scan) this runs the full preprocessing chain on the burst, crops the region, saves it as
<out-dir>/<region>.npz, and writes <out-dir>/index.csv with one row per usable patch.

Checks before any heavy work: regions in different splits must be farther apart than the buffer, measured
in azimuth time and range sample (so burst overlaps are caught). Use --dry-run to see the plan only.

Usage (from the repo root, venv active):
    python scripts/make_patches.py "$SAFE" --dry-run
    python scripts/make_patches.py "$SAFE" --out-dir "$HOME/data/sar/regions"
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from sar_denoise.patches import (
    PatchConfig,
    assign_splits,
    build_index,
    conflicts_with,
    crop_products,
    find_conflicts,
    load_region,
    pick_negatives,
    regions_from_config,
    region_time_span,
    save_region,
    summarise_index,
    write_index_csv,
)
from sar_denoise.preprocess import process_burst
from sar_denoise.safe_io import find_swath_files, parse_annotation


def read_tiles(tiles_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(tiles_dir.glob("*_tiles.csv")):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def describe(regions, infos) -> str:
    lines = ["name                        split kind      swath burst  lines (TIFF)        samples            time span [s]"]
    for r in regions:
        t0, t1 = region_time_span(infos[r.swath], r.burst, r.lines)
        lines.append(f"{r.name:27s} {r.split:5s} {r.kind:9s} {r.swath:5s} {r.burst:5d}  "
                     f"{r.lines[0]:6d}-{r.lines[1]:<6d}   {r.samples[0]:6d}-{r.samples[1]:<6d}   {t0:7.2f}-{t1:.2f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cut training regions and index their patches")
    ap.add_argument("safe_dir")
    ap.add_argument("--config", default="configs/regions.yaml")
    ap.add_argument("--out-dir", default="~/data/sar/regions")
    ap.add_argument("--tiles-dir", default="~/data/sar/quicklook", help="folder with the quick-look *_tiles.csv files")
    ap.add_argument("--negatives", type=int, default=None, help="number of incoherent tiles to add (default: config)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the split checks, then stop")
    ap.add_argument("--overwrite", action="store_true", help="recompute regions whose archive already exists")
    ap.add_argument("--allow-conflicts", action="store_true", help="continue although regions of different splits are too close")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pcfg = PatchConfig.from_dict(cfg.get("patch"))
    buf = cfg.get("buffer", {})
    buf_lines, buf_samples = int(buf.get("lines", 128)), int(buf.get("samples", 1024))
    trim_rows = int(cfg.get("preprocess", {}).get("trim_rows", 32))
    margin = int(cfg.get("preprocess", {}).get("column_margin", 64))
    neg_cfg = cfg.get("negatives", {})
    n_neg = int(neg_cfg.get("count", 0)) if args.negatives is None else args.negatives

    regions = regions_from_config(cfg)
    swaths = {r.swath for r in regions} | set(neg_cfg.get("swaths", ["iw1", "iw2"]))
    infos = {sw: parse_annotation(find_swath_files(args.safe_dir, sw, "vv")[1]) for sw in sorted(swaths)}

    bad = find_conflicts(regions, infos, buf_lines, buf_samples)
    for a, b in bad:
        print(f"CONFLICT: {a.name} ({a.split}) is within the buffer of {b.name} ({b.split})")
    if bad and not args.allow_conflicts:
        print("Fix regions.yaml (or pass --allow-conflicts).")
        return 1

    rng = np.random.default_rng(args.seed)
    if n_neg > 0:
        tiles = read_tiles(Path(args.tiles_dir).expanduser())
        if not tiles:
            print(f"no *_tiles.csv in {args.tiles_dir}; skipping negatives")
        else:
            free = lambda reg, chosen: not conflicts_with(reg, regions + chosen, infos, buf_lines, buf_samples,
                                                          ignore_same_split=False)
            negs = pick_negatives(tiles, n_neg, rng, free, swaths=tuple(neg_cfg.get("swaths", ["iw1", "iw2"])),
                                  max_gamma_p90=float(neg_cfg.get("max_gamma_p90", 0.12)))
            negs = assign_splits(negs, tuple(neg_cfg.get("fractions", [0.7, 0.15, 0.15])), rng)
            print(f"picked {len(negs)} incoherent tiles")
            regions = regions + negs

    names = [r.name for r in regions]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        print(f"ERROR: duplicate region names: {dup}")
        return 1
    bad = find_conflicts(regions, infos, buf_lines, buf_samples)
    for a, b in bad:
        print(f"CONFLICT: {a.name} ({a.split}) is within the buffer of {b.name} ({b.split})")
    if bad and not args.allow_conflicts:
        return 1

    print(describe(regions, infos))
    if args.dry_run:
        return 0

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for i, reg in enumerate(regions, 1):
        path = out_dir / f"{reg.name}.npz"
        t0 = time.time()
        if path.exists() and not args.overwrite:
            arrays, meta = load_region(path)
            how = "loaded"
        else:
            cols = (reg.samples[0] - margin, reg.samples[1] + margin)
            try:
                prod = process_burst(args.safe_dir, reg.swath, reg.burst, cols=cols, trim_rows=trim_rows)
                arrays, meta = crop_products(prod, reg, pcfg.ml_window)
            except ValueError as e:
                print(f"[{i}/{len(regions)}] {reg.name}: skipped ({e})")
                continue
            save_region(path, arrays, meta)
            how = "computed"
        rows = build_index(arrays, reg, pcfg)
        index += rows
        print(f"[{i}/{len(regions)}] {reg.name}: {how} in {time.time() - t0:.0f}s, "
              f"{arrays['c2_a'].shape[1]} x {arrays['c2_a'].shape[2]} looks, {len(rows)} patches")

    if not index:
        print("no patches produced")
        return 1
    write_index_csv(index, out_dir / "index.csv")
    print(f"\nwrote {out_dir / 'index.csv'} ({len(index)} patches of {pcfg.rows} x {pcfg.cols}, "
          f"stride {pcfg.stride_rows} x {pcfg.stride_cols}; overlapping patches are NOT independent)\n")
    print(summarise_index(index))
    return 0


if __name__ == "__main__":
    sys.exit(main())