"""Slide check: averaging many independent speckle realisations of ONE synthetic scene recovers the designed coherence.

    python scripts/check_sim_truth.py [--looks 1000] [--size 256] [--seed 0]

Per field (pooled over the field's pixels and all looks) and per point scatterer (single pixel, all looks):
designed |gamma| vs recovered |gamma| and the phase error. The known VH thermal noise is subtracted from C22.
"""
import argparse
import time

import numpy as np

from sar_denoise.sim_scene import SceneConfig, check_truth, make_scene


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--looks", type=int, default=1000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    scene = make_scene(SceneConfig(shape=(args.size, args.size)), rng)
    t0 = time.time()
    rows = check_truth(scene, n_looks=args.looks, rng=rng)
    print(f"scene {args.size}x{args.size}, {len(scene.points)} point scatterers, "
          f"VH NESZ {10 * np.log10(scene.n_vh):.1f} dB, {args.looks} independent looks ({time.time() - t0:.1f} s)")
    print(f"{'field':>6} {'n_px':>6} {'|g| true':>9} {'|g| est':>9} {'diff':>8} {'phase err':>10}")
    for r in rows:
        print(f"{str(r['field']):>6} {r['n_px']:6d} {r['gamma_true']:9.3f} {r['gamma_est']:9.3f} "
              f"{r['gamma_est'] - r['gamma_true']:+8.4f} {r['phase_err_deg']:+9.2f}d")
    f = [r for r in rows if r["field"] != "point"]
    p = [r for r in rows if r["field"] == "point"]
    coh = [r for r in f if r["gamma_true"] > 0.2]
    print(f"fields: max ||g| error| {max(abs(r['gamma_est'] - r['gamma_true']) for r in f):.4f}, "
          f"max |phase error| (|g|>0.2) {max(abs(r['phase_err_deg']) for r in coh):.2f} deg")
    if p:
        print(f"points: max ||g| error| {max(abs(r['gamma_est'] - r['gamma_true']) for r in p):.4f} "
              f"(single pixel, {args.looks} looks)")


if __name__ == "__main__":
    main()
