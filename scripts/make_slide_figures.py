"""Figures and numbers for the presentation slides, with the exact file names used in slides.tex.

    python scripts/make_slide_figures.py --complex ~/runs/sim_complex/best.pt --real ~/runs/sim_real/best.pt \
        --runs ~/runs/sim_complex ~/runs/sim_real --out-dir docs/simulation/slides

Writes truth_scene.png, showcase_noisy.png, showcase_denoised.png, coherence_maps.png, point_scatterers.png,
gamma_bias.png, loss_curve.png, truth_check.png and slide_numbers.txt (every number the slides ask for).
Uses the same seeded showcase / test scenes as sar_denoise.sim_eval, so the numbers match metrics.md.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from sar_denoise.sim_eval import (BOXCAR, COLORS, GRID, INK, INK2, LABELS, _gamma_cmap, _style, composite,
                                  composite_limits, estimates, fig_gamma_bias, run_test_set, scene_metrics,
                                  showcase_scene, test_scenes)
from sar_denoise.sim_metrics import db, gamma_of, point_metrics
from sar_denoise.sim_scene import SceneConfig, check_truth, homogeneous_mask, make_scene
from sar_denoise.sim_train import load_model, pick_device

TRUTH_CHECK_LOOKS = 1000


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def fig_pair(c2, lims, title, path, hom=None, note=None):
    """One estimate: intensity composite | |gamma|, side by side."""
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.4, 3.4), gridspec_kw={"wspace": 0.05})
    a0.imshow(composite(c2, lims), interpolation="nearest")
    a1.imshow(np.abs(gamma_of(c2)), cmap=_gamma_cmap(), vmin=0, vmax=1, interpolation="nearest")
    if hom is not None:
        a0.contour(hom, levels=[0.5], colors=["#ffffff"], linewidths=1.2, linestyles="--")
    a0.set_title("RGB: VV | VH | VV/VH", fontsize=9, color=INK2)
    a1.set_title("|gamma|", fontsize=9, color=INK2)
    if note:
        a1.text(0.5, 0.5, note, transform=a1.transAxes, ha="center", va="center", fontsize=9, color=INK,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    for a in (a0, a1):
        _clean(a)
    if title:
        fig.suptitle(title, fontsize=11, color=INK, y=0.98)
    fig.savefig(path)
    plt.close(fig)


def fig_truth_scene(scene, lims, path):
    truth = scene.truth_c2(np.float64)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(3.4, 6.2), gridspec_kw={"hspace": 0.14})
    a0.imshow(composite(truth, lims), interpolation="nearest")
    a0.contour(homogeneous_mask(scene), levels=[0.5], colors=["#ffffff"], linewidths=1.2, linestyles="--")
    im = a1.imshow(np.abs(scene.gamma), cmap=_gamma_cmap(), vmin=0, vmax=1, interpolation="nearest")
    a0.set_title("Truth: RGB VV | VH | VV/VH (dB)", fontsize=9, color=INK)
    a1.set_title("Truth: |gamma|", fontsize=9, color=INK)
    for a in (a0, a1):
        _clean(a)
    cb = fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def phase_image(g):
    """Phase of gamma on a cyclic map, faded to white where |gamma| is small (phase is meaningless there)."""
    rgb = plt.get_cmap("twilight")((np.angle(g) + np.pi) / (2 * np.pi))[..., :3]
    a = np.clip(np.abs(g), 0, 1)[..., None]
    return a * rgb + (1 - a) * 1.0


def fig_coherence_maps(scene, est, path):
    cols = [("Truth", scene.truth_c2(np.float64)), (LABELS["boxcar"], est["boxcar"]), (LABELS["complex"], est["complex"])]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.4), gridspec_kw={"hspace": 0.1, "wspace": 0.04})
    for j, (name, c2) in enumerate(cols):
        g = gamma_of(c2)
        axes[0, j].imshow(np.abs(g), cmap=_gamma_cmap(), vmin=0, vmax=1, interpolation="nearest")
        axes[1, j].imshow(phase_image(g), interpolation="nearest")
        axes[0, j].set_title(name, color=INK)
        for a in axes[:, j]:
            _clean(a)
    axes[0, 0].set_ylabel("|gamma|", color=INK2)
    axes[1, 0].set_ylabel("arg gamma (faded by |gamma|)", color=INK2)
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), _gamma_cmap()), ax=axes[0, :].tolist(), fraction=0.02, pad=0.01)
    cb.set_label("|gamma|", color=INK2)
    cb.outline.set_visible(False)
    cb2 = fig.colorbar(ScalarMappable(Normalize(-180, 180), plt.get_cmap("twilight")), ax=axes[1, :].tolist(),
                       fraction=0.02, pad=0.01, ticks=[-180, -90, 0, 90, 180])
    cb2.set_label("phase (deg)", color=INK2)
    cb2.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def zoom_point(scene):
    """Same point as sim_eval.fig_point_zoom: highest truth peak-to-background ratio."""
    truth = scene.truth_c2(np.float64)
    pm = point_metrics(truth[0], truth[0], scene.points)
    return int(np.argmax([p["true_peak_db"] - p["bg_db"] for p in pm]))


def fig_point_scatterers(scene, est, path, half=16):
    truth = scene.truth_c2(np.float64)
    k = zoom_point(scene)
    r, c = scene.points[k]
    bg = point_metrics(truth[0], truth[0], [(r, c)])[0]
    vmin, vmax = bg["bg_db"] - 8, bg["true_peak_db"]
    sl = (slice(r - half, r + half + 1), slice(c - half, c + half + 1))
    cols = [("Noisy (single look)", est["noisy"][0]), ("Despeckled (complex U-Net)", est["complex"][0])]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.7), gridspec_kw={"wspace": 0.06})
    rows = []
    for ax, (name, img) in zip(axes, cols):
        p = point_metrics(img, truth[0], [(r, c)])[0]
        rows.append(p)
        im = ax.imshow(db(img[sl]), cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        for rad in (2, 8):
            ax.add_patch(Circle((half, half), rad, fill=False, ec="#eb6834", lw=1.0, ls="--"))
        ax.set_title(name, color=INK, fontsize=10)
        ax.set_xlabel(f"peak {p['peak_db']:+.1f} dB (truth {p['true_peak_db']:+.1f})\n"
                      f"bright px in ring: {p['bright_px']} / {p['ring_px']}", color=INK, fontsize=9)
        _clean(ax)
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02)
    cb.set_label("VV intensity (dB)", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)
    return rows


def read_log(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def fig_loss_curve(runs: dict, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name, rows in runs.items():
        step = [r["step"] for r in rows]
        ax.plot(step, [r["val_kl"] for r in rows], "-o", color=COLORS[name], lw=1.6, ms=5, mec="white", mew=1,
                label=f"{LABELS[name]}: validation")
        tr = [(r["step"], r["train_kl"]) for r in rows if r["train_kl"] < 10]   # drops the off-scale first interval
        ax.plot(*zip(*tr), "--", color=COLORS[name], lw=1.3, label=f"{LABELS[name]}: training (mean of interval)")
    ax.set_yscale("log")
    ax.set(xlabel="training step (batch of 8 fresh 128x128 scenes)", ylabel="KL loss per pixel (log scale)",
           title="Supervised KL loss")
    ax.grid(True, which="both", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.text(0.5, -0.03, "Training value of the first interval (step 250) omitted where it is off-scale: an early "
             "spike of the real U-Net (KL 6.9e4), recovered by gradient clipping.", ha="center", fontsize=7.5,
             color=INK2)
    fig.savefig(path)
    plt.close(fig)


def fig_truth_check(rows, n_looks, path):
    f = [r for r in rows if r["field"] != "point"]
    p = [r for r in rows if r["field"] == "point"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"wspace": 0.32})
    a0.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    a0.plot([r["gamma_true"] for r in f], [r["gamma_est"] for r in f], "o", color=COLORS["boxcar"], ms=6,
            mec="white", mew=1, label=f"fields + roads ({len(f)})")
    a0.plot([r["gamma_true"] for r in p], [r["gamma_est"] for r in p], "D", color=COLORS["complex"], ms=6,
            mec="white", mew=1, label=f"point scatterers ({len(p)})")
    a0.set(xlim=(0, 1), ylim=(0, 1), xlabel="designed |gamma|", ylabel=f"recovered |gamma| (M = {n_looks})",
           title="Coherence magnitude")
    a0.legend(frameon=False, loc="upper left", fontsize=8.5)
    a1.axhline(0, color=INK2, lw=1, ls="--")
    a1.plot([r["gamma_true"] for r in f], [r["gamma_est"] - r["gamma_true"] for r in f], "o",
            color=COLORS["boxcar"], ms=6, mec="white", mew=1)
    a1.plot([r["gamma_true"] for r in p], [r["gamma_est"] - r["gamma_true"] for r in p], "D",
            color=COLORS["complex"], ms=6, mec="white", mew=1)
    a1.set(xlim=(0, 1), xlabel="designed |gamma|", ylabel="recovered - designed |gamma|", title="Error")
    lim = max(abs(r["gamma_est"] - r["gamma_true"]) for r in rows) * 1.3
    a1.set_ylim(-lim, lim)
    for a in (a0, a1):
        a.grid(True, color=GRID, lw=0.7)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def test_extras(models, dev, n, size):
    """Overall |gamma| RMSE (all non-point test pixels pooled) and phase error in rad, per method."""
    se, cnt, ph = {}, 0, {}
    for scene, (s_vv, s_vh) in test_scenes(n, size):
        est = estimates(scene, s_vv, s_vh, models, dev)
        m = scene_metrics(scene, est)
        sel = np.abs(scene.a_vv) == 0
        gt = np.abs(scene.gamma[sel])
        cnt += sel.sum()
        for name, c2 in est.items():
            se[name] = se.get(name, 0.0) + float(np.sum((np.abs(gamma_of(c2)[sel]) - gt) ** 2))
            ph.setdefault(name, []).append(m[name]["phase_err_deg"])
    return {k: {"gamma_rmse": math.sqrt(se[k] / cnt), "phase_err_rad": math.radians(float(np.nanmean(ph[k])))}
            for k in se}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--complex", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--runs", nargs=2, required=True, metavar=("COMPLEX_RUN", "REAL_RUN"),
                    help="run directories with log.csv and config.json")
    ap.add_argument("--out-dir", default="docs/simulation/slides")
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    out = Path(a.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    _style(plt)
    dev = pick_device(a.device)
    models, cks = {}, {}
    for kind, path in (("complex", a.complex), ("real", a.real)):
        models[kind], cks[kind] = load_model(Path(path).expanduser(), dev)

    scene, (s_vv, s_vh) = showcase_scene(a.size)
    est = estimates(scene, s_vv, s_vh, models, dev)
    show = scene_metrics(scene, est)
    lims = composite_limits(scene.truth_c2(np.float64))
    hom = homogeneous_mask(scene)
    fig_truth_scene(scene, lims, out / "truth_scene.png")
    fig_pair(est["noisy"], lims, None, out / "showcase_noisy.png", hom, note="single look:\n|gamma| = 1")
    fig_pair(est["complex"], lims, None, out / "showcase_denoised.png", hom)
    fig_coherence_maps(scene, est, out / "coherence_maps.png")
    pts = fig_point_scatterers(scene, est, out / "point_scatterers.png")

    test = run_test_set(models, dev, a.n_test, a.size)
    fig_gamma_bias(test, ["boxcar", "complex", "real"], out / "gamma_bias.png")
    extra = test_extras(models, dev, a.n_test, a.size)

    logs = {"complex": read_log(Path(a.runs[0]).expanduser() / "log.csv"),
            "real": read_log(Path(a.runs[1]).expanduser() / "log.csv")}
    fig_loss_curve(logs, out / "loss_curve.png")

    rng = np.random.default_rng(0)
    tscene = make_scene(SceneConfig(shape=(256, 256)), rng)
    trows = check_truth(tscene, n_looks=TRUTH_CHECK_LOOKS, rng=rng)
    fig_truth_check(trows, TRUTH_CHECK_LOOKS, out / "truth_check.png")

    # ------------------------------------------------------------------ numbers for the slides
    L = []
    w = L.append
    w("SHOWCASE slide (showcase scene, 256x256)")
    for n in ("noisy", "complex"):
        s = show[n]
        w(f"  {LABELS[n]:28s} ENL = {s['enl']:.1f}   PSNR = {s['psnr_db']:.1f} dB   CSSIM = {s['cssim']:.2f}")
    w("")
    w("POINT SCATTERERS slide (zoomed point | test-set average over all points)")
    tp = {n: [] for n in ("noisy", "complex")}
    tb = {n: test[n]["point_bright_px"] for n in tp}
    for n in tp:
        tp[n] = 10 ** (test[n]["point_peak_err_db"] / 10)
    w(f"  peak ratio I_hat/I : noisy {10 ** ((pts[0]['peak_db'] - pts[0]['true_peak_db']) / 10):.2f} | "
      f"despeckled {10 ** ((pts[1]['peak_db'] - pts[1]['true_peak_db']) / 10):.2f}    "
      f"(test, from mean dB error: noisy {tp['noisy']:.2f} | despeckled {tp['complex']:.2f})")
    w(f"  bright px in ring  : noisy {pts[0]['bright_px']} / {pts[0]['ring_px']} | despeckled "
      f"{pts[1]['bright_px']} / {pts[1]['ring_px']}    (test mean per point: noisy {tb['noisy']:.1f} | "
      f"despeckled {tb['complex']:.1f})")
    w("")
    w(f"QUANTITATIVE table ({a.n_test} test scenes of {a.size}x{a.size})")
    w("  method                      ENL     PSNR(dB)  CSSIM   |g| RMSE  phase err (rad)")
    for n in ("noisy", "boxcar", "real", "complex"):
        t = test[n]
        w(f"  {LABELS[n]:26s} {t['enl']:7.1f}  {t['psnr_db']:7.1f}  {t['cssim']:6.3f}  {extra[n]['gamma_rmse']:7.3f}  "
          f"{extra[n]['phase_err_rad']:7.3f}")
    w("  (phase error: median |arg(g_hat g*)| where true |g| > 0.3, mean over scenes; |g| RMSE over all non-point pixels)")
    w("")
    f = [r for r in trows if r["field"] != "point"]
    p = [r for r in trows if r["field"] == "point"]
    w(f"TRUTH CHECK slide: M = {TRUTH_CHECK_LOOKS}, {len(f)} fields (incl. roads) + {len(p)} point scatterers; max ||g| error| "
      f"{max(abs(r['gamma_est'] - r['gamma_true']) for r in f):.4f} (fields), "
      f"{max(abs(r['gamma_est'] - r['gamma_true']) for r in p):.4f} (points); max phase error (|g|>0.2) "
      f"{max(abs(r['phase_err_deg']) for r in f if r['gamma_true'] > 0.2):.2f} deg")
    w("")
    w("TRAINING table")
    for n in ("complex", "real"):
        cfgs, ck, lg = cks[n], cks[n], logs[n]
        w(f"  {LABELS[n]:24s} depth {ck['args']['depth']}, base {ck['args']['base']}"
          f"{' complex' if n == 'complex' else ''} channels, "
          f"{sum(v.numel() for v in models[n].parameters()):,} parameters, {int(lg[-1]['step'])} steps, "
          f"{lg[-1]['minutes']:.1f} min, best val KL {ck['val']['kl']:.4f} (step {ck['step']})")
    w("  patch 128x128, batch 8, AdamW lr 1e-3 (100-step warm-up, cosine decay), weight decay 1e-4, grad clip 1.0, "
      "Apple M3 (MPS)")
    w(f"  boxcar window {BOXCAR}x{BOXCAR}")
    text = "\n".join(L) + "\n"
    (out / "slide_numbers.txt").write_text(text)
    print(text)
    print(f"figures in {out.resolve()}")


if __name__ == "__main__":
    main()
