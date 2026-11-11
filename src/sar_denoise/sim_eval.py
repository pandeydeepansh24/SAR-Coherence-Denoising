"""Evaluate the simulation experiment and draw the presentation figures.

    python -m sar_denoise.sim_eval --complex ~/runs/sim_complex/best.pt --real ~/runs/sim_real/best.pt \
        --out-dir ~/runs/sim_figures

Methods: noisy single look | 5x5 boxcar | complex U-Net | real U-Net (either network may be omitted).
Data: one fixed showcase scene (256 x 256, land-like, with point scatterers) and N seeded test scenes, all generated
with seeds that training never uses. Networks see the whole image (fully convolutional), per-image normalised exactly
as in training.

Writes:
  showcase_main.png   truth | noisy | complex U-Net: intensity composite and |gamma|, ENL / PSNR / complex SSIM under each
  showcase_all.png    the same for all methods
  point_zoom.png      one point scatterer, VV dB, all methods, with peak intensity and bright-pixel count in the 2-8 px ring
  gamma_bias.png      estimated vs true |gamma| (and RMSE) over the test scenes
  metrics.json, metrics.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .covariance import form_c2, multilook
from .models import covariance_stack
from .sim_metrics import (complex_ssim, db, enl, gamma_bins, gamma_of, phase_error_deg, point_metrics, psnr_db)
from .sim_scene import SceneConfig, homogeneous_mask, make_scene, observe
from .sim_train import load_model, model_input, normalise_observation, pick_device

SHOWCASE_SEED = 30_000
TEST_SEED = 20_000
BOXCAR = 5
MIN_ENL_PX = 200
LABELS = {"noisy": "Noisy (single look)", "boxcar": f"{BOXCAR}x{BOXCAR} boxcar", "complex": "Complex U-Net",
          "real": "Real U-Net (C2 input)"}
# categorical slots 1-3 of the reference palette (fixed order: entity -> colour)
COLORS = {"boxcar": "#2a78d6", "complex": "#eb6834", "real": "#1baf7a"}
SHORT = {"boxcar": "boxcar", "complex": "complex", "real": "real"}
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"


# ----------------------------------------------------------------------------------------------- estimates
@torch.no_grad()
def net_estimate(model, kind: str, s_vv, s_vh, device) -> np.ndarray:
    obs = normalise_observation(s_vv, s_vh)
    batch = {k: torch.from_numpy(obs[k])[None].to(device) for k in ("x", "c2")}
    c2 = covariance_stack(model(model_input(kind, batch)))[0].cpu().double().numpy()
    s11, s22 = obs["scales"]
    return c2 * np.array([s11, s22, math.sqrt(s11 * s22), math.sqrt(s11 * s22)])[:, None, None]


def estimates(scene, s_vv, s_vh, models: dict, device) -> dict:
    c2 = form_c2(s_vv, s_vh, dtype=np.float64)
    out = {"noisy": c2, "boxcar": multilook(c2, BOXCAR)}
    for kind, model in models.items():
        out[kind] = net_estimate(model, kind, s_vv, s_vh, device)
    return out


def scene_metrics(scene, est: dict) -> dict:
    truth = scene.truth_c2(np.float64)
    gt = scene.gamma
    hom = homogeneous_mask(scene)
    res = {}
    for name, c2 in est.items():
        g = gamma_of(c2)
        res[name] = {
            "enl": enl(c2[0], hom) if hom.sum() >= MIN_ENL_PX else float("nan"),
            "psnr_db": psnr_db(c2[0], truth[0]),
            "cssim": complex_ssim(g, gt),
            "phase_err_deg": phase_error_deg(g, gt),
            "points": point_metrics(c2[0], truth[0], scene.points),
        }
    return res


def showcase_scene(size: int = 256, n_points: int = 5, seed: int = SHOWCASE_SEED, min_hom_px: int = 800):
    """First seeded scene with a large homogeneous field and all requested point scatterers (deterministic)."""
    cfg = SceneConfig(shape=(size, size))
    for k in range(200):
        rng = np.random.default_rng([seed, k])
        scene = make_scene(cfg, rng, n_points=n_points)
        if len(scene.points) == n_points and homogeneous_mask(scene).sum() >= min_hom_px:
            return scene, observe(scene, rng)
    raise RuntimeError("no showcase scene found")


def test_scenes(n: int, size: int = 256, seed: int = TEST_SEED):
    cfg = SceneConfig(shape=(size, size))
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        scene = make_scene(cfg, rng)
        yield scene, observe(scene, rng)


def run_test_set(models: dict, device, n: int, size: int) -> dict:
    per = {}
    g_all = {}
    gt_all = []
    for scene, (s_vv, s_vh) in test_scenes(n, size):
        est = estimates(scene, s_vv, s_vh, models, device)
        m = scene_metrics(scene, est)
        not_point = np.abs(scene.a_vv) == 0
        gt_all.append(scene.gamma[not_point])
        for name, c2 in est.items():
            per.setdefault(name, []).append(m[name])
            g_all.setdefault(name, []).append(gamma_of(c2)[not_point])
    gt = np.concatenate(gt_all)
    summary = {}
    for name, rows in per.items():
        pts = [p for r in rows for p in r["points"]]
        summary[name] = {
            "enl": float(np.nanmean([r["enl"] for r in rows])),
            "psnr_db": float(np.mean([r["psnr_db"] for r in rows])),
            "cssim": float(np.mean([r["cssim"] for r in rows])),
            "phase_err_deg": float(np.nanmean([r["phase_err_deg"] for r in rows])),
            "point_bright_px": float(np.mean([p["bright_px"] for p in pts])) if pts else float("nan"),
            "point_peak_err_db": float(np.mean([p["peak_db"] - p["true_peak_db"] for p in pts])) if pts else float("nan"),
            "n_points": len(pts),
            "gamma_bins": gamma_bins(np.concatenate(g_all[name]), gt),
        }
    summary["_truth_bright_px"] = 0
    summary["_n_scenes"] = n
    return summary


# ----------------------------------------------------------------------------------------------- figures
def _style(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.titlesize": 11, "figure.dpi": 100,
                         "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def _gamma_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    steps = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    return LinearSegmentedColormap.from_list("gamma_blue", steps)


def composite(c2: np.ndarray, lims) -> np.ndarray:
    """RGB = VV dB, VH dB, VV/VH dB, each clipped to fixed limits taken from the truth (same for every panel)."""
    chans = [db(c2[0]), db(c2[1]), db(c2[0]) - db(c2[1])]
    return np.dstack([np.clip((c - lo) / (hi - lo), 0, 1) for c, (lo, hi) in zip(chans, lims)])


def composite_limits(truth: np.ndarray):
    chans = [db(truth[0]), db(truth[1]), db(truth[0]) - db(truth[1])]
    return [tuple(np.percentile(c, [1, 99])) for c in chans]


def _metric_line(m: dict | None) -> str:
    if m is None:
        return "reference"
    e = m["enl"]
    enl_s = ">1000" if e > 1000 else f"{e:.1f}"
    return f"ENL {enl_s}   PSNR {m['psnr_db']:.1f} dB\ncomplex SSIM {m['cssim']:.2f}"


def fig_showcase(scene, est: dict, metrics: dict, names: list[str], path):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    _style(plt)
    truth = scene.truth_c2(np.float64)
    lims = composite_limits(truth)
    cols = ["truth"] + names
    fig, axes = plt.subplots(2, len(cols), figsize=(3.1 * len(cols), 6.6), gridspec_kw={"hspace": 0.12, "wspace": 0.04})
    cmap = _gamma_cmap()
    hom = homogeneous_mask(scene)
    for j, name in enumerate(cols):
        c2 = truth if name == "truth" else est[name]
        a0, a1 = axes[0, j], axes[1, j]
        a0.imshow(composite(c2, lims), interpolation="nearest")
        a1.imshow(np.abs(gamma_of(c2)), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        a0.set_title("Truth (noise-free)" if name == "truth" else LABELS[name], color=INK)
        if name == "truth":
            a0.contour(hom, levels=[0.5], colors=["#ffffff"], linewidths=1.2, linestyles="--")
        for a in (a0, a1):
            a.set_xticks([])
            a.set_yticks([])
        a1.set_xlabel(_metric_line(None if name == "truth" else metrics[name]), fontsize=10, color=INK, labelpad=6)
    axes[0, 0].set_ylabel("RGB: VV | VH | VV/VH (dB)", color=INK2)
    axes[1, 0].set_ylabel("|gamma| (VV-VH coherence)", color=INK2)
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap), ax=axes.ravel().tolist(), fraction=0.012, pad=0.01, shrink=0.45,
                      anchor=(0.0, 0.05))
    cb.set_label("|gamma|", color=INK2)
    cb.outline.set_visible(False)
    if "noisy" in names:
        j = cols.index("noisy")
        axes[1, j].text(0.5, 0.5, "single look:\n|gamma| = 1 by construction", transform=axes[1, j].transAxes,
                        ha="center", va="center", fontsize=9, color=INK,
                        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    fig.text(0.5, 0.005, "Dashed outline (truth): homogeneous field used for ENL (mu^2/var of VV intensity). "
             "PSNR on VV intensity in dB vs truth; complex SSIM on gamma vs truth.", ha="center", fontsize=8.5, color=INK2)
    fig.savefig(path)
    plt.close(fig)


def fig_point_zoom(scene, est: dict, names: list[str], path, half: int = 16):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    _style(plt)
    truth = scene.truth_c2(np.float64)
    # the most isolated-looking point: highest truth peak-to-background ratio
    pm = point_metrics(truth[0], truth[0], scene.points)
    k = int(np.argmax([p["true_peak_db"] - p["bg_db"] for p in pm]))
    r, c = scene.points[k]
    sl = (slice(r - half, r + half + 1), slice(c - half, c + half + 1))
    bg = pm[k]["bg_db"]
    vmin, vmax = bg - 8, pm[k]["true_peak_db"]
    cols = ["truth"] + names
    fig, axes = plt.subplots(1, len(cols), figsize=(2.9 * len(cols), 3.7), gridspec_kw={"wspace": 0.05})
    for ax, name in zip(axes, cols):
        img = truth[0] if name == "truth" else est[name][0]
        p = point_metrics(img, truth[0], [(r, c)])[0]
        ax.imshow(db(img[sl]), cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        for rad in (2, 8):
            ax.add_patch(Circle((half, half), rad, fill=False, ec="#eb6834", lw=1.0, ls="--"))
        ax.set_title("Truth" if name == "truth" else LABELS[name], color=INK)
        ax.set_xlabel(f"peak {p['peak_db']:+.1f} dB (truth {p['true_peak_db']:+.1f})\n"
                      f"bright px in ring: {p['bright_px']} / {p['ring_px']}", color=INK, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, -0.06, f"VV intensity, dB ({vmin:.0f} to {vmax:.0f}). Ring 2-8 px around the scatterer; "
             f"bright = > 3x the true background mean ({bg:.1f} dB). Single-look speckle exceeds that on ~5% of pixels.",
             ha="center", fontsize=8.5, color=INK2)
    fig.savefig(path)
    plt.close(fig)
    return {"row": r, "col": c}


def fig_gamma_bias(summary: dict, names: list[str], path):
    import matplotlib.pyplot as plt
    _style(plt)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"wspace": 0.42})
    a0.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    a0.text(0.93, 0.97, "ideal", color=INK2, fontsize=9, ha="right", va="top", rotation=0)
    for name in names:
        rows = summary[name]["gamma_bins"]
        t = [r["true"] for r in rows]
        a0.plot(t, [r["est"] for r in rows], "-o", color=COLORS[name], lw=1.6, ms=5.5, mec="white", mew=1.2,
                label=LABELS[name])
        a1.plot(t, [r["rmse"] for r in rows], "-o", color=COLORS[name], lw=1.6, ms=5.5, mec="white", mew=1.2,
                label=LABELS[name])
        a0.annotate(SHORT[name], (t[-1], rows[-1]["est"]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    a0.set(xlim=(0, 1.0), ylim=(0, 1.0), xlabel="true |gamma| (bin mean)", ylabel="mean estimated |gamma|",
           title="Coherence magnitude: estimate vs truth")
    a1.set(xlim=(0, 1.0), ylim=(0, None), xlabel="true |gamma| (bin mean)", ylabel="RMSE of |gamma|",
           title="Coherence magnitude error")
    for a in (a0, a1):
        a.grid(True, color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    a1.legend(frameon=False, loc="upper right")
    fig.text(0.5, -0.04, f"{summary['_n_scenes']} seeded test scenes, all non-point pixels, bins of 0.1 in true |gamma|. "
             "Single look not shown: its |gamma| is 1 everywhere.", ha="center", fontsize=8.5, color=INK2)
    fig.savefig(path)
    plt.close(fig)


def metrics_markdown(show: dict, test: dict, names: list[str]) -> str:
    lines = ["| method | ENL (showcase) | PSNR dB (showcase) | complex SSIM (showcase) | ENL (test) | PSNR dB (test) | "
             "complex SSIM (test) | median phase err deg (test, true |g|>0.3) | bright px / point (test) | "
             "peak error dB (test) |", "|" + "---|" * 10]
    for n in names:
        s, t = show[n], test[n]
        lines.append(f"| {LABELS[n]} | {s['enl']:.1f} | {s['psnr_db']:.1f} | {s['cssim']:.3f} | {t['enl']:.1f} | "
                     f"{t['psnr_db']:.1f} | {t['cssim']:.3f} | {t['phase_err_deg']:.1f} | {t['point_bright_px']:.1f} | "
                     f"{t['point_peak_err_db']:+.1f} |")
    lines += ["", "| method | " + " | ".join(f"{r['lo']:.1f}-{r['hi']:.1f}" for r in test[names[0]]["gamma_bins"]) + " |",
              "|" + "---|" * (1 + len(test[names[0]]["gamma_bins"]))]
    for n in names:
        if n == "noisy":
            continue
        lines.append(f"| {LABELS[n]} bias | " + " | ".join(f"{r['bias']:+.3f}" for r in test[n]["gamma_bins"]) + " |")
        lines.append(f"| {LABELS[n]} RMSE | " + " | ".join(f"{r['rmse']:.3f}" for r in test[n]["gamma_bins"]) + " |")
    return "\n".join(lines) + "\n"


def run(ckpts: dict, out_dir, n_test: int = 20, size: int = 256, device: str = "auto", log=print) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)
    models = {}
    for kind, path in ckpts.items():
        if path:
            model, ck = load_model(path, dev)
            if ck["kind"] != kind:
                raise ValueError(f"{path} is a {ck['kind']} model, not {kind}")
            models[kind] = model
            log(f"{kind}: {path} (step {ck['step']}, val KL {ck['val']['kl']:.4f})")
    names = ["noisy", "boxcar"] + [k for k in ("complex", "real") if k in models]

    scene, (s_vv, s_vh) = showcase_scene(size, n_points=5 if size >= 192 else 2,
                                         min_hom_px=int(800 * (size / 256) ** 2))
    est = estimates(scene, s_vv, s_vh, models, dev)
    show = scene_metrics(scene, est)
    main_names = ["noisy"] + (["complex"] if "complex" in models else names[1:2])
    fig_showcase(scene, est, show, main_names, out / "showcase_main.png")
    fig_showcase(scene, est, show, names, out / "showcase_all.png")
    zoom = fig_point_zoom(scene, est, names, out / "point_zoom.png")

    test = run_test_set(models, dev, n_test, size)
    fig_gamma_bias(test, [n for n in names if n != "noisy"], out / "gamma_bias.png")
    md = metrics_markdown(show, test, names)
    (out / "metrics.md").write_text(md)
    (out / "metrics.json").write_text(json.dumps({"showcase": show, "test": test, "zoom_point": zoom},
                                                 indent=2, default=float))
    log(md)
    return {"showcase": show, "test": test}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate the simulation experiment and draw the figures.")
    ap.add_argument("--complex", default=None, help="checkpoint of the complex U-Net (best.pt)")
    ap.add_argument("--real", default=None, help="checkpoint of the real U-Net baseline (best.pt)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    run({"complex": a.complex, "real": a.real}, a.out_dir, a.n_test, a.size, a.device)


if __name__ == "__main__":
    main()
