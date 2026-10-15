"""Does the network use the VV-VH coherence, or does it only despeckle the powers?

    python -m sar_denoise.evaluate --run ~/runs/r1 --split val

Every method predicts the covariance of one look from the OTHER look (independent speckle and noise), and is
scored with the same likelihood the network was trained on (lower is better), per patch class:

    net          the trained network, Sigma_B = C_hat(A) + diag(noise_B)
    net_nocoh    the same network with its coherence switched off (off-diagonal = 0)
    box WxH full boxcar-multilooked covariance of look A (the classical estimator), Sigma_B = boxcar(A)
    box WxH diag the same boxcar with the off-diagonal set to 0 (a power-only despeckler)

Reading it:  coherence gain = nll(net_nocoh) - nll(net)  is how much of the network's score comes from
the VV-VH coherence. If it is ~0 on coherent patches, the network is not using coherence. net vs box full
says whether it beats the standard estimator. Differences are what matter; absolute NLL values carry an
arbitrary constant and depend on the signal level, so do not compare them across classes.

Coherence quality (pixels where the independent reference, a 9x65 boxcar of the other look, has |gamma| > 0.3):
    sign agreement   share of pixels with Re(gamma_hat conj(gamma_ref)) > 0   (is the sign of C12 right?)
    median phase err median |angle(gamma_hat conj(gamma_ref))| in radians     (wrap-safe)
False coherence: mean |gamma_hat| and the share of pixels with |gamma_hat| > 0.3. On incoherent patches both
should be small.
The reference is itself a noisy, noise-biased estimate, so use these numbers to compare checkpoints and
methods, not as absolute truth.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import PatchDataset, collate
from .losses import covariance_nll
from .models import UNet, make_input, raw_to_covariance
from .patches import CLASSES, PatchConfig
from .train import pick_device, to_device

REF_WINDOW = (9, 65)
REF_GAMMA_MIN = 0.3
BIN_EDGES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01)
SWEEP_KS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def boxcar(c2: torch.Tensor, window: tuple[int, int]) -> torch.Tensor:
    """Boxcar average of (B,4,H,W) over (azimuth, range); odd windows, reflect padding."""
    wr, wc = window
    if wr % 2 == 0 or wc % 2 == 0:
        raise ValueError("boxcar windows must be odd")
    x = F.pad(c2, (wc // 2, wc // 2, wr // 2, wr // 2), mode="reflect")
    return F.avg_pool2d(x, (wr, wc), stride=1)


def covariance_to_raw(c2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Inverse of the head: a covariance stack (B,4,H,W) -> raw (log C11, log C22, u, v), |gamma| capped below 1."""
    c11, c22 = c2[:, 0].clamp_min(eps), c2[:, 1].clamp_min(eps)
    mag = torch.hypot(c2[:, 2], c2[:, 3])
    g = (mag / torch.sqrt(c11 * c22)).clamp(max=1.0 - 1e-6)
    scale = torch.atanh(g) / mag.clamp_min(eps)
    return torch.stack([torch.log(c11), torch.log(c22), c2[:, 2] * scale, c2[:, 3] * scale], dim=1)


def scale_gamma(raw: torch.Tensor, k: float) -> torch.Tensor:
    """Multiply the predicted coherence by k (phase and powers untouched), keeping |gamma| < 1."""
    u, v = raw[:, 2], raw[:, 3]
    m = torch.sqrt(u * u + v * v + 1e-8)
    g = (k * torch.tanh(m)).clamp(max=1.0 - 1e-6)
    f = torch.atanh(g) / m
    out = raw.clone()
    out[:, 2], out[:, 3] = u * f, v * f
    return out


def _gamma_of(c2: torch.Tensor, eps: float = 1e-12):
    s = torch.sqrt(c2[:, 0].clamp_min(eps) * c2[:, 1].clamp_min(eps))
    return c2[:, 2] / s, c2[:, 3] / s


@torch.no_grad()
def score_dataset(model, dataset, device, use_noise_input=True, windows=((5, 33), (9, 65)), batch_size=8,
                  return_bins=False):
    """Per-class scores (see module docstring). model: callable (B,C,H,W) -> raw (B,4,H,W).

    With return_bins=True returns (per_class, bins): bins[k] describes the pixels whose reference |gamma|
    (9x65 boxcar of the other look) falls in BIN_EDGES[k]..BIN_EDGES[k+1], over all classes."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    nll_sum = defaultdict(lambda: defaultdict(float))
    n_px = defaultdict(int)
    n_patch = defaultdict(int)
    gam_sum, gam_hi = defaultdict(float), defaultdict(float)
    phase = defaultdict(list)
    nb = len(BIN_EDGES) - 1
    bins = {k: defaultdict(float) for k in range(nb)}
    for b in loader:
        b = to_device(b, device)
        valid = b["valid"]
        for tag, other in (("a", "b"), ("b", "a")):
            x, xn = b[f"c2_{tag}"], b[f"noise_{tag}"]
            t, tn = b[f"c2_{other}"], b[f"noise_{other}"]
            raw = model(make_input(x, xn if use_noise_input else None))
            raw0 = raw.clone()
            raw0[:, 2:] = 0.0
            maps = {"net": covariance_nll(raw, t, tn, valid, "map"), "net_nocoh": covariance_nll(raw0, t, tn, valid, "map")}
            zero_noise = torch.zeros_like(tn)
            for w in windows:
                est = boxcar(x, w)
                est_d = est.clone()
                est_d[:, 2:] = 0.0
                name = f"box{w[0]}x{w[1]}"
                maps[f"{name}_full"] = covariance_nll(covariance_to_raw(est), t, zero_noise, valid, "map")
                maps[f"{name}_diag"] = covariance_nll(covariance_to_raw(est_d), t, zero_noise, valid, "map")

            g_abs = raw_to_covariance(raw)[-1]
            g_re, g_im = raw_to_covariance(raw)[4], raw_to_covariance(raw)[5]
            r_re, r_im = _gamma_of(boxcar(t, REF_WINDOW))
            ref_abs = torch.hypot(r_re, r_im)
            coh = valid & (ref_abs > REF_GAMMA_MIN)
            d_re, d_im = g_re * r_re + g_im * r_im, g_im * r_re - g_re * r_im       # gamma_hat * conj(gamma_ref)
            perr = torch.atan2(d_im, d_re)
            gain_map = maps["net_nocoh"] - maps["net"]
            for k in range(nb):
                sel = valid & (ref_abs >= BIN_EDGES[k]) & (ref_abs < BIN_EDGES[k + 1])
                bk = bins[k]
                bk["n"] += sel.sum().item()
                bk["g_hat"] += g_abs[sel].sum().item()
                bk["g_ref"] += ref_abs[sel].sum().item()
                bk["sign"] += (torch.cos(perr[sel]) > 0).sum().item()
                bk["perr"] += perr[sel].abs().sum().item()
                bk["gain"] += gain_map[sel].sum().item()
            for i, k in enumerate(b["class"].tolist()):
                c = CLASSES[k]
                m = valid[i]
                for name, mp in maps.items():
                    nll_sum[c][name] += mp[i][m].sum().item()
                n_px[c] += int(m.sum().item())
                n_patch[c] += 1 if tag == "a" else 0
                gam_sum[c] += g_abs[i][m].sum().item()
                gam_hi[c] += (g_abs[i][m] > 0.3).sum().item()
                phase[c].append(perr[i][coh[i]].cpu().numpy())

    out = {}
    for c in CLASSES:
        if not n_px[c]:
            continue
        p = np.concatenate(phase[c]) if phase[c] else np.zeros(0)
        row = {"patches": n_patch[c], **{k: v / n_px[c] for k, v in nll_sum[c].items()}}
        row["coherence_gain"] = row["net_nocoh"] - row["net"]
        row["gamma_mean"] = gam_sum[c] / n_px[c]
        row["gamma_gt03"] = gam_hi[c] / n_px[c]
        row["ref_coherent_px"] = int(p.size)
        row["sign_agreement"] = float(np.mean(np.cos(p) > 0)) if p.size else float("nan")
        row["phase_err_median"] = float(np.median(np.abs(p))) if p.size else float("nan")
        out[c] = row
    if not return_bins:
        return out
    table = []
    for k in range(nb):
        d = bins[k]
        n = max(d["n"], 1)
        table.append({"ref_lo": BIN_EDGES[k], "ref_hi": min(BIN_EDGES[k + 1], 1.0), "pixels": int(d["n"]),
                      "mean_ref": d["g_ref"] / n, "mean_pred": d["g_hat"] / n, "sign_agreement": d["sign"] / n,
                      "phase_err_mean": d["perr"] / n, "coherence_gain": d["gain"] / n})
    return out, table


def format_report(res: dict) -> str:
    methods = [k for k in next(iter(res.values())) if k.startswith(("net", "box"))]
    lines = ["NLL per valid pixel, lower is better (compare columns within a row, not across rows)"]
    head = f"{'class':11s}{'patches':>8s}" + "".join(f"{m:>16s}" for m in methods) + f"{'coh gain':>10s}"
    lines += [head, "-" * len(head)]
    for c, r in res.items():
        lines.append(f"{c:11s}{r['patches']:8d}" + "".join(f"{r[m]:16.4f}" for m in methods) + f"{r['coherence_gain']:10.4f}")
    lines += ["", "coherence quality (reference = 9x65 boxcar of the other look, |gamma_ref| > 0.3) and false coherence"]
    head = f"{'class':11s}{'ref px':>10s}{'sign agree':>12s}{'phase err':>11s}{'mean |g^|':>11s}{'|g^|>0.3':>10s}"
    lines += [head, "-" * len(head)]
    for c, r in res.items():
        lines.append(f"{c:11s}{r['ref_coherent_px']:10d}{r['sign_agreement']:12.3f}{r['phase_err_median']:11.3f}"
                     f"{r['gamma_mean']:11.3f}{r['gamma_gt03']:10.3f}")
    return "\n".join(lines)


@torch.no_grad()
def calibration_sweep(model, dataset, device, use_noise_input=True, ks=SWEEP_KS, batch_size=8) -> list[dict]:
    """Is the predicted coherence the right size? No reference needed, only the likelihood.

    The network's |gamma_hat| is multiplied by k (k = 0 switches coherence off) and the NLL of the other look is
    recomputed. Pixels are grouped by the network's own |gamma_hat|. In each group the k that minimises the NLL
    is the best rescaling: k* ~ 1 means calibrated, k* > 1 means the network under-states the coherence, k* < 1
    that it over-states it. gain(k) = NLL(k = 0) - NLL(k), per pixel, larger is better."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    nb = len(BIN_EDGES) - 1
    nll = np.zeros((nb, len(ks)))
    n = np.zeros(nb)
    gsum = np.zeros(nb)
    for b in loader:
        b = to_device(b, device)
        valid = b["valid"]
        for tag, other in (("a", "b"), ("b", "a")):
            raw = model(make_input(b[f"c2_{tag}"], b[f"noise_{tag}"] if use_noise_input else None))
            g = raw_to_covariance(raw)[-1]
            maps = [covariance_nll(scale_gamma(raw, k), b[f"c2_{other}"], b[f"noise_{other}"], valid, "map") for k in ks]
            for j in range(nb):
                sel = valid & (g >= BIN_EDGES[j]) & (g < BIN_EDGES[j + 1])
                n[j] += sel.sum().item()
                gsum[j] += g[sel].sum().item()
                for ki, mp in enumerate(maps):
                    nll[j, ki] += mp[sel].sum().item()
    rows = []
    i0, i1 = ks.index(0.0), ks.index(1.0)
    for j in range(nb):
        m = max(n[j], 1)
        gain = (nll[j, i0] - nll[j]) / m
        best = int(np.argmax(gain))
        rows.append({"pred_lo": BIN_EDGES[j], "pred_hi": min(BIN_EDGES[j + 1], 1.0), "pixels": int(n[j]),
                     "mean_pred": gsum[j] / m, "gain_k1": float(gain[i1]), "best_k": float(ks[best]),
                     "gain_best": float(gain[best]), **{f"gain_k{k:g}": float(g) for k, g in zip(ks, gain)}})
    return rows


def format_sweep(rows: list[dict]) -> str:
    head = f"{'pred |gamma|':>13s}{'pixels':>10s}{'mean pred':>11s}{'gain k=1':>10s}{'best k':>8s}{'gain best':>11s}"
    lines = ["rescaling test: multiply the network's |gamma_hat| by k and re-score (no reference involved)",
             "best k > 1: network understates coherence; best k < 1: overstates; ~1: calibrated", head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['pred_lo']:6.1f} - {r['pred_hi']:4.1f}{r['pixels']:10d}{r['mean_pred']:11.3f}{r['gain_k1']:10.4f}"
                     f"{r['best_k']:8.2f}{r['gain_best']:11.4f}")
    return "\n".join(lines)


def format_bins(table: list[dict]) -> str:
    head = (f"{'ref |gamma|':>12s}{'pixels':>10s}{'mean ref':>10s}{'mean pred':>11s}{'sign agree':>12s}"
            f"{'phase err':>11s}{'coh gain':>10s}")
    lines = ["calibration: pixels grouped by the REFERENCE |gamma| (9x65 boxcar of the other look)",
             "mean pred = network |gamma_hat| in the same pixels; coh gain = NLL(no coherence) - NLL(network) per pixel", head,
             "-" * len(head)]
    for r in table:
        lines.append(f"{r['ref_lo']:5.1f} - {r['ref_hi']:4.1f}{r['pixels']:10d}{r['mean_ref']:10.3f}{r['mean_pred']:11.3f}"
                     f"{r['sign_agreement']:12.3f}{r['phase_err_mean']:11.3f}{r['coherence_gain']:10.4f}")
    return "\n".join(lines)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    meta = ck["meta"]
    net = UNet(in_ch=meta["in_ch"], base=meta["base"], depth=meta["depth"])
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), meta, ck


def parse_windows(text: str):
    return tuple(tuple(int(v) for v in w.split("x")) for w in text.split(","))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory written by sar_denoise.train")
    ap.add_argument("--ckpt", default="best", help="best | last | path to a .pt file")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--regions-dir", default="~/data/sar/regions")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--windows", default="5x33,9x65", help="boxcar baselines, azimuth x range, odd numbers")
    ap.add_argument("--patch-rows", type=int, default=64)
    ap.add_argument("--patch-cols", type=int, default=512)
    a = ap.parse_args(argv)

    run = Path(a.run).expanduser()
    ckpt = Path(a.ckpt).expanduser() if a.ckpt.endswith(".pt") else run / f"{a.ckpt}.pt"
    dev = pick_device(a.device)
    model, meta, ck = load_model(ckpt, dev)
    ds = PatchDataset(a.regions_dir, a.split, PatchConfig(rows=a.patch_rows, cols=a.patch_cols))
    print(f"checkpoint {ckpt.name} (step {ck.get('step')}) | split {a.split}: {len(ds)} patches | device {dev}\n")
    res, bins = score_dataset(model, ds, dev, meta.get("use_noise_input", True), parse_windows(a.windows),
                              return_bins=True)
    print(format_report(res))
    print("\n" + format_bins(bins))
    sweep = calibration_sweep(model, ds, dev, meta.get("use_noise_input", True))
    print("\n" + format_sweep(sweep))
    with open(run / f"eval_{a.split}_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)
    with open(run / f"eval_{a.split}_bins.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bins[0]))
        w.writeheader()
        w.writerows(bins)
    out = run / f"eval_{a.split}.csv"
    with open(out, "w", newline="") as f:
        cols = ["class"] + list(next(iter(res.values())))
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c, r in res.items():
            w.writerow({"class": c, **r})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()