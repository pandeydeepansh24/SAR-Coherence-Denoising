"""Supervised despeckling on fully synthetic scenes (simulation experiment, see TASK_simulation.md).

    python -m sar_denoise.sim_train --model complex --out-dir ~/runs/sim_complex --steps 3000
    python -m sar_denoise.sim_train --model real    --out-dir ~/runs/sim_real    --steps 3000

Every step draws FRESH scenes (sim_scene.make_scene) and fresh speckle, so the network cannot memorise anything;
the target is the known noise-free covariance and the loss is the 2x2 complex-Gaussian KL (sim_losses.kl_loss).

Models (same PSD head, same loss, same data):
  complex : ComplexUNet on the two complex single-look pixels (S_VV, S_VH), with a random common phase e^{j phi}
  real    : the existing UNet on the single-look C2 [C11, C22, Re C12, Im C12] (baseline). C2 is blind to the common
            phase, so the augmentation does nothing for it.
Per-patch normalisation (from the OBSERVATION, like patches.normalise_pair): S_VV / sqrt(s11), S_VH / sqrt(s22),
s = mean single-look power; the truth gets the same D C D scaling, so gamma and its phase are untouched.

Writes to --out-dir: config.json, log.csv, best.pt (lowest validation KL), last.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .complex_models import ComplexUNet, n_params
from .covariance import form_c2
from .losses import coherence_metrics
from .models import UNet
from .sim_losses import kl_loss
from .sim_scene import SceneConfig, make_scene, observe

VAL_SEED = 10_000
MODELS = ("complex", "real")


# ----------------------------------------------------------------------------------------------- data
def normalise_observation(s_vv: np.ndarray, s_vh: np.ndarray, scene=None) -> dict:
    """Per-patch normalisation of one observation (and, if given, of its truth) with the observation's mean powers."""
    s11 = float(np.mean(np.abs(s_vv) ** 2))
    s22 = float(np.mean(np.abs(s_vh) ** 2))
    vv, vh = s_vv / math.sqrt(s11), s_vh / math.sqrt(s22)
    out = {
        "x": np.stack([vv.real, vh.real, vv.imag, vh.imag]).astype(np.float32),
        "c2": form_c2(vv, vh),
        "scales": np.array([s11, s22], dtype=np.float64),
    }
    if scene is not None:
        d = np.array([1 / s11, 1 / s22, 1 / math.sqrt(s11 * s22), 1 / math.sqrt(s11 * s22)])[:, None, None]
        out["truth"] = (scene.truth_c2(np.float64) * d).astype(np.float32)
    return out


def make_sample(cfg: SceneConfig, rng, phase_aug: bool = True, return_scene: bool = False) -> dict:
    scene = make_scene(cfg, rng)
    s_vv, s_vh = observe(scene, rng)
    if phase_aug:
        rot = np.exp(1j * rng.uniform(-np.pi, np.pi))
        s_vv, s_vh = s_vv * rot, s_vh * rot
    out = normalise_observation(s_vv, s_vh, scene)
    if return_scene:
        out.update(scene=scene, s_vv=s_vv, s_vh=s_vh)
    return out


def stack(samples: list[dict]) -> dict:
    return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in ("x", "c2", "truth")}


class SimStream(IterableDataset):
    """Infinite stream of freshly simulated batches; each DataLoader worker gets its own seed."""

    def __init__(self, cfg: SceneConfig, batch_size: int, seed: int):
        self.cfg, self.batch_size, self.seed = cfg, batch_size, seed

    def __iter__(self):
        wi = get_worker_info()
        rng = np.random.default_rng([self.seed, 0 if wi is None else wi.id + 1])
        while True:
            yield stack([make_sample(self.cfg, rng) for _ in range(self.batch_size)])


def fixed_set(cfg: SceneConfig, n: int, seed: int) -> dict:
    """Seeded validation / test scenes (same for every run and model)."""
    return stack([make_sample(cfg, np.random.default_rng([seed, i])) for i in range(n)])


# ----------------------------------------------------------------------------------------------- models
def build_model(kind: str, base: int | None = None, depth: int = 3):
    if kind == "complex":
        args = {"in_ch": 2, "base": base or 16, "depth": depth}
        return ComplexUNet(**args), args
    if kind == "real":
        args = {"in_ch": 4, "base": base or 24, "depth": depth}
        return UNet(**args), args
    raise ValueError(f"unknown model {kind!r}, expected one of {MODELS}")


def model_input(kind: str, batch: dict) -> torch.Tensor:
    return batch["x"] if kind == "complex" else batch["c2"]


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = build_model(ck["kind"], ck["args"]["base"], ck["args"]["depth"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, kind: str, data: dict, device, batch_size: int = 8) -> dict:
    model.eval()
    sums, n = {}, 0
    for i in range(0, len(data["x"]), batch_size):
        b = {k: v[i:i + batch_size].to(device) for k, v in data.items()}
        raw = model(model_input(kind, b))
        parts = {"kl": kl_loss(raw, b["truth"])}
        parts.update(coherence_metrics(raw, b["truth"]))
        k = len(b["x"])
        for key, val in parts.items():
            sums[key] = sums.get(key, 0.0) + float(val) * k
        n += k
    model.train()
    return {k: v / n for k, v in sums.items()}


# ----------------------------------------------------------------------------------------------- training
def train(kind: str, out_dir, steps: int = 3000, batch_size: int = 8, lr: float = 1e-3, base: int | None = None,
          depth: int = 3, size: int = 128, n_val: int = 16, eval_every: int = 250, device: str = "auto",
          seed: int = 0, workers: int = 3, warmup: int = 100, points_per_10k: float | None = None, log=print) -> dict:
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    dev = pick_device(device)
    cfg = SceneConfig(shape=(size, size))
    if points_per_10k is not None:
        cfg.points_per_10k_px = points_per_10k
    model, margs = build_model(kind, base, depth)
    model.to(dev).train()
    config = {"kind": kind, "args": margs, "steps": steps, "batch_size": batch_size, "lr": lr, "size": size,
              "n_val": n_val, "seed": seed, "device": str(dev), "n_params": n_params(model), "scene": asdict(cfg)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    log(f"{kind} U-Net, {n_params(model):,} parameters, device {dev}, {steps} steps x {batch_size} fresh {size}x{size} scenes")

    val = fixed_set(cfg, n_val, VAL_SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    loader = DataLoader(SimStream(cfg, batch_size, seed), batch_size=None, num_workers=workers,
                        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)

    def save(path, step, metrics):
        torch.save({"kind": kind, "args": margs, "state_dict": model.state_dict(), "step": step, "val": metrics,
                    "scene": asdict(cfg)}, path)

    best, t0, run_loss = float("inf"), time.time(), []
    fields = ["step", "train_kl", "val_kl", "val_gamma_abs_mae", "val_gamma_complex_rmse", "val_phase_abs_err_rad",
              "val_log_c11_mae", "val_log_c22_mae", "lr", "minutes"]
    with open(out / "log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        it = iter(loader)
        for step in range(1, steps + 1):
            b = {k: v.to(dev, non_blocking=True) for k, v in next(it).items()}
            loss = kl_loss(model(model_input(kind, b)), b["truth"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss.append(loss.detach())          # no host sync per step (keeps the MPS queue full)
            if step % eval_every == 0 or step == steps:
                train_kl = float(torch.stack(run_loss).mean())
                if not math.isfinite(train_kl):
                    raise FloatingPointError(f"non-finite training loss before step {step}")
                m = evaluate(model, kind, val, dev)
                row = {"step": step, "train_kl": train_kl, "lr": sched.get_last_lr()[0],
                       "minutes": (time.time() - t0) / 60, **{f"val_{k}": v for k, v in m.items()}}
                w.writerow(row)
                fh.flush()
                run_loss = []
                tag = ""
                if m["kl"] < best:
                    best = m["kl"]
                    save(out / "best.pt", step, m)
                    tag = "  *best"
                log(f"step {step:5d}  train KL {row['train_kl']:.4f}  val KL {m['kl']:.4f}  "
                    f"|g| MAE {m['gamma_abs_mae']:.4f}  phase err {math.degrees(m['phase_abs_err_rad']):5.1f} deg  "
                    f"logC11 MAE {m['log_c11_mae']:.3f}  {row['minutes']:.1f} min{tag}")
    save(out / "last.pt", steps, m)
    return {"best_val_kl": best, "minutes": (time.time() - t0) / 60}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Supervised despeckling on synthetic scenes (complex or real U-Net).")
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=None, help="width (complex: 16 complex ch, real: 24 ch by default)")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-val", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--points-per-10k", type=float, default=None,
                    help="point scatterers per 100x100 px in TRAINING scenes (default 1.5; more = more weight on points)")
    a = ap.parse_args(argv)
    res = train(a.model, a.out_dir, a.steps, a.batch_size, a.lr, a.base, a.depth, a.size, a.n_val, a.eval_every,
                a.device, a.seed, a.workers, points_per_10k=a.points_per_10k)
    print(f"done: best val KL {res['best_val_kl']:.4f} in {res['minutes']:.1f} min")


if __name__ == "__main__":
    main()
