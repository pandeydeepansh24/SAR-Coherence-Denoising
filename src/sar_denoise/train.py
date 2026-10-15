"""Train the covariance-preserving despeckler on the region archives.

    python -m sar_denoise.train --regions-dir ~/data/sar/regions --out-dir ~/runs/smoke --steps 200

Each step: draw a class-balanced batch of patches, then the symmetric Noise2Noise loss (Look A -> Look B and
Look B -> Look A) with the noise-aware covariance likelihood. Every --eval-every steps the whole val split is
scored (no sampling, no weighting) and reported per class, together with the mean predicted |gamma| per
class. The incoherent classes should stay near the noise floor; if their mean |gamma| climbs, the model is
inventing coherence.

Writes to --out-dir: config.json, log.csv, last.pt, best.pt (lowest val loss).
A falling loss is NOT proof that coherence is preserved; scripts/evaluate.py (next) checks that.

Overfitting warning (seen in simulation): on a SMALL STATIC set of patches the network memorises the target
speckle (train loss keeps falling, val loss rises, predicted |gamma| drifts to ~0.9 everywhere, also in the
incoherent class). Fresh data every step does not do this. So: watch val_loss and val_gamma_incoherent, keep
best.pt, and stop early; more independent data (a second SLC product) is the real fix.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import PatchDataset, class_balanced_weights, collate, parse_mix
from .losses import covariance_nll, n2n_loss
from .models import UNet, make_input, raw_to_covariance
from .patches import CLASSES, PatchConfig


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_split(model, dataset, device, use_noise_input=True, batch_size=8) -> dict:
    """Symmetric N2N loss over the whole split, per class, plus mean predicted |gamma| per class.

    Means are over valid pixels of both directions (A->B and B->A)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    acc = {c: {"loss": 0.0, "gamma": 0.0, "n": 0} for c in CLASSES}
    for b in loader:
        b = to_device(b, device)
        valid = b["valid"]
        for tag, other in (("a", "b"), ("b", "a")):
            raw = model(make_input(b[f"c2_{tag}"], b[f"noise_{tag}"] if use_noise_input else None))
            nll = covariance_nll(raw, b[f"c2_{other}"], b[f"noise_{other}"], valid, reduction="map")
            g = raw_to_covariance(raw)[-1]
            for i, cls in enumerate(b["class"].tolist()):
                m = valid[i]
                d = acc[CLASSES[cls]]
                d["loss"] += nll[i][m].sum().item()
                d["gamma"] += g[i][m].sum().item()
                d["n"] += int(m.sum().item())
    model.train()
    out = {}
    for c, d in acc.items():
        if d["n"]:
            out[f"val_loss_{c}"] = d["loss"] / d["n"]
            out[f"val_gamma_{c}"] = d["gamma"] / d["n"]
    n_all = sum(d["n"] for d in acc.values())
    out["val_loss"] = sum(d["loss"] for d in acc.values()) / max(n_all, 1)
    return out


def train(regions_dir, out_dir, steps=200, batch_size=8, lr=2e-3, base=24, depth=3, device="auto", seed=0,
          eval_every=50, class_mix=None, use_noise_input=True, patch_cfg=None, weight_decay=0.0, log=print) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = pick_device(device)

    cfg = patch_cfg or PatchConfig()          # must match the patch size used by make_patches.py
    train_ds = PatchDataset(regions_dir, "train", cfg)
    try:
        val_ds = PatchDataset(regions_dir, "val", cfg)
    except ValueError:
        val_ds = None
    w = class_balanced_weights(train_ds.classes, class_mix)
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=steps * batch_size, replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate)

    in_ch = 6 if use_noise_input else 4
    model = UNet(in_ch=in_ch, base=base, depth=depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 0.05)
    meta = dict(steps=steps, batch_size=batch_size, lr=lr, base=base, depth=depth, in_ch=in_ch, seed=seed,
                use_noise_input=use_noise_input, class_mix=class_mix, weight_decay=weight_decay, device=str(dev), n_train=len(train_ds),
                n_val=0 if val_ds is None else len(val_ds))
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))
    log(f"device {dev} | train patches {len(train_ds)} | val patches {meta['n_val']} | "
        f"params {sum(p.numel() for p in model.parameters()):,}")

    fields = ["step", "train_loss", "lr", "seconds"] + [f"val_{k}_{c}" for k in ("loss", "gamma") for c in CLASSES] + ["val_loss"]
    best, rows, t0, ema = float("inf"), [], time.time(), None
    with open(out_dir / "log.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for step, batch in enumerate(loader, start=1):
            batch = to_device(batch, dev)
            opt.zero_grad()
            loss, parts = n2n_loss(model, batch, use_noise_input)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
            row = {"step": step, "train_loss": round(ema, 5), "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
            if val_ds is not None and (step % eval_every == 0 or step == steps):
                row.update({k: round(v, 5) for k, v in evaluate_split(model, val_ds, dev, use_noise_input).items()})
                state = {"model": model.state_dict(), "meta": meta, "step": step, "val": row}
                torch.save(state, out_dir / "last.pt")
                if row["val_loss"] < best:
                    best = row["val_loss"]
                    torch.save(state, out_dir / "best.pt")
                log(f"step {step:5d} | train {ema:.4f} | val {row['val_loss']:.4f} | " +
                    " | ".join(f"{c[:3]} loss {row.get(f'val_loss_{c}', float('nan')):.3f} |g| {row.get(f'val_gamma_{c}', float('nan')):.3f}" for c in CLASSES))
            elif step % 10 == 0:
                log(f"step {step:5d} | train {ema:.4f} | {row['seconds']}s")
            wr.writerow(row)
            f.flush()
            rows.append(row)
    torch.save({"model": model.state_dict(), "meta": meta, "step": steps}, out_dir / "last.pt")
    return {"rows": rows, "best_val": best, "out_dir": out_dir}


def main(argv=None) -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions-dir", default="~/data/sar/regions")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--class-mix", default="0.4,0.2,0.4", help="sampling mix: coherent,weak,incoherent")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--patch-rows", type=int, default=64, help="must match make_patches.py")
    ap.add_argument("--patch-cols", type=int, default=512, help="must match make_patches.py")
    ap.add_argument("--no-noise-input", action="store_true", help="do not feed the noise maps to the network")
    a = ap.parse_args(argv)
    train(a.regions_dir, a.out_dir, a.steps, a.batch_size, a.lr, a.base, a.depth, a.device, a.seed, a.eval_every,
          parse_mix(a.class_mix), not a.no_noise_input, PatchConfig(rows=a.patch_rows, cols=a.patch_cols), a.weight_decay)


if __name__ == "__main__":
    main()