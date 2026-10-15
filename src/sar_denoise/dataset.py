"""Training data: patches from the region archives written by scripts/make_patches.py.

PatchDataset reads index.csv, keeps one split, loads each region archive once (lazily), and returns a
patch normalised with patches.normalise_pair, so |gamma| and the phase of C12 are untouched and the data
and noise maps share one scale. No augmentation: rotations and transposes would break the azimuth/range
asymmetry and flips have not been tested, so they are left out on purpose.

class_balanced_weights gives each patch a sampling weight so that classes are drawn in a chosen mix. This
matters because overlapping coherent patches are few (about a quarter of the train patches) and the
incoherent ones would otherwise dominate the loss.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .patches import CLASSES, PatchConfig, extract_patch, load_region

DEFAULT_CLASS_MIX = {"coherent": 0.4, "weak": 0.2, "incoherent": 0.4}


def read_index(path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["row"], r["col"] = int(r["row"]), int(r["col"])
    return rows


class PatchDataset(Dataset):
    def __init__(self, regions_dir, split: str, cfg: PatchConfig | None = None, classes=None, index_path=None):
        self.dir = Path(regions_dir).expanduser()
        self.cfg = cfg or PatchConfig()
        rows = read_index(index_path or self.dir / "index.csv")
        self.rows = [r for r in rows if r["split"] == split and (classes is None or r["class"] in classes)]
        if not self.rows:
            raise ValueError(f"no patches for split={split!r} classes={classes} in {self.dir}")
        self._cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _arrays(self, region: str) -> dict:
        if region not in self._cache:
            self._cache[region], _ = load_region(self.dir / f"{region}.npz")
        return self._cache[region]

    @property
    def classes(self) -> list[str]:
        return [r["class"] for r in self.rows]

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        p = extract_patch(self._arrays(r["region"]), r["row"], r["col"], self.cfg, normalise=True)
        valid = p["valid"].astype(bool)
        out = {k: torch.from_numpy(np.nan_to_num(np.ascontiguousarray(p[k], dtype=np.float32), nan=0.0,
                                                 posinf=0.0, neginf=0.0))
               for k in ("c2_a", "c2_b", "noise_a", "noise_b")}
        out["valid"] = torch.from_numpy(valid)
        out["scales"] = torch.tensor(p["scales"], dtype=torch.float32)       # (s11, s22), to undo the normalisation
        out["class"] = CLASSES.index(r["class"])
        out["region"] = r["region"]
        return out


def class_balanced_weights(labels, mix: dict | None = None) -> np.ndarray:
    """Per-patch sampling weights so that class c is drawn with probability mix[c] (renormalised over the
    classes that actually occur). Every patch of a class gets the same weight."""
    mix = dict(DEFAULT_CLASS_MIX if mix is None else mix)
    labels = np.asarray(list(labels))
    present = [c for c in CLASSES if (labels == c).any() and mix.get(c, 0.0) > 0]
    if not present:
        raise ValueError("none of the classes in the mix occurs in the data")
    total = sum(mix[c] for c in present)
    w = np.zeros(len(labels), dtype=np.float64)
    for c in present:
        m = labels == c
        w[m] = (mix[c] / total) / m.sum()
    return w


def parse_mix(text: str) -> dict:
    """'0.4,0.2,0.4' -> {'coherent': 0.4, 'weak': 0.2, 'incoherent': 0.4}"""
    vals = [float(x) for x in text.split(",")]
    if len(vals) != len(CLASSES) or min(vals) < 0 or sum(vals) <= 0:
        raise ValueError(f"class mix needs {len(CLASSES)} non-negative numbers, got {text!r}")
    return dict(zip(CLASSES, vals))


def collate(items: list[dict]) -> dict:
    out = {}
    for k in items[0]:
        vals = [it[k] for it in items]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        elif isinstance(vals[0], int):
            out[k] = torch.tensor(vals)
        else:
            out[k] = vals
    return out