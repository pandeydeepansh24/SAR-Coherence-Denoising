import csv
import json

import numpy as np
import pytest
import torch

from sar_denoise.dataset import PatchDataset, class_balanced_weights, collate, parse_mix
from sar_denoise.patches import PatchConfig, Region, build_index, save_region, write_index_csv
from sar_denoise.simulate import simulate_look_pair
from sar_denoise.train import evaluate_split, main, train
from sar_denoise.models import UNet

CFG = PatchConfig(rows=16, cols=64, stride_rows=16, stride_cols=64, min_valid=0.99, ml_window=(3, 9))
H, W = 32, 128


def make_region(rng, name, gamma, n11=0.05, n22=0.3):
    """A region with constant gamma (complex), as an archive dict like crop_products returns."""
    c11, c22 = 1.0, 0.4
    pair = simulate_look_pair(c11, c22, gamma * np.sqrt(c11 * c22), n11, n22, shape=(H, W), rng=rng)
    pair["pnr_db"] = np.full((2, H, W), 10.0, np.float32)
    pair["gamma"] = np.full((H, W), abs(gamma), np.float32)
    pair["valid"] = np.ones((H, W), bool)
    return pair


@pytest.fixture()
def regions_dir(tmp_path):
    rng = np.random.default_rng(0)
    specs = [("tr_pos", "train", 0.8), ("tr_neg", "train", -0.8), ("tr_inc", "train", 0.0),
             ("va_pos", "val", 0.8), ("va_inc", "val", 0.0)]
    rows = []
    for name, split, g in specs:
        arr = make_region(rng, name, g)
        reg = Region(name=name, split=split, swath="iw1", burst=0, lines=(0, 1), samples=(0, 1), kind="positive")
        save_region(tmp_path / f"{name}.npz", arr, {"region": name})
        rows += build_index(arr, reg, CFG)
    write_index_csv(rows, tmp_path / "index.csv")
    return tmp_path


def test_dataset_item_is_normalised_finite_and_keeps_gamma(regions_dir):
    ds = PatchDataset(regions_dir, "train", CFG)
    item = ds[0]
    assert item["c2_a"].shape == (4, 16, 64) and item["noise_a"].shape == (2, 16, 64) and item["valid"].dtype == torch.bool
    for k in ("c2_a", "c2_b", "noise_a", "noise_b"):
        assert torch.isfinite(item[k]).all()
    assert item["scales"].shape == (2,) and item["c2_a"][0].mean().item() == pytest.approx(1.0, abs=0.35)
    b = collate([ds[0], ds[1]])
    assert b["c2_a"].shape == (2, 4, 16, 64) and b["class"].dtype == torch.int64 and len(b["region"]) == 2


def test_dataset_split_filter_and_empty_split_error(regions_dir):
    assert {r["split"] for r in PatchDataset(regions_dir, "val", CFG).rows} == {"val"}
    with pytest.raises(ValueError):
        PatchDataset(regions_dir, "test", CFG)


def test_class_balanced_weights_hit_the_requested_mix():
    labels = ["coherent"] * 10 + ["weak"] * 30 + ["incoherent"] * 160
    w = class_balanced_weights(labels, {"coherent": 0.5, "weak": 0.1, "incoherent": 0.4})
    assert w.sum() == pytest.approx(1.0)
    p = {c: w[[l == c for l in labels]].sum() for c in ("coherent", "weak", "incoherent")}
    assert p == pytest.approx({"coherent": 0.5, "weak": 0.1, "incoherent": 0.4})
    w2 = class_balanced_weights(["coherent"] * 5 + ["incoherent"] * 5)            # a class missing: renormalised
    assert w2.sum() == pytest.approx(1.0)
    assert parse_mix("1,2,3") == {"coherent": 1.0, "weak": 2.0, "incoherent": 3.0}
    with pytest.raises(ValueError):
        parse_mix("1,2")


def test_training_runs_logs_checkpoints_and_lowers_the_loss(regions_dir, tmp_path):
    out = tmp_path / "run"
    msgs = []
    res = train(regions_dir, out, steps=60, batch_size=4, lr=3e-3, base=8, depth=2, device="cpu", eval_every=30,
                patch_cfg=CFG, log=msgs.append)
    assert (out / "config.json").exists() and (out / "best.pt").exists() and (out / "last.pt").exists()
    rows = list(csv.DictReader(open(out / "log.csv")))
    assert len(rows) == 60 and float(rows[-1]["train_loss"]) < float(rows[0]["train_loss"])
    assert rows[29]["val_loss"] != "" and rows[-1]["val_gamma_coherent"] != ""
    ck = torch.load(out / "best.pt", map_location="cpu")
    net = UNet(in_ch=ck["meta"]["in_ch"], base=8, depth=2)
    net.load_state_dict(ck["model"])
    assert json.loads((out / "config.json").read_text())["n_train"] > 0


def test_evaluate_split_reports_every_class_with_finite_numbers(regions_dir):
    net = UNet(in_ch=6, base=8, depth=2)
    m = evaluate_split(net, PatchDataset(regions_dir, "val", CFG), torch.device("cpu"))
    for c in ("coherent", "incoherent"):
        assert np.isfinite(m[f"val_loss_{c}"]) and 0.0 <= m[f"val_gamma_{c}"] < 1.0
    assert np.isfinite(m["val_loss"])


def test_cli_parses_and_runs(regions_dir, tmp_path):
    main(["--regions-dir", str(regions_dir), "--out-dir", str(tmp_path / "cli"), "--steps", "4", "--batch-size", "2",
          "--base", "8", "--depth", "2", "--device", "cpu", "--eval-every", "4",
          "--patch-rows", "16", "--patch-cols", "64"])
    assert (tmp_path / "cli" / "last.pt").exists()