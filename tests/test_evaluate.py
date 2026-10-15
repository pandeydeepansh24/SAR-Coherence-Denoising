import csv

import numpy as np
import pytest
import torch

from sar_denoise.dataset import PatchDataset
from sar_denoise.models import raw_to_covariance
from sar_denoise.evaluate import BIN_EDGES, boxcar, calibration_sweep, format_sweep, scale_gamma, covariance_to_raw, format_bins, format_report, main, score_dataset
from sar_denoise.models import covariance_stack
from sar_denoise.train import train
from tests.test_train import CFG, regions_dir  # noqa: F401  (fixture)

WINDOWS = ((3, 9), (5, 17))


def oracle(gamma_true: float):
    """A stand-in 'network' that knows the true coherence and estimates the powers from the whole patch."""
    def model(x):
        c11 = x[:, 0].mean((1, 2), keepdim=True) - x[:, 4].mean((1, 2), keepdim=True)
        c22 = x[:, 1].mean((1, 2), keepdim=True) - x[:, 5].mean((1, 2), keepdim=True)
        u = torch.full_like(x[:, 0], float(np.arctanh(gamma_true))) if gamma_true else torch.zeros_like(x[:, 0])
        return torch.stack([torch.log(c11).expand_as(u), torch.log(c22).expand_as(u), u, torch.zeros_like(u)], dim=1)
    return model


def test_boxcar_keeps_constants_and_rejects_even_windows():
    x = torch.ones(1, 4, 16, 64) * torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1, 1)
    assert torch.allclose(boxcar(x, (5, 17)), x, atol=1e-6)
    with pytest.raises(ValueError):
        boxcar(x, (4, 9))


def test_covariance_to_raw_round_trips():
    raw = torch.randn(2, 4, 6, 6) * 0.8
    cov = covariance_stack(raw)
    back = covariance_stack(covariance_to_raw(cov))
    assert torch.allclose(back, cov, rtol=1e-3, atol=1e-4)


def test_oracle_with_true_coherence_beats_boxcar_and_gains_from_coherence(regions_dir):
    res = score_dataset(oracle(0.8), PatchDataset(regions_dir, "val", CFG), torch.device("cpu"), True, WINDOWS)
    r = res["coherent"]
    assert r["net"] < r["box5x17_full"] - 0.01                     # knowing the truth beats a 85-sample estimate
    assert r["coherence_gain"] > 0.3                                # switching coherence off must cost a lot
    assert r["box3x9_full"] < r["box3x9_diag"]                      # even the boxcar profits from the off-diagonal
    assert r["sign_agreement"] > 0.95 and r["phase_err_median"] < 0.35
    assert r["gamma_mean"] == pytest.approx(0.8, abs=0.01)


def test_no_coherence_is_not_rewarded_and_false_coherence_is_reported(regions_dir):
    res = score_dataset(oracle(0.0), PatchDataset(regions_dir, "val", CFG), torch.device("cpu"), True, WINDOWS)
    r = res["incoherent"]
    assert r["coherence_gain"] == pytest.approx(0.0, abs=1e-6)     # nothing to switch off
    assert r["gamma_mean"] < 0.01 and r["gamma_gt03"] == 0.0
    wrong = score_dataset(oracle(0.8), PatchDataset(regions_dir, "val", CFG), torch.device("cpu"), True, WINDOWS)
    assert wrong["incoherent"]["gamma_gt03"] == 1.0                 # a model that invents coherence is flagged
    assert wrong["incoherent"]["coherence_gain"] < -0.3             # and its coherence term HURTS the score


def test_report_and_cli(regions_dir, tmp_path, capsys):
    run = tmp_path / "run"
    train(regions_dir, run, steps=20, batch_size=4, base=8, depth=2, device="cpu", eval_every=20, patch_cfg=CFG,
          log=lambda *_: None)
    main(["--run", str(run), "--regions-dir", str(regions_dir), "--split", "val", "--patch-rows", "16",
          "--patch-cols", "64", "--windows", "3x9"])
    text = capsys.readouterr().out
    assert "coherent" in text and "sign agree" in text and "box3x9_full" in text and "calibration" in text and "rescaling test" in text
    rows = list(csv.DictReader(open(run / "eval_val.csv")))
    assert {r["class"] for r in rows} == {"coherent", "incoherent"} and (run / "eval_val_bins.csv").exists()
    res = score_dataset(oracle(0.8), PatchDataset(regions_dir, "val", CFG), torch.device("cpu"), True, WINDOWS)
    assert "coh gain" in format_report(res)


def test_calibration_bins_partition_the_pixels_and_show_the_oracle_is_calibrated(regions_dir):
    ds = PatchDataset(regions_dir, "val", CFG)
    res, bins = score_dataset(oracle(0.8), ds, torch.device("cpu"), True, WINDOWS, return_bins=True)
    assert len(bins) == len(BIN_EDGES) - 1
    total = sum(r["pixels"] for r in bins)
    assert total == 2 * sum(int(it["valid"].sum()) for it in (ds[i] for i in range(len(ds))))     # both directions, every pixel once
    strong = [r for r in bins if r["ref_lo"] >= 0.3 and r["pixels"] > 500]
    assert strong, "the coherent region must populate a high-reference bin"
    for r in strong:
        assert r["sign_agreement"] > 0.95 and r["coherence_gain"] > 0.2 and r["mean_pred"] == pytest.approx(0.8, abs=0.01)
    assert "calibration" in format_bins(bins)


def test_scale_gamma_changes_only_the_size_of_the_coherence():
    raw = torch.randn(1, 4, 5, 5)
    out = scale_gamma(raw, 0.5)
    _, _, _, _, g_re0, g_im0, g0 = raw_to_covariance(raw)
    _, _, _, _, g_re1, g_im1, g1 = raw_to_covariance(out)
    assert torch.allclose(g1, 0.5 * g0, atol=1e-4)
    assert torch.allclose(torch.atan2(g_im1, g_re1), torch.atan2(g_im0, g_re0), atol=1e-3)
    assert torch.equal(out[:, :2], raw[:, :2]) and torch.allclose(scale_gamma(raw, 0.0)[:, 2:].abs().max(), torch.tensor(0.0), atol=1e-3)


def test_rescaling_test_finds_the_right_size_and_flags_understated_coherence(regions_dir):
    ds = PatchDataset(regions_dir, "val", CFG, classes=["coherent"])      # the oracle's 0.8 is only right here
    ok = calibration_sweep(oracle(0.8), ds, torch.device("cpu"), True)
    row = [r for r in ok if r["pixels"] > 0][-1]
    assert row["best_k"] == 1.0 and row["gain_k1"] == pytest.approx(row["gain_best"], abs=1e-6)
    shy = calibration_sweep(oracle(0.4), ds, torch.device("cpu"), True)          # truth is 0.8, prediction half of it
    row = [r for r in shy if r["pixels"] > 0][-1]
    assert row["best_k"] == 2.0 and row["gain_best"] > row["gain_k1"] + 0.05
    assert "best k" in format_sweep(shy)