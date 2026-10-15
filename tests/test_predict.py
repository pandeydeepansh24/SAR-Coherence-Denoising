import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from sar_denoise.patches import PatchConfig, save_region
from sar_denoise.predict import predict_region, reference_gamma, tile_origins
from sar_denoise.simulate import simulate_look_pair
from sar_denoise.train import train
from tests.test_evaluate import oracle
from tests.test_train import CFG, regions_dir  # noqa: F401

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict_region.py"


def region(gamma, h=40, w=150, seed=0):
    pair = simulate_look_pair(1.0, 0.4, gamma * np.sqrt(0.4), 0.05, 0.3, shape=(h, w), rng=np.random.default_rng(seed))
    pair["pnr_db"] = np.full((2, h, w), 10.0, np.float32)
    pair["gamma"] = np.full((h, w), abs(gamma), np.float32)
    pair["valid"] = np.ones((h, w), bool)
    return pair


def test_tiles_cover_every_pixel_even_when_the_size_is_not_a_multiple_of_the_stride():
    cfg = PatchConfig(rows=16, cols=64, stride_rows=16, stride_cols=64)
    cover = np.zeros((40, 150), int)
    for r, c in tile_origins(cover.shape, cfg):
        cover[r:r + 16, c:c + 64] += 1
    assert cover.min() >= 1
    with pytest.raises(ValueError):
        tile_origins((10, 150), cfg)


def test_predict_region_stitches_a_full_map_with_the_right_values():
    arrays = region(0.8)
    out = predict_region(oracle(0.8), arrays, CFG, "cpu", True, look="a")
    assert out["g_abs"].shape == (40, 150) and out["count"].min() >= 1
    assert out["g_abs"].mean() == pytest.approx(0.8, abs=0.01) and out["g_im"].std() < 1e-6
    assert out["c11"].mean() == pytest.approx(1.05, rel=0.2)          # powers are back in archive units (signal + ~0 noise)


def test_reference_gamma_sees_the_sign_and_magnitude_of_coherence():
    pos = reference_gamma(region(0.8), "b", (5, 17))
    neg = reference_gamma(region(-0.8, seed=1), "b", (5, 17))
    assert pos["g_re"].mean() > 0.3 and neg["g_re"].mean() < -0.3
    assert pos["g_abs"].mean() == pytest.approx(neg["g_abs"].mean(), abs=0.08)


def test_script_runs_end_to_end_on_a_region_archive(regions_dir, tmp_path, capsys):
    run = tmp_path / "run"
    train(regions_dir, run, steps=15, batch_size=4, base=8, depth=2, device="cpu", eval_every=15, patch_cfg=CFG,
          log=lambda *_: None)
    spec = importlib.util.spec_from_file_location("predict_region_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    png = tmp_path / "p.png"
    mod.main([str(regions_dir / "va_pos.npz"), "--run", str(run), "--out", str(png), "--patch-rows", "16",
              "--patch-cols", "64"])
    text = capsys.readouterr().out
    assert png.exists() and png.stat().st_size > 5000 and "sign agreement" in text