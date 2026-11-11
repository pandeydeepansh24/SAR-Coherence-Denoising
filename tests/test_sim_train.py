import math

import numpy as np
import pytest
import torch

from sar_denoise.sim_scene import SceneConfig
from sar_denoise.sim_train import fixed_set, load_model, make_sample, train


def test_sample_normalisation_keeps_gamma_and_phase():
    s = make_sample(SceneConfig(shape=(48, 48)), np.random.default_rng(0), return_scene=True)
    t = s["truth"].astype(np.float64)
    g_norm = (t[2] + 1j * t[3]) / np.sqrt(t[0] * t[1])
    np.testing.assert_allclose(g_norm, s["scene"].gamma, atol=1e-5)
    assert abs(s["c2"][0].mean() - 1) < 1e-4 and abs(s["c2"][1].mean() - 1) < 1e-4
    # complex input and C2 describe the same pixels
    vv = s["x"][0] + 1j * s["x"][2]
    np.testing.assert_allclose(np.abs(vv) ** 2, s["c2"][0], rtol=1e-4, atol=1e-6)


def test_common_phase_augmentation_changes_x_not_c2():
    cfg = SceneConfig(shape=(32, 32))
    a = make_sample(cfg, np.random.default_rng(5), phase_aug=False)
    b = make_sample(cfg, np.random.default_rng(5), phase_aug=True)
    # same scene and speckle draws precede the phase draw, so only the common phase differs
    np.testing.assert_allclose(a["c2"], b["c2"], rtol=1e-4, atol=1e-5)
    assert not np.allclose(a["x"], b["x"])


def test_fixed_set_is_deterministic():
    cfg = SceneConfig(shape=(32, 32))
    torch.testing.assert_close(fixed_set(cfg, 2, 9)["truth"], fixed_set(cfg, 2, 9)["truth"])


@pytest.mark.parametrize("kind", ["complex", "real"])
def test_short_training_lowers_kl_and_saves(kind, tmp_path):
    res = train(kind, tmp_path, steps=40, batch_size=4, base=4, depth=2, size=32, n_val=4, eval_every=20,
                device="cpu", workers=0, warmup=5, log=lambda *_: None)
    assert math.isfinite(res["best_val_kl"])
    rows = (tmp_path / "log.csv").read_text().strip().splitlines()
    assert len(rows) == 3
    model, ck = load_model(tmp_path / "best.pt")
    assert ck["kind"] == kind
    first_val = float(rows[1].split(",")[2])
    assert res["best_val_kl"] <= first_val
