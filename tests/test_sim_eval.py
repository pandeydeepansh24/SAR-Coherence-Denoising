import json

import numpy as np

from sar_denoise.sim_eval import estimates, run, scene_metrics, showcase_scene
from sar_denoise.sim_train import train


def test_baselines_behave_as_expected_on_showcase_scene():
    scene, (s_vv, s_vh) = showcase_scene(256)
    m = scene_metrics(scene, estimates(scene, s_vv, s_vh, {}, "cpu"))
    assert 0.7 < m["noisy"]["enl"] < 1.4                 # single look
    assert m["boxcar"]["enl"] > 10                         # ~25 looks minus correlation
    assert m["boxcar"]["psnr_db"] > m["noisy"]["psnr_db"]
    assert m["boxcar"]["cssim"] > m["noisy"]["cssim"]
    noisy_bright = np.mean([p["bright_px"] for p in m["noisy"]["points"]])
    assert noisy_bright > 3                                # speckle spots around the scatterer


def test_eval_run_writes_figures(tmp_path):
    for kind in ("complex", "real"):
        train(kind, tmp_path / kind, steps=2, batch_size=2, base=4, depth=2, size=32, n_val=2, eval_every=2,
              device="cpu", workers=0, warmup=1, log=lambda *_: None)
    out = tmp_path / "fig"
    res = run({"complex": tmp_path / "complex" / "best.pt", "real": tmp_path / "real" / "best.pt"}, out,
              n_test=2, size=96, device="cpu", log=lambda *_: None)
    for f in ("showcase_main.png", "showcase_all.png", "point_zoom.png", "gamma_bias.png", "metrics.md", "metrics.json"):
        assert (out / f).stat().st_size > 0
    assert set(res["test"]) >= {"noisy", "boxcar", "complex", "real"}
    json.loads((out / "metrics.json").read_text())
