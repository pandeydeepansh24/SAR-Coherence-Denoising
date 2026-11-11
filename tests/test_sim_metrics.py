import numpy as np

from sar_denoise.sim_metrics import (complex_ssim, enl, gamma_bins, phase_error_deg, point_metrics, psnr_db,
                                     ring_mask)


def test_enl_of_exponential_speckle_is_one_and_of_constant_is_inf():
    rng = np.random.default_rng(0)
    i = rng.exponential(2.0, size=(200, 200))
    m = np.ones_like(i, bool)
    assert abs(enl(i, m) - 1) < 0.05
    assert abs(enl(rng.gamma(9, 1 / 9, size=(200, 200)), m) - 9) < 0.5
    assert enl(np.ones((5, 5)), np.ones((5, 5), bool)) == float("inf")


def test_psnr_increases_with_accuracy():
    t = np.linspace(0.01, 1, 400).reshape(20, 20)
    assert psnr_db(t * 1.01, t) > psnr_db(t * 1.5, t)


def test_complex_ssim_one_at_truth_and_sensitive_to_phase():
    rng = np.random.default_rng(1)
    g = 0.7 * np.exp(1j * rng.uniform(-np.pi, np.pi)) * np.ones((32, 32))
    assert abs(complex_ssim(g, g) - 1) < 1e-12
    # property of the formula: |sum x y*| ignores a CONSTANT phase offset (that is why phase_error_deg is reported too)
    assert abs(complex_ssim(g * np.exp(1j * np.pi / 2), g) - 1) < 1e-12
    # ... but a phase error that varies within the window is penalised
    noisy_phase = g * np.exp(1j * rng.uniform(-np.pi, np.pi, size=g.shape))
    assert complex_ssim(noisy_phase, g) < 0.3
    assert complex_ssim(0.5 * g, g) < 1


def test_gamma_bins_and_phase_error():
    gt = np.array([0.05, 0.15, 0.55, 0.55]) * np.exp(1j * 0.3)
    ge = np.array([0.15, 0.15, 0.45, 0.65]) * np.exp(1j * 0.4)
    rows = gamma_bins(ge, gt)
    assert [r["n"] for r in rows] == [1, 1, 2]
    assert abs(rows[0]["bias"] - 0.1) < 1e-9 and abs(rows[2]["bias"]) < 1e-9 and abs(rows[2]["rmse"] - 0.1) < 1e-9
    assert abs(phase_error_deg(ge, gt) - np.degrees(0.1)) < 1e-6


def test_point_metrics_counts_bright_ring_pixels():
    truth = np.ones((40, 40))
    truth[20, 20] = 100.0
    est = truth.copy()
    est[20, 24] = 5.0          # inside the ring, > 3 x background
    est[20, 21] = 9.0          # distance 1: inside r_in, ignored
    (row,) = point_metrics(est, truth, [(20, 20)])
    assert row["bright_px"] == 1 and abs(row["peak_db"] - 20) < 1e-9 and row["ring_px"] == ring_mask((40, 40), 20, 20).sum()
