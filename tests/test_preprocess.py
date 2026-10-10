import numpy as np
import pytest

import sar_denoise.preprocess as pp
from sar_denoise.calibration import Lut
from sar_denoise.preprocess import edge_deviation_rows, process_arrays, process_burst
from sar_denoise.sublooks import SubLookConfig, hamming_weight
from tests.test_deramp import make_info
from tests.test_sublooks import ALPHA, B, DT, shaped_dualpol

C11, C22 = 4.0, 0.5
C12 = 0.6 * np.exp(0.7j) * np.sqrt(C11 * C22)
N, COLS = 1460, 400


def white_noise_in_band(n_lines, cols, power, rng):
    """Thermal-noise-like data: white inside the processed band, realised mean power = `power`."""
    z = rng.standard_normal((n_lines, cols)) + 1j * rng.standard_normal((n_lines, cols))
    f = np.fft.fftfreq(n_lines, d=DT)
    z = np.fft.ifft(np.fft.fft(z, axis=0) * (np.abs(f) <= B / 2)[:, None], axis=0)
    return z * np.sqrt(power / np.mean(np.abs(z) ** 2))


def scene(noise_vv=0.2, noise_vh=0.5, seed=0):
    sig_vv, sig_vh = shaped_dualpol(C11, C22, C12, n=N, cols=COLS, seed=seed)
    rng = np.random.default_rng(seed + 100)
    nz_vv = white_noise_in_band(N, COLS, noise_vv, rng)
    nz_vh = white_noise_in_band(N, COLS, noise_vh, rng)
    ones = np.ones((N, COLS), dtype=np.float32)
    return dict(sig=(sig_vv, sig_vh), vv=sig_vv + nz_vv, vh=sig_vh + nz_vh,
                a=ones, n_vv=ones * noise_vv, n_vh=ones * noise_vh)


def run(d, **kw):
    return process_arrays(d["vv"], d["vh"], d["a"], d["a"], d["n_vv"], d["n_vh"], DT, **kw)


def test_noise_aware_coherence_recovers_the_signal_coherence():
    d = scene()
    prod = run(d, trim_rows=0)
    sv, sh = d["sig"]
    true_gamma = abs(np.mean(sv * np.conj(sh))) / np.sqrt(np.mean(np.abs(sv) ** 2) * np.mean(np.abs(sh) ** 2))

    for c2, noise in ((prod.c2_a, prod.noise_a), (prod.c2_b, prod.noise_b)):
        c12 = np.mean(c2[2]) + 1j * np.mean(c2[3])
        raw = abs(c12) / np.sqrt(np.mean(c2[0]) * np.mean(c2[1]))
        fixed = abs(c12) / np.sqrt((np.mean(c2[0]) - np.mean(noise[0])) * (np.mean(c2[1]) - np.mean(noise[1])))
        assert raw < true_gamma - 0.1                       # thermal noise biases coherence low
        assert fixed == pytest.approx(true_gamma, abs=0.03)  # adding the known noise back removes the bias
        assert np.angle(c12 * np.exp(-0.7j)) == pytest.approx(0.0, abs=0.04)


def test_noise_map_matches_the_noise_in_a_noise_only_scene():
    rng = np.random.default_rng(5)
    nz_vv = white_noise_in_band(N, COLS, 0.3, rng)
    nz_vh = white_noise_in_band(N, COLS, 0.7, rng)
    ones = np.ones((N, COLS), dtype=np.float32)
    prod = process_arrays(nz_vv, nz_vh, ones, ones, ones * 0.3, ones * 0.7, DT, trim_rows=0)
    for i in (0, 1):
        assert np.mean(prod.c2_a[i]) == pytest.approx(np.mean(prod.noise_a[i]), rel=0.04)
        assert np.mean(prod.c2_b[i]) == pytest.approx(np.mean(prod.noise_b[i]), rel=0.04)


def test_looks_have_equal_mean_power_and_shapes_are_consistent():
    d = scene()
    prod = run(d, trim_rows=0)
    rows = N // 2
    assert prod.c2_a.shape == (4, rows, COLS) and prod.c2_b.shape == (4, rows, COLS)
    assert prod.noise_a.shape == (2, rows, COLS) and prod.pnr_db.shape == (2, rows, COLS)
    assert prod.valid.shape == (rows, COLS) and prod.valid.all()
    assert prod.c2_a.dtype == prod.c2_b.dtype == np.float32
    assert prod.noise_a.dtype == prod.pnr_db.dtype == np.float32
    p_full = np.mean(np.abs(d["vv"]) ** 2)
    assert np.mean(prod.c2_a[0]) == pytest.approx(p_full, rel=0.01)
    assert np.mean(prod.c2_b[0]) == pytest.approx(p_full, rel=0.01)
    assert prod.dt_out == pytest.approx(2 * DT)


def test_power_to_noise_map_matches_measured_power_over_noise():
    d = scene()
    prod = run(d, trim_rows=0)
    for i, (key, n) in enumerate((("vv", 0.2), ("vh", 0.5))):
        expected = 10 * np.log10(np.mean(np.abs(d[key]) ** 2) / n)       # measured power / noise power
        assert np.median(prod.pnr_db[i]) == pytest.approx(expected, abs=0.7)
    assert np.median(prod.pnr_db[0]) > np.median(prod.pnr_db[1]) + 6.0


def test_valid_mask_trims_the_ends_and_drops_all_zero_pixels():
    d = scene()
    d["vv"][100, 50] = 0
    d["vh"][100, 50] = 0
    d["vv"][200, 60] = 0                                 # only one channel zero: stays valid
    prod = run(d, trim_rows=10)
    assert not prod.valid[:10].any() and not prod.valid[-10:].any()
    assert not prod.valid[50, 50]                        # input row 100 -> look row 50
    assert prod.valid[100, 60]
    assert prod.valid[10:-10].mean() > 0.999


def test_inputs_must_have_matching_shapes():
    d = scene()
    with pytest.raises(ValueError):
        process_arrays(d["vv"], d["vh"][:-1], d["a"], d["a"], d["n_vv"], d["n_vh"], DT)


def test_meta_is_json_friendly():
    import json
    meta = run(scene(), trim_rows=0).meta
    json.dumps(meta)
    assert meta["decimation"] == 2 and meta["noise_model"] == "white"
    assert set(meta["gains"]) == {"a_vv", "a_vh", "b_vv", "b_vh"}


def test_edge_deviation_rows():
    profile = np.ones(400)
    profile[:12] = 2.0
    profile[-5:] = 0.5
    assert edge_deviation_rows(profile) == (12, 5)
    assert edge_deviation_rows(np.ones(100)) == (0, 0)


def test_process_burst_with_mocked_files(monkeypatch):
    info = make_info(n_lines=512, n_samples=300)
    sig_vv, sig_vh = shaped_dualpol(C11, C22, C12, n=512, cols=300, seed=3)
    dn = {"vv": (300.0 * sig_vv).astype(np.complex64), "vh": (300.0 * sig_vh).astype(np.complex64)}

    monkeypatch.setattr(pp, "find_swath_files", lambda safe, swath, pol: (pol, pol))
    monkeypatch.setattr(pp, "parse_annotation", lambda xml: info)
    monkeypatch.setattr(pp, "read_burst", lambda tiff, inf, k: dn[tiff])
    monkeypatch.setattr(pp, "deramp_burst", lambda data, inf, k, demodulate=True: data)   # data already baseband
    monkeypatch.setattr(pp, "find_calibration_files", lambda safe, swath, pol: ("cal", "noise"))
    const = lambda v: Lut(lines=np.array([0.0, 4000.0]), pixels=[np.array([0.0, 1000.0])] * 2,
                          values=[np.full(2, v)] * 2)
    monkeypatch.setattr(pp, "load_calibration_lut", lambda path, kind: const(300.0))
    monkeypatch.setattr(pp, "load_noise_lut", lambda path: const(50.0))

    prod = process_burst("fake/PRODUCT.SAFE", "iw2", 0, cols=(10, 110), trim_rows=8)
    assert prod.c2_a.shape == (4, 256, 100) and prod.noise_a.shape == (2, 256, 100)
    # calibration divides by 300, so sigma units are recovered
    assert np.mean(prod.c2_a[0]) == pytest.approx(np.mean(np.abs(sig_vv[:, 10:110]) ** 2), rel=0.05)
    assert np.mean(prod.noise_a[0]) == pytest.approx(50.0 / 300.0 ** 2, rel=0.1)
    assert prod.meta["product"] == "PRODUCT.SAFE" and prod.meta["samples"] == [10, 110]
    assert prod.meta["deramp_vv_vs_vh_mismatch"]["f_dc_hz"] == 0.0
    with pytest.raises(ValueError):
        process_burst("fake/PRODUCT.SAFE", "iw2", 0, cols=(400, 500))