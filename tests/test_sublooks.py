from types import SimpleNamespace

import numpy as np
import pytest

from sar_denoise.simulate import simulate_dualpol_slc
from sar_denoise.sublooks import (
    SubLookConfig,
    band_edges,
    c12_speckle_correlation,
    complex_correlation,
    dark_column_spectrum,
    edge_centre_db,
    hamming_weight,
    look_filters,
    noise_factors,
    speckle_intensity_correlation,
    split_sublooks,
)

DT = 2.055556e-3
B, ALPHA = 313.0, 0.75
N, COLS = 1460, 600


def shaped_dualpol(c11, c22, c12, n=N, cols=COLS, seed=0):
    """VV/VH speckle with covariance C and the S-1 Hamming azimuth spectrum (same filter for both)."""
    rng = np.random.default_rng(seed)
    vv, vh = simulate_dualpol_slc(c11, c22, c12, (n, cols), rng)
    w = hamming_weight(np.fft.fftfreq(n, d=DT), B, ALPHA)[:, None]
    shape = lambda x: np.fft.ifft(np.fft.fft(x, axis=0) * w, axis=0)
    return shape(vv), shape(vh)


def power_spectrum(x):
    return np.fft.fftshift(np.mean(np.abs(np.fft.fft(x, axis=0)) ** 2, axis=1))


def test_hamming_weight_shape():
    f = np.array([0.0, 156.5, -156.5, 200.0])
    w = hamming_weight(f, B, ALPHA)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5, abs=1e-3) and w[2] == pytest.approx(0.5, abs=1e-3)
    assert w[3] == 0.0


def test_bands_are_separated_by_the_guard():
    (lo_a, hi_a), (lo_b, hi_b) = band_edges(SubLookConfig(guard_hz=10.0, split_hz=2.0))
    assert lo_a == pytest.approx(-156.5) and hi_b == pytest.approx(156.5)
    assert lo_b - hi_a == pytest.approx(10.0)
    assert (hi_a + lo_b) / 2 == pytest.approx(2.0)


def test_looks_keep_the_input_power_and_are_flat_with_deweighting():
    vv, vh = shaped_dualpol(4.0, 0.5, 0.6 * np.exp(0.7j) * np.sqrt(2.0))
    sl = split_sublooks(vv, vh, DT, SubLookConfig(deweight=True))
    p_in = np.mean(np.abs(vv) ** 2)
    assert np.mean(np.abs(sl.a_vv) ** 2) == pytest.approx(p_in, rel=0.03)
    assert np.mean(np.abs(sl.b_vv) ** 2) == pytest.approx(p_in, rel=0.03)

    spec = power_spectrum(sl.a_vv)
    f = np.fft.fftshift(np.fft.fftfreq(sl.a_vv.shape[0], d=sl.dt_out))
    half_width = (band_edges(sl.cfg)[0][1] - band_edges(sl.cfg)[0][0]) / 2
    inner = spec[np.abs(f) < 0.2 * half_width].mean()
    outer = spec[(np.abs(f) > 0.6 * half_width) & (np.abs(f) < 0.9 * half_width)].mean()
    assert 0.85 < outer / inner < 1.15


def test_without_deweighting_the_looks_keep_the_sloped_spectrum():
    vv, vh = shaped_dualpol(4.0, 0.5, 0.6 * np.exp(0.7j) * np.sqrt(2.0))
    sl = split_sublooks(vv, vh, DT, SubLookConfig(deweight=False))
    spec = power_spectrum(sl.b_vv)            # look B: strongest at its low-frequency (inner) edge
    f = np.fft.fftshift(np.fft.fftfreq(sl.b_vv.shape[0], d=sl.dt_out))
    half_width = (band_edges(sl.cfg)[1][1] - band_edges(sl.cfg)[1][0]) / 2
    inner = spec[f < -0.6 * half_width].mean()
    outer = spec[f > 0.6 * half_width].mean()
    assert outer / inner < 0.6


def test_looks_are_uncorrelated():
    vv, vh = shaped_dualpol(4.0, 0.5, 0.6 * np.exp(0.7j) * np.sqrt(2.0), seed=3)
    sl = split_sublooks(vv, vh, DT)
    assert abs(complex_correlation(sl.a_vv, sl.b_vv)) < 0.02
    assert abs(complex_correlation(sl.a_vh, sl.b_vh)) < 0.02
    assert abs(speckle_intensity_correlation(sl.a_vv, sl.b_vv)) < 0.03


def test_covariance_is_preserved_in_each_look():
    c12 = 0.6 * np.exp(0.7j) * np.sqrt(4.0 * 0.5)
    vv, vh = shaped_dualpol(4.0, 0.5, c12, seed=5)
    sl = split_sublooks(vv, vh, DT)
    c_in = np.mean(vv * np.conj(vh))
    for a, b in ((sl.a_vv, sl.a_vh), (sl.b_vv, sl.b_vh)):
        c_look = np.mean(a * np.conj(b))
        assert abs(c_look) == pytest.approx(abs(c_in), rel=0.06)
        assert np.angle(c_look * np.conj(c_in)) == pytest.approx(0.0, abs=0.04)
        assert np.mean(np.abs(a) ** 2) == pytest.approx(np.mean(np.abs(vv) ** 2), rel=0.04)
        assert np.mean(np.abs(b) ** 2) == pytest.approx(np.mean(np.abs(vh) ** 2), rel=0.04)


def test_a_tone_ends_up_in_the_right_look():
    t = np.arange(N) * DT
    tone = np.tile(np.exp(2j * np.pi * (-100.0) * t)[:, None], (1, 8))
    sl = split_sublooks(tone, tone, DT, SubLookConfig(equalize=False))
    assert np.mean(np.abs(sl.a_vv) ** 2) > 1e3 * np.mean(np.abs(sl.b_vv) ** 2)


def test_shapes_timing_and_chunk_invariance():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1461, 70)) + 1j * rng.standard_normal((1461, 70))
    y = rng.standard_normal((1461, 70)) + 1j * rng.standard_normal((1461, 70))
    s1 = split_sublooks(x, y, DT, col_chunk=16)
    s2 = split_sublooks(x, y, DT, col_chunk=70)
    assert s1.a_vv.shape == (730, 70) and s1.n_lines_used == 1460 and s1.decimation == 2
    assert s1.dt_out == pytest.approx(2 * DT)
    for k in ("a_vv", "a_vh", "b_vv", "b_vh"):
        np.testing.assert_allclose(getattr(s1, k), getattr(s2, k), rtol=1e-4, atol=1e-4)
    assert s1.a_vv.dtype == np.complex64


def test_too_much_decimation_is_rejected():
    x = np.ones((256, 4), dtype=complex)
    with pytest.raises(ValueError, match="alias"):
        split_sublooks(x, x, DT, SubLookConfig(decimate=4))


def test_input_validation():
    with pytest.raises(ValueError):
        split_sublooks(np.ones((256, 4)), np.ones((256, 4)), DT)
    with pytest.raises(ValueError):
        split_sublooks(np.ones((256, 4), dtype=complex), np.ones((256, 5), dtype=complex), DT)


def test_config_from_annotation():
    info = SimpleNamespace(processing_bandwidth=313.0, window_coefficient=0.75, window_type="Hamming")
    cfg = SubLookConfig.from_info(info, guard_hz=20.0)
    assert (cfg.bandwidth_hz, cfg.window_coefficient, cfg.guard_hz) == (313.0, 0.75, 20.0)
    assert cfg.deweight is False and cfg.equalize is True            # defaults
    assert SubLookConfig.from_info(info, deweight=True).deweight is True
    other = SimpleNamespace(processing_bandwidth=313.0, window_coefficient=None, window_type="KAISER")
    assert SubLookConfig.from_info(other).deweight is False


def test_filters_have_unit_radiometric_gain_on_white_input():
    f = np.fft.fftfreq(N, d=DT)
    w = hamming_weight(f, B, ALPHA)
    for filt, _ in look_filters(f, SubLookConfig()):
        assert np.sum((filt * w) ** 2) == pytest.approx(np.sum(w ** 2))


def test_noise_factors_window_model_is_unity_and_white_is_larger():
    wa, wb = noise_factors(N, DT, SubLookConfig(), "window")
    assert wa == pytest.approx(1.0, rel=1e-6) and wb == pytest.approx(1.0, rel=1e-6)
    ha, hb = noise_factors(N, DT, SubLookConfig(deweight=True), "white")
    # mean(w^2) * mean(1/w^2) = 0.594 * 2.121 = 1.26 for Hamming 0.75
    assert 1.2 < ha < 1.32 and 1.2 < hb < 1.32
    # without de-weighting, each look keeps about its share of white noise: factor close to 1
    na, nb = noise_factors(N, DT, SubLookConfig(deweight=False), "white")
    assert 0.95 < na < 1.1 and 0.95 < nb < 1.1
    with pytest.raises(ValueError):
        noise_factors(N, DT, SubLookConfig(), "pink")


def test_noise_factors_match_what_split_sublooks_does_to_noise():
    rng = np.random.default_rng(11)
    shape = (N, 600)
    white = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    f = np.fft.fftfreq(N, d=DT)
    inband = (np.abs(f) <= B / 2)[:, None]
    noise = np.fft.ifft(np.fft.fft(white, axis=0) * inband, axis=0)       # white inside the band
    fa, fb = noise_factors(N, DT, SubLookConfig(), "white")
    p_in = np.mean(np.abs(noise) ** 2)
    sl = split_sublooks(noise, noise, DT, SubLookConfig(equalize=False))
    assert np.mean(np.abs(sl.a_vv) ** 2) / p_in == pytest.approx(fa, rel=0.04)
    assert np.mean(np.abs(sl.b_vv) ** 2) / p_in == pytest.approx(fb, rel=0.04)


def test_dark_column_spectrum_distinguishes_white_from_window_shaped_noise():
    rng = np.random.default_rng(2)
    f = np.fft.fftfreq(N, d=DT)
    inband = (np.abs(f) <= B / 2)[:, None]
    w = hamming_weight(f, B, ALPHA)[:, None]
    cn = lambda cols: rng.standard_normal((N, cols)) + 1j * rng.standard_normal((N, cols))
    shaped = lambda cols, filt: np.fft.ifft(np.fft.fft(cn(cols), axis=0) * filt, axis=0)

    for filt, lo, hi in ((w, -7.0, -4.5), (inband.astype(float), -1.0, 1.0)):
        freq, power = dark_column_spectrum(shaped(300, filt), DT, fraction=0.5)
        assert lo < edge_centre_db(freq, power, B) < hi

    # the darkest columns are the ones that are analysed: bright window-shaped + dark white
    mixed = np.concatenate([10.0 * shaped(200, w), shaped(200, inband.astype(float))], axis=1)
    freq, power = dark_column_spectrum(mixed, DT, fraction=0.4)
    assert -1.0 < edge_centre_db(freq, power, B) < 1.0


def _tilted(c11, c22, c12, tilt, seed=0):
    """Shaped dual-pol data whose power spectrum is tilted: PSD ~ w^2 * (1 + tilt * f / (B/2))."""
    rng = np.random.default_rng(seed)
    vv, vh = simulate_dualpol_slc(c11, c22, c12, (N, COLS), rng)
    f = np.fft.fftfreq(N, d=DT)
    amp = (hamming_weight(f, B, ALPHA) * np.sqrt(1.0 + tilt * f / (B / 2)))[:, None]
    shape = lambda x: np.fft.ifft(np.fft.fft(x, axis=0) * amp, axis=0)
    return shape(vv), shape(vh)


def test_equalisation_removes_a_spectral_tilt():
    vv, vh = _tilted(4.0, 0.5, 0.6 * np.exp(0.7j) * np.sqrt(2.0), tilt=0.2)
    raw = split_sublooks(vv, vh, DT, SubLookConfig(equalize=False))
    eq = split_sublooks(vv, vh, DT, SubLookConfig(equalize=True))
    db = lambda a, b: 10 * np.log10(np.mean(np.abs(a) ** 2) / np.mean(np.abs(b) ** 2))
    assert abs(db(raw.a_vv, raw.b_vv)) > 0.5                       # the tilt shows up as an A/B offset
    assert abs(db(eq.a_vv, eq.b_vv)) < 0.02                        # and is gone after equalisation
    p_in = np.mean(np.abs(vv) ** 2)
    assert np.mean(np.abs(eq.a_vv) ** 2) == pytest.approx(p_in, rel=0.005)
    assert eq.gains["a_vv"] > 1.0 > eq.gains["b_vv"]
    assert raw.gains == {k: 1.0 for k in raw.gains}


def test_equalisation_leaves_coherence_and_phase_unchanged():
    c12 = 0.6 * np.exp(0.7j) * np.sqrt(4.0 * 0.5)
    vv, vh = _tilted(4.0, 0.5, c12, tilt=0.2, seed=1)
    raw = split_sublooks(vv, vh, DT, SubLookConfig(equalize=False))
    eq = split_sublooks(vv, vh, DT, SubLookConfig(equalize=True))

    def gamma_phase(a, b):
        c = np.mean(a * np.conj(b))
        return abs(c) / np.sqrt(np.mean(np.abs(a) ** 2) * np.mean(np.abs(b) ** 2)), np.angle(c)

    for look in ("a", "b"):
        g0, p0 = gamma_phase(getattr(raw, f"{look}_vv"), getattr(raw, f"{look}_vh"))
        g1, p1 = gamma_phase(getattr(eq, f"{look}_vv"), getattr(eq, f"{look}_vh"))
        assert g1 == pytest.approx(g0, abs=1e-4) and p1 == pytest.approx(p0, abs=1e-4)


def test_absurd_equalisation_gain_is_refused():
    t = np.arange(N) * DT
    tone = np.tile(np.exp(2j * np.pi * (-100.0) * t)[:, None], (1, 8))
    with pytest.warns(UserWarning, match="equalisation gain"):
        sl = split_sublooks(tone, tone, DT, SubLookConfig(equalize=True))
    assert sl.gains["b_vv"] == 1.0
    assert np.mean(np.abs(sl.a_vv) ** 2) > 1e3 * np.mean(np.abs(sl.b_vv) ** 2)


def test_c12_speckle_correlation_detects_shared_speckle():
    c12 = 0.6 * np.exp(0.7j) * np.sqrt(2.0)
    vv, vh = shaped_dualpol(4.0, 0.5, c12, seed=7)
    ok = split_sublooks(vv, vh, DT, SubLookConfig(guard_hz=10.0))
    assert c12_speckle_correlation(ok.a_vv, ok.a_vh, ok.b_vv, ok.b_vh) < 0.03
    overlap = split_sublooks(vv, vh, DT, SubLookConfig(guard_hz=-80.0))     # the bands share 80 Hz
    assert c12_speckle_correlation(overlap.a_vv, overlap.a_vh, overlap.b_vv, overlap.b_vh) > 0.15