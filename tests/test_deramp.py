import numpy as np
import pytest

from sar_denoise.deramp import (
    antenna_doppler_rate,
    burst_mid_time,
    deramp_burst,
    deramp_params,
    deramp_phase,
    mean_azimuth_spectrum,
    spacecraft_speed,
    spectrum_stats,
)
from sar_denoise.safe_io import Burst, SwathInfo, TimedPoly

# numbers from ESA's worked example (COPE-GSEG-EOPG-TN-14-0025, section 6)
FC = 5.405000454334350e9
STEERING_DEG_S = 1.590368784
FM_POLY = np.array([-2.311921087579409e03, 4.483154182859674e05, 7.926006904805526e07])
DC_POLY = np.array([3.486574e01, -2.299984e04, 8.332020e06])
TAU0 = 5.356509155649830e-03
DT = 2.055556280538440e-03


def make_info(n_lines=512, n_samples=300):
    t0 = np.datetime64("2018-01-25T14:16:04.000000")
    burst = Burst(index=0, line_start=0, n_lines=n_lines, n_samples=n_samples, azimuth_time=t0,
                  first_valid=np.zeros(n_lines, dtype=int),
                  last_valid=np.full(n_lines, n_samples - 1, dtype=int))
    orbit_t = t0 + np.arange(-2, 4) * np.timedelta64(1, "s")
    orbit_v = 7589.75 - 0.0185 * np.arange(-2, 4)
    return SwathInfo(
        swath="IW2", polarisation="VV", n_lines=n_lines, n_samples=n_samples,
        lines_per_burst=n_lines, samples_per_burst=n_samples, azimuth_time_interval=DT,
        range_pixel_spacing=2.33, azimuth_pixel_spacing=13.9, radar_frequency=FC,
        azimuth_steering_rate=STEERING_DEG_S, processing_bandwidth=313.0,
        window_type="Hamming", window_coefficient=0.75, bursts=[burst],
        fm_rates=[TimedPoly(t0, TAU0, FM_POLY)], dc_estimates=[TimedPoly(t0, TAU0, DC_POLY)],
        slant_range_time=TAU0, range_sampling_rate=6.434523812571428e07,
        orbit_times=orbit_t, orbit_speeds=orbit_v,
    )


def test_antenna_doppler_rate_matches_esa_example():
    assert antenna_doppler_rate(7589.7505, FC, STEERING_DEG_S) == pytest.approx(7596.3984, rel=1e-6)


def test_parameters_at_first_range_sample():
    info = make_info()
    p = deramp_params(info, 0)
    ks = antenna_doppler_rate(p.v_s, FC, STEERING_DEG_S)
    assert p.k_s == pytest.approx(ks)
    assert p.k_a[0] == pytest.approx(FM_POLY[0], rel=1e-12)
    assert p.f_dc[0] == pytest.approx(DC_POLY[0], rel=1e-12)
    assert p.k_t[0] == pytest.approx(FM_POLY[0] * ks / (FM_POLY[0] - ks), rel=1e-12)
    assert p.k_t[0] == pytest.approx(1772.5, rel=1e-3)      # hand value for the ESA example
    assert np.all(p.k_a < 0) and np.all(p.k_t > 0)


def test_reference_time_is_zero_at_mid_swath():
    info = make_info()
    p = deramp_params(info, 0)
    assert p.eta_ref[info.n_samples // 2] == pytest.approx(0.0, abs=1e-15)


def test_mid_burst_time_and_speed_interpolation():
    info = make_info()
    t_mid = burst_mid_time(info, info.bursts[0])
    expected = info.bursts[0].azimuth_time + np.timedelta64(int(round(DT * 512 / 2 * 1e9)), "ns")
    assert t_mid == expected
    v = spacecraft_speed(info, t_mid)
    assert 7589.6 < v < 7589.8


def test_deramp_cancels_the_modelled_chirp_with_demodulation():
    info = make_info()
    p = deramp_params(info, 0)
    eta = (np.arange(512) - 256.0) * DT
    chirp = np.exp(-1j * deramp_phase(p, eta, demodulate=True)).astype(np.complex64)
    out = deramp_burst(chirp, info, 0, demodulate=True)
    assert np.allclose(out, 1.0, atol=1e-5)


def test_deramp_without_demodulation_removes_only_the_chirp():
    info = make_info()
    p = deramp_params(info, 0)
    eta = (np.arange(512) - 256.0) * DT
    chirp = np.exp(-1j * deramp_phase(p, eta, demodulate=False)).astype(np.complex64)
    out = deramp_burst(chirp, info, 0, demodulate=False)
    assert np.allclose(out, 1.0, atol=1e-5)


def test_deramp_rejects_wrong_shape():
    info = make_info()
    with pytest.raises(ValueError):
        deramp_burst(np.ones((10, 10), dtype=np.complex64), info, 0)


def test_spectrum_stats_locates_a_tone():
    dt, n, f0 = 0.002, 1000, 40.0
    eta = np.arange(n) * dt
    block = np.tile(np.exp(2j * np.pi * f0 * eta)[:, None], (1, 8)).astype(np.complex64)
    f, pw = mean_azimuth_spectrum(block, dt, column_step=1)
    assert spectrum_stats(f, pw)["centroid_hz"] == pytest.approx(f0, abs=1.0)


def test_deramping_narrows_a_tops_modulated_spectrum():
    """Band-limited noise multiplied by the TOPS chirp looks wideband; deramping restores it."""
    n_lines, n_cols = 1509, 64
    info = make_info(n_lines=n_lines, n_samples=n_cols)
    rng = np.random.default_rng(0)

    # complex noise limited to |f| < 80 Hz (about half the S-1 processed bandwidth)
    noise = rng.standard_normal((n_lines, n_cols)) + 1j * rng.standard_normal((n_lines, n_cols))
    spec = np.fft.fft(noise, axis=0)
    freq = np.fft.fftfreq(n_lines, d=DT)
    spec[np.abs(freq) > 80.0] = 0.0
    base = np.fft.ifft(spec, axis=0).astype(np.complex64)

    p = deramp_params(info, 0)
    eta = (np.arange(n_lines) - n_lines / 2.0) * DT
    raw = (base * np.exp(-1j * deramp_phase(p, eta, demodulate=True))).astype(np.complex64)
    fixed = deramp_burst(raw, info, 0, demodulate=True)

    raw_w = spectrum_stats(*mean_azimuth_spectrum(raw, DT, 1))["width_hz"]
    fixed_stats = spectrum_stats(*mean_azimuth_spectrum(fixed, DT, 1))
    assert raw_w > 350.0                          # smeared over most of the 486 Hz sampled band
    assert fixed_stats["width_hz"] < 160.0        # back to roughly the 160 Hz noise band
    assert abs(fixed_stats["centroid_hz"]) < 5.0