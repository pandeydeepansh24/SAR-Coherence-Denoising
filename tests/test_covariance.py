import numpy as np
import pytest

from sar_denoise.covariance import coherence, form_c2, multilook, psd_violation
from sar_denoise.simulate import simulate_dualpol_slc

C11, C22 = 4.0, 0.5
GAMMA = 0.6 * np.exp(1j * 0.7)
C12 = GAMMA * np.sqrt(C11 * C22)
SHAPE = (400, 1000)  # 400k pixels


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return simulate_dualpol_slc(C11, C22, C12, SHAPE, rng)


def test_mean_of_single_look_c2_equals_true_covariance(data):
    c2 = form_c2(*data)
    m = c2.reshape(4, -1).mean(axis=1)
    assert m[0] == pytest.approx(C11, rel=0.02)
    assert m[1] == pytest.approx(C22, rel=0.02)
    assert m[2] == pytest.approx(C12.real, abs=0.03)
    assert m[3] == pytest.approx(C12.imag, abs=0.03)


def test_single_look_c2_is_rank_one_so_coherence_is_one(data):
    c2 = form_c2(*data)
    gamma, _ = coherence(c2)
    ok = (c2[0] * c2[1]) > 1e-6          # ignore numerically tiny pixels
    assert np.allclose(gamma[ok], 1.0, atol=1e-3)


def test_multilooked_coherence_recovers_truth(data):
    c2 = form_c2(*data).reshape(4, -1).mean(axis=1)[:, None]
    gamma, phase = coherence(c2)
    assert gamma[0] == pytest.approx(0.6, abs=0.02)
    assert phase[0] == pytest.approx(0.7, abs=0.03)


def test_common_phase_leaves_c2_unchanged(data):
    rng = np.random.default_rng(1)
    ramp = np.exp(1j * rng.uniform(-np.pi, np.pi, SHAPE))
    a = form_c2(*data)
    b = form_c2(data[0] * ramp, data[1] * ramp)
    assert np.allclose(a, b, rtol=1e-3, atol=1e-3)


def test_relative_phase_rotates_c12_phase(data):
    theta = 0.4
    c2 = form_c2(data[0], data[1] * np.exp(1j * theta)).reshape(4, -1).mean(axis=1)[:, None]
    _, phase = coherence(c2)
    assert phase[0] == pytest.approx(0.7 - theta, abs=0.03)


def test_multilooking_keeps_matrix_positive_semidefinite(data):
    c2 = form_c2(*data)
    ml = multilook(c2, (5, 5))
    scale = ml[0] * ml[1]
    assert np.all(psd_violation(ml) <= 1e-4 * scale + 1e-6)


def test_simulator_rejects_non_psd_covariance():
    with pytest.raises(ValueError):
        simulate_dualpol_slc(1.0, 1.0, 1.5, (4, 4))


def test_form_c2_rejects_real_input():
    with pytest.raises(ValueError):
        form_c2(np.ones((3, 3)), np.ones((3, 3)))