import numpy as np
import pytest

from sar_denoise.sim_scene import (POINT_GAMMA, SceneConfig, check_truth, ensemble_covariance, homogeneous_mask,
                                   make_scene, observe)


@pytest.fixture(scope="module")
def scene():
    return make_scene(SceneConfig(shape=(96, 96)), np.random.default_rng(1), n_points=3)


def test_truth_is_positive_definite_and_in_design_ranges(scene):
    g = np.abs(scene.gamma)
    assert np.all(scene.c11 > 0) and np.all(scene.c22 > 0)
    assert g.max() <= POINT_GAMMA + 1e-9          # strictly < 1 -> det C > 0 everywhere
    det = scene.c11 * scene.c22 - np.abs(scene.c12) ** 2
    assert np.all(det > 0)
    fields = scene.labels >= 0
    assert np.all(g[fields & (np.abs(scene.a_vv) == 0)] <= 0.9 + 1e-9)


def test_points_are_bright_and_coherent(scene):
    assert len(scene.points) == 3
    for r, c in scene.points:
        bg = np.median(scene.c11[r - 3:r + 4, c - 3:c + 4])
        assert 10 * np.log10(scene.c11[r, c] / bg) > 14.9
        assert abs(abs(scene.gamma[r, c]) - POINT_GAMMA) < 1e-9


def test_diffuse_part_is_psd(scene):
    d11, d22, d12 = scene.diffuse()
    assert np.all(d11 * d22 - np.abs(d12) ** 2 >= -1e-15)


def test_single_look_is_rank_one_and_speckled(scene):
    s_vv, s_vh = observe(scene, np.random.default_rng(2))
    hm = homogeneous_mask(scene)
    assert hm.sum() > 100
    i = np.abs(s_vv[hm]) ** 2
    enl = i.mean() ** 2 / i.var()
    assert 0.7 < enl < 1.4                     # fully developed single-look speckle: ENL ~ 1


def test_homogeneous_mask_is_texture_free(scene):
    hm = homogeneous_mask(scene)
    vals = scene.c11[hm]
    assert vals.std() / vals.mean() < 1e-12


def test_ensemble_average_recovers_designed_coherence(scene):
    """Averaging many independent speckle realisations recovers the truth (per field pooled + per pixel)."""
    rows = check_truth(scene, n_looks=400, rng=np.random.default_rng(3))
    for row in rows:
        if row["field"] == "point":
            assert abs(row["gamma_est"] - row["gamma_true"]) < 0.03
        elif row["n_px"] > 50:
            assert abs(row["gamma_est"] - row["gamma_true"]) < 0.02, row
            if row["gamma_true"] > 0.2:
                assert abs(row["phase_err_deg"]) < 5.0, row


def test_ensemble_covariance_is_unbiased_with_known_noise(scene):
    e11, e22, e12 = ensemble_covariance(scene, 300, np.random.default_rng(4))
    hm = homogeneous_mask(scene)
    assert abs(e11[hm].mean() / scene.c11[hm].mean() - 1) < 0.02
    assert abs((e22[hm].mean() - scene.n_vh) / scene.c22[hm].mean() - 1) < 0.03


def test_scene_is_reproducible():
    a = make_scene(SceneConfig(shape=(64, 64)), np.random.default_rng(7))
    b = make_scene(SceneConfig(shape=(64, 64)), np.random.default_rng(7))
    np.testing.assert_array_equal(a.c12, b.c12)
