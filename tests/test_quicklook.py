import csv

import numpy as np
import pytest

import sar_denoise.quicklook as ql
from sar_denoise.calibration import Lut
from sar_denoise.quicklook import coarse_c2, coherence_map, save_mosaic, scan_swath, tile_stats, write_tiles_csv
from sar_denoise.simulate import simulate_dualpol_slc
from tests.test_deramp import make_info

C11, C22 = 4.0, 0.5
GAMMA = 0.6 * np.exp(0.7j)
C12 = GAMMA * np.sqrt(C11 * C22)


def test_coarse_c2_matches_a_direct_block_average():
    rng = np.random.default_rng(0)
    vv = (rng.standard_normal((40, 130)) + 1j * rng.standard_normal((40, 130))).astype(np.complex64)
    vh = (rng.standard_normal((40, 130)) + 1j * rng.standard_normal((40, 130))).astype(np.complex64)
    c11, c22, c12 = coarse_c2(vv, vh, block=(8, 16), chunk_rows=16)
    assert c11.shape == (5, 8)                                   # partial blocks are dropped
    blk = lambda x, i, j: x[i * 8:(i + 1) * 8, j * 16:(j + 1) * 16]
    assert c11[2, 3] == pytest.approx(np.mean(np.abs(blk(vv, 2, 3)) ** 2), rel=1e-5)
    assert c22[4, 7] == pytest.approx(np.mean(np.abs(blk(vh, 4, 7)) ** 2), rel=1e-5)
    assert c12[1, 5] == pytest.approx(np.mean(blk(vv, 1, 5) * np.conj(blk(vh, 1, 5))), rel=1e-4, abs=1e-5)


def test_coarse_c2_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        coarse_c2(np.ones((16, 16), complex), np.ones((16, 17), complex))


def test_block_coherence_recovers_truth_and_pure_speckle_sits_at_a_small_floor():
    vv, vh = simulate_dualpol_slc(C11, C22, C12, (256, 1024), np.random.default_rng(1))
    g = coherence_map(*coarse_c2(vv, vh, block=(8, 64)))
    assert np.mean(g) == pytest.approx(0.6, abs=0.03)

    vv, vh = simulate_dualpol_slc(C11, C22, 0.0, (256, 1024), np.random.default_rng(2))
    g0 = coherence_map(*coarse_c2(vv, vh, block=(8, 64)))
    assert 0.02 < np.mean(g0) < 0.08                           # about 0.89 / sqrt(512) = 0.04


def test_tile_stats_separates_coherent_from_incoherent_tiles():
    gamma = np.full((16, 32), 0.05)
    gamma[:, 16:] = 0.7
    pnr = np.full((16, 32), 6.0)
    sig = np.full((16, 32), -10.0)
    left, right = tile_stats(gamma, pnr, sig, tile=(16, 16))
    assert (left["row"], left["col"], right["col"]) == (0, 0, 16)
    assert left["frac_gamma_above"] == 0.0 and right["frac_gamma_above"] == 1.0
    assert right["gamma_p90"] == pytest.approx(0.7) and left["pnr_vh_db"] == pytest.approx(6.0)


def test_scan_swath_with_mocked_files(monkeypatch, tmp_path):
    info = make_info(n_lines=512, n_samples=300)
    vv, vh = simulate_dualpol_slc(C11, C22, C12, (512, 300), np.random.default_rng(3))
    dn = {"vv": (300.0 * vv).astype(np.complex64), "vh": (300.0 * vh).astype(np.complex64)}
    monkeypatch.setattr(ql, "find_swath_files", lambda safe, swath, pol: (pol, pol))
    monkeypatch.setattr(ql, "parse_annotation", lambda xml: info)
    monkeypatch.setattr(ql, "read_burst", lambda tiff, inf, k: dn[tiff])
    monkeypatch.setattr(ql, "find_calibration_files", lambda safe, swath, pol: ("cal", "noise"))
    const = lambda v: Lut(lines=np.array([0.0, 4000.0]), pixels=[np.array([0.0, 1000.0])] * 2,
                          values=[np.full(2, v)] * 2)
    monkeypatch.setattr(ql, "load_calibration_lut", lambda path, kind: const(300.0))
    monkeypatch.setattr(ql, "load_noise_lut", lambda path: const(50.0))

    res = scan_swath("fake.SAFE", "iw2", block=(8, 16), tile=(8, 8), progress=lambda *_: None)
    maps = res["maps"]
    assert maps["gamma"].shape == (64, 18)                        # 512/8 rows, 300/16 columns
    assert set(maps) == {"sigma_vv_db", "sigma_vh_db", "pnr_vh_db", "gamma"}
    assert np.mean(maps["gamma"]) == pytest.approx(0.6, abs=0.05)
    assert np.mean(maps["sigma_vv_db"]) == pytest.approx(10 * np.log10(C11), abs=0.4)
    assert np.mean(maps["pnr_vh_db"]) == pytest.approx(10 * np.log10(C22 / (50.0 / 300.0 ** 2)), abs=0.6)

    tiles = res["tiles"]
    assert len(tiles) == 8 * 2                                    # 64/8 tile rows x 18//8 tile columns
    t = tiles[0]
    assert (t["swath"], t["burst"], t["line"], t["sample"], t["lines"], t["samples"]) == ("iw2", 0, 0, 0, 64, 128)

    save_mosaic(res, tmp_path / "ql.png")
    assert (tmp_path / "ql.png").stat().st_size > 1000
    write_tiles_csv(tiles, tmp_path / "tiles.csv")
    with open(tmp_path / "tiles.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(tiles) and rows[0]["swath"] == "iw2" and "gamma_p90" in rows[0]