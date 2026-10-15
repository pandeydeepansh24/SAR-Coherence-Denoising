from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sar_denoise.covariance import coherence, form_c2
from sar_denoise.patches import (
    PatchConfig,
    Region,
    assign_splits,
    build_index,
    classify,
    conflicts_with,
    crop_products,
    extract_patch,
    find_conflicts,
    load_region,
    local_coherence,
    normalise_pair,
    patch_origins,
    pick_negatives,
    region_time_span,
    regions_from_config,
    save_region,
    summarise_index,
    window_to_slices,
)
from sar_denoise.preprocess import BurstProducts
from sar_denoise.simulate import simulate_dualpol_slc

DT = 2.0556e-3
LINES_PER_BURST = 1509
BURST_STEP_LINES = 1344                      # bursts overlap by 165 lines


def fake_info(n_bursts=4):
    base = np.datetime64("2018-01-25T14:15:53.000000000")
    bursts = [SimpleNamespace(line_start=k * LINES_PER_BURST,
                              azimuth_time=base + np.timedelta64(int(round(k * BURST_STEP_LINES * DT * 1e9)), "ns"))
              for k in range(n_bursts)]
    return SimpleNamespace(bursts=bursts, azimuth_time_interval=DT)


def region(name, split, burst, lines, samples, swath="iw1", **kw):
    return Region(name=name, split=split, swath=swath, burst=burst, lines=lines, samples=samples, **kw)


# ----------------------------------------------------------------------------- configuration

def test_region_from_dict_validates():
    d = {"name": "r", "split": "train", "swath": "IW1", "burst": 2, "lines": [10, 20], "samples": [5, 50]}
    r = Region.from_dict(d)
    assert r.swath == "iw1" and r.lines == (10, 20) and r.kind == "positive"
    with pytest.raises(ValueError):
        Region.from_dict({**d, "split": "dev"})
    with pytest.raises(ValueError):
        Region.from_dict({**d, "lines": [20, 20]})
    with pytest.raises(ValueError):
        regions_from_config({"regions": [d, d]})


# ----------------------------------------------------------------------------- split hygiene

def test_time_span_is_continuous_across_burst_overlap():
    info = fake_info()
    # the last line of burst 0 that is also the first line of burst 1 in time
    t0 = region_time_span(info, 0, (BURST_STEP_LINES, BURST_STEP_LINES + 10))
    t1 = region_time_span(info, 1, (LINES_PER_BURST, LINES_PER_BURST + 10))
    assert t0[0] == pytest.approx(t1[0], abs=1e-6) and t0[1] == pytest.approx(t1[1], abs=1e-6)


def test_conflict_found_through_burst_overlap_but_not_by_line_number():
    infos = {"iw1": fake_info()}
    # the same ground: burst 0 lines 1344..1380 are burst 1 lines 1509..1545 (burst 1 starts 1344 lines later)
    a = region("a", "train", 0, (1344, 1380), (1000, 2000))
    b = region("b", "val", 1, (LINES_PER_BURST, LINES_PER_BURST + 36), (1000, 2000))
    assert b.lines[0] - a.lines[1] >= 128                       # a plain line-number buffer of 128 would miss this
    assert find_conflicts([a, b], infos) == [(a, b)]


def test_buffer_distance_and_same_split_and_other_swath():
    infos = {"iw1": fake_info(), "iw2": fake_info()}
    a = region("a", "train", 0, (100, 228), (1000, 2024))
    near = region("n", "val", 0, (228 + 64, 228 + 192), (1000, 2024))           # 64 lines apart < 128
    far = region("f", "val", 0, (228 + 200, 228 + 328), (1000, 2024))           # 200 lines apart
    aside = region("s", "val", 0, (100, 228), (2024 + 500, 2024 + 1500))        # 500 samples apart < 1024
    same = region("t", "train", 0, (100, 228), (1000, 2024))
    other = region("o", "val", 0, (100, 228), (1000, 2024), swath="iw2")
    assert find_conflicts([a, near], infos) and not find_conflicts([a, far], infos)
    assert find_conflicts([a, aside], infos)
    assert not find_conflicts([a, same], infos)                                  # same split is not leakage
    assert find_conflicts([a, same], infos, ignore_same_split=False)
    assert not find_conflicts([a, other], infos)
    assert conflicts_with(a, [near, far], infos) == [near]


def test_assign_splits_and_pick_negatives_respect_free_check():
    rng = np.random.default_rng(0)
    tiles = []
    for i in range(40):
        tiles.append({"swath": "iw1" if i % 2 else "iw2", "burst": i % 9, "line": 100 + 128 * i, "lines": 128,
                      "sample": 1024 * (i % 5), "samples": 1024, "gamma_p90": 0.05 + 0.002 * i,
                      "pnr_vh_db": [1.0, 5.0, 12.0, 6.0][i % 4], "sigma_vv_db": [-15.0, -12.0, -14.0, -3.0][i % 4]})
    tiles.append({**tiles[0], "gamma_p90": 0.5})                      # coherent tile must never be picked
    tiles.append({**tiles[0], "swath": "iw3", "gamma_p90": 0.05})     # iw3 excluded
    taken = []
    chosen = pick_negatives(tiles, 12, rng, lambda reg, ch: all(reg.name != c.name for c in ch))
    assert 8 <= len(chosen) <= 12 and all(r.kind == "negative" for r in chosen)
    assert all(r.swath in ("iw1", "iw2") for r in chosen)
    assert {r.note for r in chosen} >= {"dark", "clean"}
    assert len({r.name for r in chosen}) == len(chosen)
    none = pick_negatives(tiles, 12, rng, lambda reg, ch: False)
    assert none == []

    split = assign_splits(chosen, (0.5, 0.25, 0.25), rng)
    assert [r.name for r in split] == [r.name for r in chosen] and {r.split for r in split} <= {"train", "val", "test"}


# ----------------------------------------------------------------------------- cutting regions

def make_products(rows=120, cols=300, valid_lines=(7, 7 + 240), line_start=3000, first_sample=500):
    rng = np.random.default_rng(1)
    c2 = lambda: rng.random((4, rows, cols)).astype(np.float32) + 0.1
    meta = {"decimation": 2, "line_start": line_start, "valid_lines": list(valid_lines),
            "samples": [first_sample, first_sample + cols]}
    return BurstProducts(c2_a=c2(), c2_b=c2(), noise_a=rng.random((2, rows, cols)).astype(np.float32),
                         noise_b=rng.random((2, rows, cols)).astype(np.float32),
                         pnr_db=rng.random((2, rows, cols)).astype(np.float32),
                         valid=np.ones((rows, cols), bool), dt_out=2 * DT, meta=meta)


def test_window_to_slices_maps_lines_and_samples():
    prod = make_products()
    first_line = 3000 + 7
    rs, cs = window_to_slices(prod.meta, (first_line + 20, first_line + 60), (520, 600), prod.valid.shape)
    assert (rs.start, rs.stop) == (10, 30) and (cs.start, cs.stop) == (20, 100)
    # odd start rounds up to the next look row
    rs, _ = window_to_slices(prod.meta, (first_line + 21, first_line + 61), (520, 600), prod.valid.shape)
    assert (rs.start, rs.stop) == (11, 31)
    # clipped at the product edge
    rs, cs = window_to_slices(prod.meta, (first_line, first_line + 10_000), (0, 10_000), prod.valid.shape)
    assert (rs.start, rs.stop, cs.start, cs.stop) == (0, 120, 0, 300)
    with pytest.raises(ValueError):
        window_to_slices(prod.meta, (0, 100), (520, 600), prod.valid.shape)


def test_crop_products_and_archive_roundtrip(tmp_path):
    prod = make_products()
    reg = region("r", "train", 2, (3007 + 20, 3007 + 100), (520, 700))
    arrays, meta = crop_products(prod, reg)
    assert arrays["c2_a"].shape == (4, 40, 180) and arrays["gamma"].shape == (40, 180)
    assert np.array_equal(arrays["c2_b"], prod.c2_b[:, 10:50, 20:200])
    assert meta["region_lines_used"] == [3007 + 20, 3007 + 100] and meta["region_samples_used"] == [520, 700]
    assert meta["region"]["name"] == "r" and meta["decimation"] == 2

    path = tmp_path / "r.npz"
    save_region(path, arrays, meta)
    arrays2, meta2 = load_region(path)
    assert set(arrays2) == set(arrays) and meta2["region_samples_used"] == [520, 700]
    assert all(np.array_equal(arrays[k], arrays2[k]) for k in arrays)


# ----------------------------------------------------------------------------- normalisation

def test_normalise_pair_preserves_gamma_phase_and_psd():
    rng = np.random.default_rng(2)
    shape = (64, 256)
    c11, c22, gamma = 7.0, 0.3, 0.6 * np.exp(0.9j)
    c12 = gamma * np.sqrt(c11 * c22)
    a = form_c2(*simulate_dualpol_slc(c11, c22, c12, shape, rng))
    b = form_c2(*simulate_dualpol_slc(c11, c22, c12, shape, rng))
    na = np.stack([np.full(shape, 0.5), np.full(shape, 0.2)]).astype(np.float32)
    nb = 2 * na
    valid = np.ones(shape, bool)
    valid[:5] = False

    an, bn, nan, nbn, (s11, s22) = normalise_pair(a, b, na, nb, valid)
    assert an.dtype == np.float32
    assert 0.5 * (an[0][valid].mean() + bn[0][valid].mean()) == pytest.approx(1.0, rel=1e-5)
    assert 0.5 * (an[1][valid].mean() + bn[1][valid].mean()) == pytest.approx(1.0, rel=1e-5)

    raw_g, raw_p = coherence(a.mean(axis=(1, 2)))
    new_g, new_p = coherence(an.mean(axis=(1, 2)))
    assert new_g == pytest.approx(raw_g, rel=1e-5) and new_p == pytest.approx(raw_p, abs=1e-6)
    # per pixel: single-look |C12|^2 = C11 C22 stays exact (rank one), i.e. the PSD boundary is not crossed
    viol = an[2] ** 2 + an[3] ** 2 - an[0] * an[1]
    assert np.max(np.abs(viol)) < 1e-4 * np.max(an[0] * an[1])
    # the noise is on the same scale as the data: noise / power ratio is unchanged
    assert nan[1].mean() / an[1].mean() == pytest.approx(na[1].mean() / a[1].mean(), rel=1e-5)
    assert nbn[0].mean() == pytest.approx(2 * nan[0].mean(), rel=1e-6)

    with pytest.raises(ValueError):
        normalise_pair(a, b, na, nb, np.zeros(shape, bool))


# ----------------------------------------------------------------------------- patch index

def coherent_half_arrays(rows=128, cols=1600, seed=3):
    """Left half gamma = 0.7, right half incoherent, two independent looks."""
    rng = np.random.default_rng(seed)
    c12 = np.zeros((rows, cols), complex)
    c12[:, : cols // 2] = 0.7 * np.sqrt(1.0 * 0.5)
    looks = [form_c2(*simulate_dualpol_slc(1.0, 0.5, c12, (rows, cols), rng)) for _ in range(2)]
    gamma = local_coherence(looks[0], looks[1])
    noise = np.full((2, rows, cols), 0.05, np.float32)
    return {"c2_a": looks[0], "c2_b": looks[1], "noise_a": noise, "noise_b": noise,
            "pnr_db": np.full((2, rows, cols), 10.0, np.float32), "gamma": gamma,
            "valid": np.ones((rows, cols), bool)}


def test_patch_origins_and_classify():
    cfg = PatchConfig(rows=64, cols=512, stride_rows=32, stride_cols=256)
    org = patch_origins((128, 1100), cfg)
    assert org == [(r, c) for r in (0, 32, 64) for c in (0, 256, 512)]
    assert classify(0.5, cfg) == "coherent" and classify(0.05, cfg) == "weak" and classify(0.0, cfg) == "incoherent"


def test_build_index_separates_coherent_from_incoherent_patches():
    cfg = PatchConfig(stride_rows=64, stride_cols=256)
    arrays = coherent_half_arrays()
    reg = region("r", "train", 2, (0, 1), (0, 1))
    rows = build_index(arrays, reg, cfg)
    left = [r for r in rows if r["col"] + cfg.cols <= 800 - 40]            # entirely inside the coherent half
    right = [r for r in rows if r["col"] >= 800 + 40]
    assert left and right
    assert all(r["class"] == "coherent" and r["frac_coherent"] > 0.9 for r in left)
    assert all(r["class"] == "incoherent" and r["frac_coherent"] < 0.02 for r in right)
    assert all(r["split"] == "train" and r["pnr_vh_db"] == pytest.approx(10.0) for r in rows)
    text = summarise_index(rows)
    assert "train" in text and "coherent" in text

    arrays["valid"][:40] = False                                           # invalid rows drop the affected patches
    kept = build_index(arrays, reg, cfg)
    assert len(kept) < len(rows) and all(r["row"] >= 40 for r in kept)


def test_extract_patch_normalised_and_raw():
    cfg = PatchConfig(stride_rows=64, stride_cols=256)
    arrays = coherent_half_arrays()
    p = extract_patch(arrays, 0, 0, cfg)
    assert p["c2_a"].shape == (4, 64, 512) and p["valid"].shape == (64, 512)
    assert 0.5 * (p["c2_a"][0].mean() + p["c2_b"][0].mean()) == pytest.approx(1.0, rel=1e-5)
    raw = extract_patch(arrays, 0, 0, cfg, normalise=False)
    assert "scales" not in raw and np.array_equal(raw["c2_a"], arrays["c2_a"][:, :64, :512])
    s11, s22 = p["scales"]
    assert p["noise_a"][0].mean() == pytest.approx(0.05 / s11, rel=1e-5)


def test_pick_negatives_never_repeats_a_tile_when_it_fits_two_strata():
    infos = {"iw1": fake_info()}
    # both tiles are "dark" (VH close to the noise) AND "bright" (VV >= -8 dB), so two strata want them
    tiles = [{"swath": "iw1", "burst": 0, "line": 100 + 400 * i, "lines": 128, "sample": 1000, "samples": 1024,
              "gamma_p90": 0.05, "pnr_vh_db": 1.0, "sigma_vv_db": -3.0} for i in range(2)]
    free = lambda reg, chosen: not conflicts_with(reg, chosen, infos, 128, 1024, ignore_same_split=False)
    chosen = pick_negatives(tiles, 8, np.random.default_rng(0), free)
    assert len(chosen) == 2 and len({r.name for r in chosen}) == 2
    # a copy of an already chosen tile is a conflict even though it has the same name
    assert conflicts_with(chosen[0], [replace(chosen[0])], infos, 128, 1024, ignore_same_split=False)