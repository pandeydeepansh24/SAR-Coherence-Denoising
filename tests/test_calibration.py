import numpy as np
import pytest

from sar_denoise.calibration import (
    calibrate_complex,
    evaluate_for_burst,
    find_calibration_files,
    load_calibration_lut,
    load_noise_lut,
    noise_sigma0,
)


def plane(line, pixel):
    return 10.0 + 0.1 * line + 0.5 * pixel


def _write_cal(path, lines, pix_grids, fn=plane, tag="sigmaNought"):
    vecs = ""
    for line, pix in zip(lines, pix_grids):
        vals = " ".join(f"{fn(line, p):.10g}" for p in pix)
        vecs += (
            "<calibrationVector><azimuthTime>2018-01-25T14:15:55.000000</azimuthTime>"
            f"<line>{line}</line><pixel count=\"{len(pix)}\">{' '.join(map(str, pix))}</pixel>"
            f"<{tag}>{vals}</{tag}></calibrationVector>"
        )
    path.write_text(f"<calibration><calibrationVectorList count=\"{len(lines)}\">{vecs}"
                    "</calibrationVectorList></calibration>")


def _write_noise(path, lines, pix_grids, fn=plane, list_tag="noiseVectorList",
                 vec_tag="noiseVector", lut_tag="noiseLut"):
    vecs = ""
    for line, pix in zip(lines, pix_grids):
        vals = " ".join(f"{fn(line, p):.10g}" for p in pix)
        vecs += (
            f"<{vec_tag}><azimuthTime>2018-01-25T14:15:55.000000</azimuthTime>"
            f"<line>{line}</line><pixel count=\"{len(pix)}\">{' '.join(map(str, pix))}</pixel>"
            f"<{lut_tag}>{vals}</{lut_tag}></{vec_tag}>"
        )
    path.write_text(f"<noise><{list_tag}>{vecs}</{list_tag}></noise>")


LINES = [0, 100, 200]
GRID = [0, 50, 100]


def test_bilinear_interpolation_is_exact_for_a_plane(tmp_path):
    f = tmp_path / "cal.xml"
    _write_cal(f, LINES, [GRID] * 3)
    lut = load_calibration_lut(f)
    lines = np.array([0, 37, 100, 150, 199])
    pixels = np.array([0, 13, 50, 99])
    got = lut.evaluate(lines, pixels)
    want = plane(lines[:, None], pixels[None, :])
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_vectors_with_different_pixel_grids_give_same_result(tmp_path):
    f = tmp_path / "cal.xml"
    _write_cal(f, LINES, [[0, 100], [0, 25, 100], [0, 60, 80, 100]])
    lut = load_calibration_lut(f)
    got = lut.evaluate(np.array([20, 120]), np.array([5, 70]))
    want = plane(np.array([20, 120])[:, None], np.array([5, 70])[None, :])
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_values_are_clamped_outside_the_grid(tmp_path):
    f = tmp_path / "cal.xml"
    _write_cal(f, LINES, [GRID] * 3)
    lut = load_calibration_lut(f)
    got = lut.evaluate(np.array([-30, 250]), np.array([0, 100]))
    np.testing.assert_allclose(got[0], [plane(0, 0), plane(0, 100)], rtol=1e-5)
    np.testing.assert_allclose(got[1], [plane(200, 0), plane(200, 100)], rtol=1e-5)


def test_unsorted_vectors_are_sorted(tmp_path):
    f = tmp_path / "cal.xml"
    _write_cal(f, [200, 0, 100], [GRID] * 3)
    lut = load_calibration_lut(f)
    assert list(lut.lines) == [0, 100, 200]
    np.testing.assert_allclose(lut.evaluate(np.array([50]), np.array([50])), plane(50, 50), rtol=1e-5)


def test_noise_lut_both_layouts(tmp_path):
    a, b = tmp_path / "n1.xml", tmp_path / "n2.xml"
    _write_noise(a, LINES, [GRID] * 3)
    _write_noise(b, LINES, [GRID] * 3, list_tag="noiseRangeVectorList",
                 vec_tag="noiseRangeVector", lut_tag="noiseRangeLut")
    for f in (a, b):
        lut = load_noise_lut(f)
        np.testing.assert_allclose(lut.evaluate(np.array([50]), np.array([50])), plane(50, 50), rtol=1e-5)


def test_noise_file_without_vectors_is_rejected(tmp_path):
    f = tmp_path / "bad.xml"
    f.write_text("<noise><other/></noise>")
    with pytest.raises(ValueError):
        load_noise_lut(f)


def test_unknown_kind_is_rejected(tmp_path):
    f = tmp_path / "cal.xml"
    _write_cal(f, LINES, [GRID] * 3)
    with pytest.raises(ValueError):
        load_calibration_lut(f, "nonsense")


def test_calibrate_complex_gives_sigma0_and_noise_scaling():
    s = np.array([[3 + 4j, 6 + 8j]], dtype=np.complex64)   # |s|^2 = 25, 100
    a = np.array([[5.0, 5.0]], dtype=np.float32)
    sig0 = np.abs(calibrate_complex(s, a)) ** 2
    np.testing.assert_allclose(sig0, [[1.0, 4.0]], rtol=1e-6)
    np.testing.assert_allclose(noise_sigma0(np.array([[50.0, 50.0]], dtype=np.float32), a), [[2.0, 2.0]])


def test_evaluate_for_burst_uses_absolute_lines(tmp_path):
    class B:                      # minimal stand-in for safe_io.Burst
        line_start, n_lines, n_samples = 100, 4, 6

    f = tmp_path / "cal.xml"
    _write_cal(f, LINES, [GRID] * 3)
    got = evaluate_for_burst(load_calibration_lut(f), B)
    assert got.shape == (4, 6)
    np.testing.assert_allclose(got[0, 0], plane(100, 0), rtol=1e-5)
    np.testing.assert_allclose(got[3, 5], plane(103, 5), rtol=1e-5)


def test_find_calibration_files(tmp_path):
    d = tmp_path / "annotation" / "calibration"
    d.mkdir(parents=True)
    (d / "calibration-s1a-iw2-slc-vv-2018-005.xml").write_text("<x/>")
    (d / "noise-s1a-iw2-slc-vv-2018-005.xml").write_text("<x/>")
    (d / "calibration-s1a-iw2-slc-vh-2018-002.xml").write_text("<x/>")
    cal, noi = find_calibration_files(tmp_path, "IW2", "VV")
    assert cal.name.startswith("calibration-s1a-iw2-slc-vv")
    assert noi.name.startswith("noise-s1a-iw2-slc-vv")
    with pytest.raises(FileNotFoundError):
        find_calibration_files(tmp_path, "iw3", "vv")