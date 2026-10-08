import numpy as np
import pytest

from sar_denoise.safe_io import find_swath_files, parse_annotation

ANNOTATION = """<product>
  <adsHeader><swath>IW2</swath><polarisation>VV</polarisation></adsHeader>
  <generalAnnotation>
    <productInformation>
      <radarFrequency>5.405000454334350e+09</radarFrequency>
      <azimuthSteeringRate>9.798633249999998e-01</azimuthSteeringRate>
      <rangeSamplingRate>6.434523812571428e+07</rangeSamplingRate>
    </productInformation>
    <azimuthFmRateList>
      <azimuthFmRate>
        <azimuthTime>2018-01-25T14:16:04.000000</azimuthTime><t0>5.3565e-03</t0>
        <azimuthFmRatePolynomial count="3">-2311.9 448315.4 79260069.0</azimuthFmRatePolynomial>
      </azimuthFmRate>
      <azimuthFmRate>
        <azimuthTime>2018-01-25T14:16:07.000000</azimuthTime><t0>5.3565e-03</t0>
        <c0>-2300.0</c0><c1>440000.0</c1><c2>79000000.0</c2>
      </azimuthFmRate>
    </azimuthFmRateList>
    <orbitList>
      <orbit><time>2018-01-25T14:16:05.000000</time>
        <velocity><x>-5822.0</x><y>-1486.0</y><z>4636.0</z></velocity></orbit>
      <orbit><time>2018-01-25T14:16:04.000000</time>
        <velocity><x>-5817.0</x><y>-1487.0</y><z>4642.0</z></velocity></orbit>
    </orbitList>
  </generalAnnotation>
  <dopplerCentroid><dcEstimateList>
    <dcEstimate><azimuthTime>2018-01-25T14:16:04.500000</azimuthTime><t0>5.3564e-03</t0>
      <dataDcPolynomial count="3">34.9 -22999.8 8332020.0</dataDcPolynomial></dcEstimate>
  </dcEstimateList></dopplerCentroid>
  <imageAnnotation>
    <imageInformation>
      <slantRangeTime>5.356509155649830e-03</slantRangeTime>
      <azimuthTimeInterval>2.055556e-03</azimuthTimeInterval>
      <rangePixelSpacing>2.33</rangePixelSpacing><azimuthPixelSpacing>13.9</azimuthPixelSpacing>
      <numberOfSamples>10</numberOfSamples><numberOfLines>12</numberOfLines>
    </imageInformation>
    <processingInformation><swathProcParamsList><swathProcParams><azimuthProcessing>
      <windowType>Hamming</windowType><windowCoefficient>0.75</windowCoefficient>
      <processingBandwidth>313</processingBandwidth>
    </azimuthProcessing></swathProcParams></swathProcParamsList></processingInformation>
  </imageAnnotation>
  <swathTiming>
    <linesPerBurst>6</linesPerBurst><samplesPerBurst>10</samplesPerBurst>
    <burstList count="2">
      <burst><azimuthTime>2018-01-25T14:16:04.264578</azimuthTime>
        <firstValidSample count="6">-1 2 2 2 2 -1</firstValidSample>
        <lastValidSample count="6">-1 8 8 8 8 -1</lastValidSample></burst>
      <burst><azimuthTime>2018-01-25T14:16:07.021079</azimuthTime>
        <firstValidSample count="6">-1 1 1 1 1 -1</firstValidSample>
        <lastValidSample count="6">-1 9 9 9 9 -1</lastValidSample></burst>
    </burstList>
  </swathTiming>
</product>"""


@pytest.fixture()
def info(tmp_path):
    f = tmp_path / "ann.xml"
    f.write_text(ANNOTATION)
    return parse_annotation(f)


def test_basic_fields(info):
    assert (info.swath, info.polarisation) == ("IW2", "VV")
    assert (info.n_lines, info.n_samples) == (12, 10)
    assert (info.lines_per_burst, info.samples_per_burst) == (6, 10)
    assert info.azimuth_time_interval == pytest.approx(2.055556e-3)
    assert info.processing_bandwidth == 313.0
    assert info.window_type == "Hamming" and info.window_coefficient == 0.75


def test_bursts_and_valid_ranges(info):
    assert len(info.bursts) == 2
    b0, b1 = info.bursts
    assert (b0.line_start, b1.line_start) == (0, 6)
    assert b0.valid_line_range == (1, 5)
    assert b0.valid_sample_range == (2, 9)      # samples 2..8 inclusive
    assert b1.valid_sample_range == (1, 10)


def test_new_fields_for_deramping(info):
    assert info.slant_range_time == pytest.approx(5.356509155649830e-03)
    assert info.range_sampling_rate == pytest.approx(6.434523812571428e07)
    assert len(info.orbit_times) == 2 and info.orbit_times[0] < info.orbit_times[1]   # sorted
    v0 = np.sqrt(5817.0**2 + 1487.0**2 + 4642.0**2)
    assert info.orbit_speeds[0] == pytest.approx(v0)


def test_polynomial_records_both_layouts(info):
    assert len(info.fm_rates) == 2 and len(info.dc_estimates) == 1
    np.testing.assert_allclose(info.fm_rates[0].coeffs, [-2311.9, 448315.4, 79260069.0])
    np.testing.assert_allclose(info.fm_rates[1].coeffs, [-2300.0, 440000.0, 79000000.0])
    assert info.dc_estimates[0].t0 == pytest.approx(5.3564e-3)


def test_find_swath_files_requires_exactly_one_match(tmp_path):
    (tmp_path / "measurement").mkdir()
    (tmp_path / "annotation").mkdir()
    (tmp_path / "measurement" / "s1a-iw2-slc-vv-x-005.tiff").write_text("")
    (tmp_path / "annotation" / "s1a-iw2-slc-vv-x-005.xml").write_text("<x/>")
    tiff, xml = find_swath_files(tmp_path, "IW2", "VV")
    assert tiff.suffix == ".tiff" and xml.suffix == ".xml"
    with pytest.raises(FileNotFoundError):
        find_swath_files(tmp_path, "iw1", "vv")