import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from sar_denoise.patches import load_region
from sar_denoise.preprocess import BurstProducts
from tests.test_patches import fake_info

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_patches.py"


@pytest.fixture()
def script(monkeypatch):
    spec = importlib.util.spec_from_file_location("make_patches_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    calls = []

    def fake_process_burst(safe, swath, burst, cols=None, trim_rows=32, **kw):
        calls.append((swath, burst, cols))
        s0, s1 = max(0, cols[0]), cols[1]
        rows, width = 600, s1 - s0
        rng = np.random.default_rng(len(calls))
        valid = np.ones((rows, width), bool)
        valid[:trim_rows] = False
        valid[-trim_rows:] = False
        pos = lambda *shape: (rng.random(shape) + 0.1).astype(np.float32)
        meta = {"decimation": 2, "line_start": 0, "valid_lines": [0, 1200], "samples": [s0, s1],
                "dt_out": 4e-3, "swath": swath, "burst": burst}
        return BurstProducts(c2_a=pos(4, rows, width), c2_b=pos(4, rows, width), noise_a=pos(2, rows, width),
                             noise_b=pos(2, rows, width), pnr_db=pos(2, rows, width), valid=valid,
                             dt_out=4e-3, meta=meta)

    monkeypatch.setattr(mod, "find_swath_files", lambda safe, sw, pol: ("tiff", "xml"))
    monkeypatch.setattr(mod, "parse_annotation", lambda xml: fake_info())
    monkeypatch.setattr(mod, "process_burst", fake_process_burst)
    mod.calls = calls
    return mod


def write_config(path, regions, negatives_count=0):
    cfg = {"patch": {"rows": 64, "cols": 512, "stride_rows": 64, "stride_cols": 256},
           "buffer": {"lines": 128, "samples": 1024},
           "preprocess": {"trim_rows": 32, "column_margin": 64},
           "negatives": {"count": negatives_count, "swaths": ["iw1"], "fractions": [1.0, 0.0, 0.0]},
           "regions": regions}
    path.write_text(yaml.safe_dump(cfg))


def reg(name, split, lines, samples):
    return {"name": name, "split": split, "swath": "iw1", "burst": 0, "lines": lines, "samples": samples}


def run(script, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["make_patches.py", *argv])
    return script.main()


def test_dry_run_prints_plan_and_touches_nothing(script, monkeypatch, tmp_path, capsys):
    cfgp = tmp_path / "regions.yaml"
    write_config(cfgp, [reg("a", "train", [100, 500], [0, 1100]), reg("b", "val", [900, 1300], [4000, 5100])])
    out = tmp_path / "out"
    rc = run(script, monkeypatch, "SAFE", "--config", str(cfgp), "--out-dir", str(out), "--dry-run")
    text = capsys.readouterr().out
    assert rc == 0 and "a " in text and "b " in text and not script.calls and not out.exists()


def test_conflicting_regions_are_refused(script, monkeypatch, tmp_path, capsys):
    cfgp = tmp_path / "regions.yaml"
    write_config(cfgp, [reg("a", "train", [100, 500], [0, 1100]), reg("b", "val", [550, 900], [0, 1100])])
    rc = run(script, monkeypatch, "SAFE", "--config", str(cfgp), "--out-dir", str(tmp_path / "o"))
    assert rc == 1 and "CONFLICT" in capsys.readouterr().out and not script.calls


def test_full_run_writes_archives_and_index_and_reuses_them(script, monkeypatch, tmp_path, capsys):
    cfgp = tmp_path / "regions.yaml"
    write_config(cfgp, [reg("a", "train", [100, 500], [0, 1100]), reg("b", "val", [900, 1300], [4000, 5100])])
    out = tmp_path / "out"
    assert run(script, monkeypatch, "SAFE", "--config", str(cfgp), "--out-dir", str(out)) == 0
    text = capsys.readouterr().out
    assert (out / "a.npz").exists() and (out / "b.npz").exists() and len(script.calls) == 2
    assert script.calls[0][2] == (-64, 1164)                       # region columns plus the margin

    arrays, meta = load_region(out / "a.npz")
    assert arrays["c2_a"].shape == (4, 200, 1100) and meta["region"]["name"] == "a"

    with open(out / "index.csv") as f:
        rows = list(csv.DictReader(f))
    assert {r["region"] for r in rows} == {"a", "b"} and {r["split"] for r in rows} == {"train", "val"}
    assert all(float(r["frac_valid"]) >= 0.99 for r in rows)
    assert "train" in text and "val" in text

    # second run loads the archives instead of recomputing
    assert run(script, monkeypatch, "SAFE", "--config", str(cfgp), "--out-dir", str(out)) == 0
    assert len(script.calls) == 2 and "loaded" in capsys.readouterr().out


def test_negative_tiles_are_added_and_respect_the_buffer(script, monkeypatch, tmp_path, capsys):
    cfgp = tmp_path / "regions.yaml"
    write_config(cfgp, [reg("a", "train", [100, 500], [0, 1100])], negatives_count=4)
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    fields = ["swath", "burst", "line", "lines", "sample", "samples", "gamma_p90", "pnr_vh_db", "sigma_vv_db"]
    specs = [(1.0, -15.0), (5.0, -12.0), (12.0, -14.0), (6.0, -3.0), (1.0, -3.0)]   # the last fits two strata
    with open(tiles_dir / "iw1_tiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (pnr, vv) in enumerate(specs):
            w.writerow({"swath": "iw1", "burst": 0, "line": 100 + 300 * i, "lines": 128, "sample": 3000,
                        "samples": 1024, "gamma_p90": 0.05, "pnr_vh_db": pnr, "sigma_vv_db": vv})
        # this tile sits right next to region a and must be rejected by the buffer check
        w.writerow({"swath": "iw1", "burst": 0, "line": 200, "lines": 128, "sample": 1200, "samples": 1024,
                    "gamma_p90": 0.05, "pnr_vh_db": 5.0, "sigma_vv_db": -12.0})
    out = tmp_path / "out"
    rc = run(script, monkeypatch, "SAFE", "--config", str(cfgp), "--out-dir", str(out), "--tiles-dir", str(tiles_dir))
    assert rc == 0
    names = {p.stem for p in out.glob("neg_*.npz")}
    assert 2 <= len(names) <= 4
    assert "neg_iw1_b0_200_1200" not in names
    with open(out / "index.csv") as f:
        rows = list(csv.DictReader(f))
    assert {r["kind"] for r in rows} == {"positive", "negative"}
    keys = [(r["region"], r["row"], r["col"]) for r in rows]
    assert len(keys) == len(set(keys))                           # no patch is listed twice