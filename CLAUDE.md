# SAR-Coherence-Denoising: project context for Claude Code

## Goal
Self-supervised despeckling of dual-pol (VV/VH) Sentinel-1 IW SLC data that PRESERVES the VV-VH complex
coherence (|gamma| and phase of C12 = S_VV * conj(S_VH)).
- Doppler sub-look splitting gives Noise2Noise pairs: Look A = input, Look B = target.
- U-Net on 4 real channels [C11, C22, Re C12, Im C12].
- The covariance/coherence matrix must be involved in the training loss, not only the intensities.

## Environment and commands
- macOS arm64, Python 3.14 venv in the repo (`source venv/bin/activate`), `pip install -e .`
- Tests: `pytest tests -q` (run from repo root; some tests import `tests.test_deramp` / `tests.test_sublooks`). 147 passing.
- Data (outside the repo, never commit it):
  `export SAFE=~/Desktop/S1A_IW_SLC__1SDV_20180125T141553_20180125T141620_020312_022B03_3605.SAFE`
- Repo is public (GitHub `pandeydeepansh24/SAR-Coherence-Denoising`): no secrets, no data, no venv, no egg-info.
- Use lowercase file names (a capital `Preprocess.py` once broke imports on macOS).
- Run code from the terminal with the venv active, not the editor run button.

## Package layout (`src/sar_denoise/`)
| Module | Status | Purpose |
|---|---|---|
| safe_io.py | done | annotation parser (SwathInfo, Burst, orbit speeds, timings), `find_swath_files`, `read_burst` |
| deramp.py | done | TOPS deramp + demodulation (ESA COPE-GSEG-EOPG-TN-14-0025 Rev 3); CLI with spectrum plot |
| calibration.py | done | sigma0 / beta LUTs, range thermal-noise LUT (both layouts), `calibrate_complex`, `noise_sigma0` |
| covariance.py | done | `form_c2`, `multilook`, `coherence`, `psd_violation`; channel order C11, C22, ReC12, ImC12 |
| simulate.py | done | `simulate_dualpol_slc` with known covariance; `simulate_look_pair` (two looks + noise maps) for tests |
| sublooks.py | done | `SubLookConfig`, `split_sublooks` (A/B looks), noise factors, diagnostics, per-look power equalisation |
| preprocess.py | done | `process_burst` -> `BurstProducts` (c2_a, c2_b, noise_a/b, pnr_db, valid, meta); CLI |
| quicklook.py | done | coarse coherence / VH power-to-noise scan of whole sub-swaths, PNG mosaic and tile CSV |
| patches.py | done | `Region`, `PatchConfig`, split-leakage check in (azimuth time, sample) space, `crop_products`, `build_index`, `normalise_pair` (C -> D C D, keeps gamma and phase), region archives |
| scripts/make_patches.py | done | cuts the regions in `configs/regions.yaml` (+ random incoherent negative tiles) out of the SAFE product, writes `<out-dir>/<region>.npz` and `index.csv` |
| models.py | done | `UNet` (GroupNorm, zero-init head, in_ch 6 = C2 + 2 noise maps), `raw_to_covariance` (log C11, log C22, z -> gamma = tanh|z| z/|z|), `covariance_stack`, `make_input` |
| losses.py | done | closed-form 2x2 complex-Gaussian NLL with Sigma = C_hat + diag(noise); `covariance_nll`, `n2n_loss` (A->B and B->A), `coherence_metrics` (phase error as angle(g_hat conj(g_ref))) |
| dataset.py | done | `PatchDataset` (index.csv + region archives, normalise_pair per patch, no augmentation), `class_balanced_weights`, `collate` |
| train.py | done | `python -m sar_denoise.train --out-dir RUN`; class-balanced sampler, symmetric N2N loss, per-class val loss and mean |gamma|, best.pt/last.pt/log.csv |
| predict.py + scripts/predict_region.py | done | tiled whole-region inference (training-size patches, per-patch normalisation, overlap-averaged), boxcar reference from the other look, 6-panel figure |
| scripts/make_showcase.py | done | prof-facing before/after figure: single look vs 5x33 boxcar vs network vs 9x65 boxcar reference of the other look (single-look coherence is |gamma|=1 by construction, so shown as a note, not a map) |
| scripts/make_before_after.py | done | talk figure, BEFORE left / AFTER right: intensity composite (single look or boxcar vs network) with ENL in an auto-chosen flat box, and |gamma| (5x33 boxcar vs network) with sign agreement / phase error vs the independent look B |
| evaluate.py | done | `python -m sar_denoise.evaluate --run RUN --split val`: net vs net_nocoh vs boxcar (full/diag) NLL per class, coherence gain, sign agreement, phase error, false coherence |
| sim_scene.py | done | SIMULATION experiment (TASK_simulation.md): `make_scene` (Voronoi fields, texture, roads, point scatterers with |gamma| 0.95), `observe` (single look + VH thermal noise), `homogeneous_mask` (ENL area), `check_truth` |
| complex_models.py | done | `ComplexUNet` (Trabelsi-style: block-weight complex conv, CReLU, bmm complex whitening GroupNorm, avg-pool, complex transposed conv) -> existing PSD head |
| sim_losses.py | done | `kl_map` / `kl_loss`: closed-form 2x2 KL(CN(0,C_true) || CN(0,C_hat)); invariant to C -> D C D |
| sim_train.py | done | `python -m sar_denoise.sim_train --model complex|real --out-dir RUN`; fresh scenes every step, fixed seeded val, best.pt by val KL, `--points-per-10k` |
| sim_metrics.py | done | ENL, PSNR (VV dB), complex SSIM on gamma (NOT sensitive to a constant phase offset), |gamma| bins, phase error, point-scatterer ring counts |
| sim_eval.py | done | `python -m sar_denoise.sim_eval --complex A/best.pt --real B/best.pt --out-dir FIG`: showcase_main/all, point_zoom, gamma_bias PNGs + metrics.md/json |
| scripts/check_sim_truth.py | done | averaging 1000 realisations recovers designed |gamma| (fields within 0.0013, points 0.0033) |

Config: `configs/default.yaml` (`data.safe_dir`), `configs/regions.yaml` (train/val/test regions, patch size, buffers), `pyproject.toml` (setuptools, src layout), `requirements.txt`.

## Pipeline order (per burst, identical on VV and VH)
read burst -> calibrate (S/A_sigma) -> deramp + demodulate -> sub-look split (FFT azimuth, guard 10 Hz, cosine ramps,
integer-bin baseband shift, decimate 2, de-weighting OFF, per-look power equalisation) -> C2 for look A and look B
-> noise maps per look (LUT noise x noise_factor x gain^2) -> crop regions -> per-patch joint normalisation.

CLI examples:
- `python -m sar_denoise.safe_io "$SAFE" iw2 vv`
- `python -m sar_denoise.deramp "$SAFE" iw2 vv --plot ...`
- `python -m sar_denoise.preprocess "$SAFE" iw2 [--burst K] [--cols 4096] [--trim 32] [--out f.npz]`
- `python -m sar_denoise.quicklook "$SAFE" --swath iw2 --out-dir "$HOME/data/sar/quicklook"`
- `python scripts/make_patches.py "$SAFE" --dry-run` then without `--dry-run` (output in `~/data/sar/regions`)

## Invariants (do not break)
1. Same operations on VV and VH, always.
2. Never apply nonlinear transforms or standardise Re C12 and Im C12 separately; scale by sqrt(s11 * s22) so phase and |gamma| survive.
3. Do not subtract noise from the data; model it in the loss instead.
4. No rotation/transpose augmentation (azimuth and range are not interchangeable); no resampling or geocoding before training.
5. A single-look C2 is rank one (|gamma| = 1); coherence is only meaningful for estimates (multilooked or network output).
6. Per-look gain equalisation is a real scalar, so |gamma| and phase are unchanged.
7. Do not enable Hamming de-weighting: noise is nearly white, so it inflates noise.

## Planned model and loss
- Output head enforcing PSD: log C11, log C22, gamma_hat = tanh(|z|) * z/|z|, so C12 = gamma_hat * sqrt(C11 C22).
- Noise-aware complex Gaussian NLL: Sigma = C_hat + diag(noise_B); L = log det Sigma + tr(Sigma^-1 C_B).
  Train A->B and B->A; average both at inference.
- Optional extra terms: coherence-magnitude and phase consistency on multilooked outputs.

## Measured facts (IW2, this product)
- 9 bursts x 1509 lines x 25122 samples per burst; ~165-line burst overlap; use valid line/sample ranges.
- Azimuth sampling 486.5 Hz (dt 2.0556 ms), processing bandwidth 313 Hz, Hamming coefficient 0.75.
- Pixels 2.33 m (range) x 13.93 m (azimuth); 27.9 m azimuth after decimation by 2.
- After deramp + demod the band is about +/-156 Hz. Sub-look independence: complex correlation 0.001, C12 speckle correlation 0.001.
- VH power/noise about 1.5 dB, i.e. true VH SNR about -3.9 dB; coherence bias factor sqrt(SNR/(1+SNR)) about 0.54.
- Noise LUT variation about 0.5 dB peak to peak; old layout (no azimuth noise vectors).

## Scene (quick-look of IW1, IW2, IW3)
- IW3 is noise-dominated: VH power/noise median +1.7 dB, 99% +3.0 dB, no block with |gamma| > 0.3. NOT used.
- IW1 is the best data: VH power/noise median +6.3 dB; top tiles have |gamma| p90 0.40-0.57 and VH 8-15 dB above noise.
- IW2 is weakly coherent: best tile |gamma| p90 0.32 (burst 0), VH 11-13 dB above noise.
- The null floor of block |gamma| (8 x 64) is about 0.06-0.07 (median in incoherent areas); only values clearly above are coherence.
- Independent coherent regions (see `configs/regions.yaml`): R1 = IW1 burst 2, lines 3015-4423, samples 11563-17707 (main block);
  R2 = IW1 burst 4, lines 6520-6904, samples 3371-6443; R3 = IW1 burst 4, line 6264, sample 9515 (single tile);
  IW2 burst 0 weak regions w1 (lines 923-1307, samples 1298-4370) and w2 (lines 155-539, samples 4370-5394).
- Default split: train R1 + w1, val R2 + w2, test R3 (single tile, provisional: replace by a region of a second SLC product).
- Estimated strongly coherent non-overlapping 64 x 512 patches: about 50 in total. Overlapping patches (stride 32 x 256) are not independent.
- Neighbouring bursts overlap on the ground (~165 lines): split by region and check distances in azimuth time, not line number.

## Simulation experiment (measured, 2026-11-11)
- Speed on M3/MPS, B=8, 128x128: complex U-Net base 16 (0.97 M params) ~0.5 s/step, real U-Net base 24 (1.09 M) ~0.15 s/step.
- Complex net learns coherence only after ~300 steps (it must form S_VV S_VH* itself); at 1000 steps on 6 test scenes:
  ENL 89, PSNR 26.4 dB, complex SSIM 0.87, phase err 7.8 deg (boxcar 5x5: 24, 26.2, 0.79, 9.3).
- Weak spot: point scatterers smeared (~ -10 dB peak) because points are ~0.02% of pixels; `--points-per-10k 6` added, untested.

## Next steps
1. Run `scripts/make_patches.py` on the real product, inspect the class counts per split in the printed summary and look at a few
   regions (coherent and negative) to confirm they are what the quick-look says.
2. DONE: models.py, losses.py, simulated end-to-end test (recovers gamma = +-0.8 to within 0.01 incl. sign). Needs `pip install torch`.
2b. DONE: dataset.py, train.py (102 tests). Real run r1 (1500 steps, MPS, ~10 min): val loss 1.097 -> 1.005, no overfit; val mean |g|: coherent 0.136, weak 0.114, incoherent 0.069. evaluate on val (r1 depth 3 vs r2 depth 4: identical, coh gain 0.037/0.038, sign agreement 0.91/0.92, so capacity is NOT the bottleneck; depth 3 is 1.7x faster). r1 calibration vs reference: network |g| is compressed (ref 0.57 -> pred 0.24, ref 0.78 -> 0.30) while sign agreement is 0.93 and coh gain rises monotonically; unresolved whether that is shrinkage or the 9x65 reference spreading point-target coherence; evaluate.py now has a rescaling sweep (multiply |g^| by k, best k per predicted-|g| bin) to decide. 114 tests. NEXT: read the sweep, then decide (second SLC product, Colab only if scaling).
   Lessons: head must NOT be zero-init (saddle at gamma=0); small static datasets get memorised (watch val_gamma_incoherent, keep best.pt).
   (check recovered |gamma| and phase, bias vs. a plain MSE baseline).
3. `train.py` + Colab notebook that clones the repo and loads the region archives (the raw SAFE never goes to Colab);
   sample coherent patches more often, keep negatives; random crop offsets and A<->B swap are the only augmentations until flips are tested.
4. `evaluate.py` (|gamma| and phase error vs. multilooked reference, ENL, edge preservation, per class), README.
5. Second SLC product (more coherent land) and use one of its regions as the real test set.

## Working agreements
- Keep tests passing; add a test with every new module. Prefer small, verifiable steps and show measured numbers.
- Explain choices in terms of coherence preservation. Flag any step that could bias |gamma| or phase.
- Do not commit data, patches, or model checkpoints; keep them outside the repo or in .gitignore.
