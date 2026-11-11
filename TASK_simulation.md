# Task: simulated-data experiment with a complex-valued U-Net (for tomorrow's presentation)

Read CLAUDE.md first for the existing pipeline. Keep the existing code and tests working; add new modules next to them.

## Professor's requirements
1. The U-Net must take COMPLEX inputs: the two complex single-look pixels (S_VV, S_VH) per pixel,
   not the 4 real channels [C11, C22, Re C12, Im C12].
2. Drop the Look A / Look B self-supervised (Noise2Noise) training. Instead:
   noise-free SIMULATED scene with known coherence -> add speckle systematically -> despeckle
   with SUPERVISED training against the known truth -> show coherence is preserved.
3. Showcase on one mostly land-like patch: noisy patch (left) vs denoised patch (right), with three numbers each:
   (i) mu^2/var (ENL) of VV intensity over a homogeneous area: ~1 for the single-look noisy patch, much larger after;
   (ii) PSNR against the clean truth;
   (iii) complex-valued SSIM against the clean truth.
4. Point scatterers: in the noisy image a bright scatterer is surrounded by many bright speckle spots;
   after denoising it keeps its intensity and the surroundings are clean.

## Decisions already made (defaults agreed with the student)
- Clean data are FULLY SYNTHETIC (no real data needed): land-like scenes built from
  Voronoi "fields" with different VV sigma0 (about -18 to -4 dB), VH 5-12 dB below VV,
  smooth multiplicative texture in some fields and none in others (homogeneous areas for ENL),
  thin linear "roads", per-field complex coherence (|gamma| levels spread over 0..0.9, random phase),
  and a few bright POINT SCATTERERS (deterministic, not speckled; |gamma| ~ 0.95, 15-25 dB above background).
  Truth covariance per pixel: C11, C22, C12 (|gamma| < 1 strictly so the truth is positive definite).
- Speckle: single-look, k ~ CN(0, C) per pixel (construction as in src/sar_denoise/simulate.py),
  point scatterers = deterministic complex vector + background speckle, plus a little thermal noise in VH.
  Verify the truth: boxcar-averaging many independent speckle realisations of the same scene recovers the
  designed coherence (one printed check / slide).
- Network: complex-valued U-Net in the style of Trabelsi et al. 2018 "Deep Complex Networks":
  complex convolutions implemented as pairs of real convs (MPS on the Mac has poor complex-dtype support),
  CReLU (ReLU on real and imaginary parts), complex normalisation, avg-pool down, complex transposed conv up.
  Output: the clean covariance via the existing PSD head (models.raw_to_covariance): real 1x1 conv on
  [Re, Im] of the last features -> (log C11, log C22, u, v). Per-patch normalisation as patches.normalise_pair
  (scales VV and VH separately, keeps gamma exact). Augment with a random common phase e^{j phi} on both channels.
  Keep the existing real U-Net on C2 input as a BASELINE, trained the same way.
- Loss (supervised): KL divergence between CN(0, C_true) and CN(0, C_hat), closed form for 2x2:
  tr(C_hat^-1 C_true) - log det(C_hat^-1 C_true) - 2. It weights C12 on equal footing with the powers.
- Training data generated ON THE FLY (fresh scenes every step, so no memorisation); 128x128 patches;
  fixed seeded validation and test scenes; save best.pt by validation loss. Must run on the Mac (CPU/MPS)
  in well under an hour.
- Metrics, on a fixed showcase scene (about 256x256, mostly land, with point scatterers) and averaged over
  test scenes, for: noisy single look | 5x5 boxcar | complex U-Net | real U-Net:
  - ENL = mu^2/var of VV intensity over the largest homogeneous (texture-free) field, eroded from its edges;
  - PSNR on VV intensity in dB vs truth: 10 log10(R^2 / MSE), R = dynamic range (max - min) of the truth in dB;
  - complex SSIM on the complex coherence gamma vs truth, local Gaussian window:
    (2|sum x y*| + K) / (sum|x|^2 + sum|y|^2 + K);
  - extra: |gamma| bias and RMSE binned by true |gamma|, median phase error where true |gamma| > 0.3.
  - point scatterers: peak VV intensity before/after vs truth, and the number of bright pixels
    (> 3x the local background mean) in a ring 2-8 px around each scatterer.
- Figures: (a) noisy vs denoised showcase patch with the three numbers under each panel (intensity
  composite and |gamma|), (b) zoom on one point scatterer noisy vs denoised, (c) estimated vs true |gamma|
  curve for boxcar and networks, (d) truth panel for reference.
- Write tests for the simulator, the complex layers (shapes, real/imag algebra), the KL loss (zero at truth,
  positive otherwise) and run the full suite.

## Caveat to state in the presentation
Simulation validates the method under ideal fully developed speckle; it does not show real-data behaviour.
