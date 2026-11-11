"""Fully synthetic, land-like dual-pol scenes with a KNOWN covariance, and their single-look speckled observations.

Scene (noise-free truth, per pixel C = [[C11, C12], [C12*, C22]]):
  * Voronoi "fields", each with VV sigma0 in [-18, -4] dB, VH 5-12 dB below VV, a complex coherence
    gamma = |gamma| e^{j phi} with |gamma| uniform on [0, 0.9] and phi uniform, and either a smooth
    multiplicative texture (mean 1, applied to BOTH channels so gamma is unchanged) or none (homogeneous
    fields: these are where ENL is measured).
  * thin, dark linear "roads" (low |gamma|).
  * a few POINT SCATTERERS: one pixel holding a deterministic complex vector a = [a_vv, a_vh] 15-25 dB above
    the local VV background, on top of the background speckle. Their truth covariance is
        C_point = a a^H + C_bg + diag(0, d)
    where the extra diffuse VH power d >= 0 is chosen so that |gamma_point| = 0.95 (strictly < 1, so every
    pixel of the truth is positive definite).

Observation (one single look):  k = a + CN(0, C_diffuse) + CN(0, diag(0, n_vh))
  * C_diffuse is the truth minus a a^H (equal to the truth away from point scatterers);
  * the speckle construction is simulate.simulate_dualpol_slc (fully developed, circular Gaussian);
  * n_vh is a small VH thermal noise (NESZ about -28 to -24 dB). The truth does NOT contain it: the network
    has to remove it, like the speckle. VV thermal noise is neglected (VV is far above NESZ).

So E[k k^H] = C_truth + diag(0, n_vh): averaging many independent looks recovers the designed covariance
(exactly in VV and C12, and in VH after subtracting the known n_vh). check_truth() measures this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

from .simulate import _cn, simulate_dualpol_slc

POINT_GAMMA = 0.95


@dataclass
class SceneConfig:
    shape: tuple[int, int] = (128, 128)
    fields_per_10k_px: float = 5.0           # Voronoi seeds per 100 x 100 pixels
    vv_db: tuple[float, float] = (-18.0, -4.0)
    vh_below_vv_db: tuple[float, float] = (5.0, 12.0)
    gamma_max: float = 0.9
    p_textured: float = 0.5
    texture_std: tuple[float, float] = (0.2, 0.6)      # std of log-texture
    texture_corr_px: tuple[float, float] = (2.0, 8.0)  # Gaussian smoothing sigma of the texture field
    roads_per_10k_px: float = 0.6
    road_width: tuple[int, int] = (1, 3)
    road_vv_db: tuple[float, float] = (-22.0, -18.0)
    road_gamma_max: float = 0.2
    points_per_10k_px: float = 1.5
    point_above_bg_db: tuple[float, float] = (15.0, 25.0)
    point_vh_below_vv_db: tuple[float, float] = (3.0, 8.0)
    point_margin_px: int = 10                 # from the image edge and from other points
    nesz_vh_db: tuple[float, float] = (-28.0, -24.0)


@dataclass
class Scene:
    c11: np.ndarray            # truth, float64 (H, W)
    c22: np.ndarray
    c12: np.ndarray            # complex128
    a_vv: np.ndarray           # deterministic point-scatterer vector (0 away from points), complex128
    a_vh: np.ndarray
    n_vh: float                # VH thermal-noise power (not in the truth)
    labels: np.ndarray         # field id per pixel, -1 on roads
    textured: np.ndarray       # bool per field id
    points: list = field(default_factory=list)   # [(row, col), ...]

    @property
    def shape(self):
        return self.c11.shape

    @property
    def gamma(self) -> np.ndarray:
        return self.c12 / np.sqrt(self.c11 * self.c22)

    def truth_c2(self, dtype=np.float32) -> np.ndarray:
        """(4, H, W) [C11, C22, Re C12, Im C12] of the noise-free truth."""
        return np.stack([self.c11, self.c22, self.c12.real, self.c12.imag]).astype(dtype)

    def diffuse(self):
        """Truth covariance without the deterministic point part (what the speckle is drawn from)."""
        return (self.c11 - np.abs(self.a_vv) ** 2, self.c22 - np.abs(self.a_vh) ** 2,
                self.c12 - self.a_vv * np.conj(self.a_vh))


def _db(x):
    return 10.0 ** (np.asarray(x) / 10.0)


def _voronoi(shape, n, rng):
    h, w = shape
    seeds = rng.uniform([0, 0], [h, w], size=(n, 2))
    rr, cc = np.mgrid[0:h, 0:w]
    d2 = (rr[..., None] - seeds[:, 0]) ** 2 + (cc[..., None] - seeds[:, 1]) ** 2
    return np.argmin(d2, axis=-1)


def _texture(shape, std, corr, rng):
    g = gaussian_filter(rng.standard_normal(shape), corr, mode="wrap")
    g /= g.std() + 1e-12
    return np.exp(std * g - 0.5 * std ** 2)        # lognormal, mean 1


def _road_mask(shape, n, width_range, rng):
    h, w = shape
    rr, cc = np.mgrid[0:h, 0:w]
    mask = np.zeros(shape, dtype=bool)
    for _ in range(n):
        r0, c0 = rng.uniform(0, h), rng.uniform(0, w)
        th = rng.uniform(0, np.pi)
        dist = np.abs((rr - r0) * np.cos(th) - (cc - c0) * np.sin(th))
        mask |= dist < 0.5 * rng.integers(width_range[0], width_range[1] + 1)
    return mask


def _n(rate_per_10k, shape, rng, lo=0):
    return max(lo, int(rng.poisson(rate_per_10k * shape[0] * shape[1] / 1e4)))


def make_scene(cfg: SceneConfig | None = None, rng=None, n_points: int | None = None) -> Scene:
    cfg = SceneConfig() if cfg is None else cfg
    rng = np.random.default_rng() if rng is None else rng
    shape = tuple(cfg.shape)
    n_fields = _n(cfg.fields_per_10k_px, shape, rng, lo=2)
    labels = _voronoi(shape, n_fields, rng)

    vv = rng.uniform(*cfg.vv_db, size=n_fields)
    vh = vv - rng.uniform(*cfg.vh_below_vv_db, size=n_fields)
    gabs = rng.uniform(0.0, cfg.gamma_max, size=n_fields)
    gph = rng.uniform(-np.pi, np.pi, size=n_fields)
    textured = rng.random(n_fields) < cfg.p_textured

    c11 = _db(vv)[labels]
    c22 = _db(vh)[labels]
    gamma = (gabs * np.exp(1j * gph))[labels]
    tex = np.ones(shape)
    for i in np.flatnonzero(textured):
        t = _texture(shape, rng.uniform(*cfg.texture_std), rng.uniform(*cfg.texture_corr_px), rng)
        sel = labels == i
        tex[sel] = t[sel]
    c11, c22 = c11 * tex, c22 * tex

    roads = _road_mask(shape, _n(cfg.roads_per_10k_px, shape, rng), cfg.road_width, rng)
    if roads.any():
        r_vv = rng.uniform(*cfg.road_vv_db)
        r_vh = r_vv - rng.uniform(*cfg.vh_below_vv_db)
        c11[roads], c22[roads] = _db(r_vv), _db(r_vh)
        gamma[roads] = rng.uniform(0, cfg.road_gamma_max) * np.exp(1j * rng.uniform(-np.pi, np.pi))
        labels = np.where(roads, -1, labels)
    c12 = gamma * np.sqrt(c11 * c22)

    a_vv = np.zeros(shape, complex)
    a_vh = np.zeros(shape, complex)
    points = _place_points(shape, labels, cfg, rng, n_points)
    for r, c in points:
        p_vv = c11[r, c] * _db(rng.uniform(*cfg.point_above_bg_db))
        p_vh = p_vv / _db(rng.uniform(*cfg.point_vh_below_vv_db))
        a_vv[r, c] = np.sqrt(p_vv) * np.exp(1j * rng.uniform(-np.pi, np.pi))
        a_vh[r, c] = np.sqrt(p_vh) * np.exp(1j * rng.uniform(-np.pi, np.pi))
        t11 = c11[r, c] + p_vv
        t22 = c22[r, c] + p_vh
        t12 = c12[r, c] + a_vv[r, c] * np.conj(a_vh[r, c])
        d = max(0.0, abs(t12) ** 2 / (POINT_GAMMA ** 2 * t11) - t22)   # extra diffuse VH -> |gamma| = 0.95
        c11[r, c], c22[r, c], c12[r, c] = t11, t22 + d, t12

    n_vh = float(_db(rng.uniform(*cfg.nesz_vh_db)))
    return Scene(c11=c11, c22=c22, c12=c12, a_vv=a_vv, a_vh=a_vh, n_vh=n_vh, labels=labels,
                 textured=textured, points=points)


def _place_points(shape, labels, cfg, rng, n_points=None):
    """Points well inside a field (not on a road / boundary), away from the image edge and from each other."""
    n = _n(cfg.points_per_10k_px, shape, rng) if n_points is None else n_points
    if n == 0:
        return []
    m = cfg.point_margin_px
    inner = np.zeros(shape, bool)
    for lab in np.unique(labels[labels >= 0]):
        inner |= binary_erosion(labels == lab, iterations=4)
    inner[:m], inner[-m:], inner[:, :m], inner[:, -m:] = False, False, False, False
    cand = np.argwhere(inner)
    rng.shuffle(cand)
    pts = []
    for r, c in cand:
        if all(max(abs(r - pr), abs(c - pc)) >= m for pr, pc in pts):
            pts.append((int(r), int(c)))
            if len(pts) == n:
                break
    return pts


def observe(scene: Scene, rng=None, thermal: bool = True):
    """One single-look observation (s_vv, s_vh), complex128."""
    rng = np.random.default_rng() if rng is None else rng
    d11, d22, d12 = scene.diffuse()
    s_vv, s_vh = simulate_dualpol_slc(d11, d22, d12, rng=rng)
    s_vv = s_vv + scene.a_vv
    s_vh = s_vh + scene.a_vh
    if thermal:
        s_vh = s_vh + np.sqrt(scene.n_vh) * _cn(rng, scene.shape)
    return s_vv, s_vh


def homogeneous_mask(scene: Scene, erode: int = 4) -> np.ndarray:
    """Largest texture-free field (no road, no point scatterer), eroded by `erode` px from its edges (ENL area)."""
    best = np.zeros(scene.shape, bool)
    point_free = np.ones(scene.shape, bool)
    for r, c in scene.points:
        point_free[max(0, r - 8):r + 9, max(0, c - 8):c + 9] = False
    for lab in np.flatnonzero(~scene.textured):
        m = binary_erosion(scene.labels == lab, iterations=erode) & point_free
        if m.sum() > best.sum():
            best = m
    return best


def ensemble_covariance(scene: Scene, n_looks: int, rng=None, thermal: bool = True):
    """Per-pixel average of k k^H over n_looks independent realisations -> (c11, c22, c12)."""
    rng = np.random.default_rng() if rng is None else rng
    a11 = np.zeros(scene.shape)
    a22 = np.zeros(scene.shape)
    a12 = np.zeros(scene.shape, complex)
    for _ in range(n_looks):
        s_vv, s_vh = observe(scene, rng, thermal=thermal)
        a11 += np.abs(s_vv) ** 2
        a22 += np.abs(s_vh) ** 2
        a12 += s_vv * np.conj(s_vh)
    return a11 / n_looks, a22 / n_looks, a12 / n_looks


def check_truth(scene: Scene, n_looks: int = 500, rng=None, thermal: bool = True) -> list[dict]:
    """Average many realisations, pool per field, and compare the recovered coherence with the designed one.

    With VH thermal noise the measured covariance is C + diag(0, n_vh); the known n_vh is subtracted from C22
    before forming gamma (this is a check of the simulator, not an estimator).
    Returns one row per field (and roads), points excluded.
    """
    e11, e22, e12 = ensemble_covariance(scene, n_looks, rng, thermal=thermal)
    if thermal:
        e22 = e22 - scene.n_vh
    not_point = np.ones(scene.shape, bool)
    for r, c in scene.points:
        not_point[r, c] = False
    rows = []
    for lab in np.unique(scene.labels):
        sel = (scene.labels == lab) & not_point
        # the designed gamma is constant within a field (texture multiplies both channels), so pool C over the field
        g_true = scene.gamma[sel][0]
        g_est = e12[sel].sum() / np.sqrt(e11[sel].sum() * e22[sel].sum())
        rows.append({"field": int(lab), "n_px": int(sel.sum()), "gamma_true": abs(g_true),
                     "gamma_est": abs(g_est), "phase_err_deg": float(np.degrees(np.angle(g_est * np.conj(g_true))))})
    pts = []
    for r, c in scene.points:
        pts.append({"field": "point", "n_px": 1, "gamma_true": abs(scene.gamma[r, c]),
                    "gamma_est": abs(e12[r, c] / np.sqrt(e11[r, c] * e22[r, c])),
                    "phase_err_deg": float(np.degrees(np.angle(e12[r, c] * np.conj(scene.gamma[r, c]))))})
    return rows + pts
