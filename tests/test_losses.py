import numpy as np
import pytest
import torch

from sar_denoise.losses import coherence_metrics, covariance_nll, covariance_nll_map, n2n_loss
from sar_denoise.models import UNet, raw_to_covariance
from sar_denoise.simulate import simulate_look_pair


def reference_nll(c11, c22, c12, n11, n22, o11, o22, o12):
    """Explicit complex 2x2 matrices, numpy: log det Sigma + tr(Sigma^-1 C_obs)."""
    sig = np.array([[c11 + n11, c12], [np.conj(c12), c22 + n22]])
    obs = np.array([[o11, o12], [np.conj(o12), o22]])
    return float(np.log(np.linalg.det(sig).real) + np.trace(np.linalg.inv(sig) @ obs).real)


def test_closed_form_matches_explicit_matrices():
    rng = np.random.default_rng(0)
    raw = torch.randn(1, 4, 5, 6, dtype=torch.float64) * 0.7
    obs = torch.rand(1, 4, 5, 6, dtype=torch.float64) + 0.1
    obs[:, 2:] -= 0.5
    noise = torch.rand(1, 2, 5, 6, dtype=torch.float64) * 0.5 + 0.05
    got = covariance_nll_map(raw, obs, noise)
    c11, c22, re, im, *_ = raw_to_covariance(raw)
    for i in range(5):
        for j in range(6):
            want = reference_nll(c11[0, i, j].item(), c22[0, i, j].item(), complex(re[0, i, j], im[0, i, j]),
                                 noise[0, 0, i, j].item(), noise[0, 1, i, j].item(),
                                 obs[0, 0, i, j].item(), obs[0, 1, i, j].item(), complex(obs[0, 2, i, j], obs[0, 3, i, j]))
            assert got[0, i, j].item() == pytest.approx(want, rel=1e-5, abs=1e-6)


def test_invalid_pixels_contribute_nothing_even_when_nan():
    raw = torch.randn(1, 4, 8, 8, requires_grad=True)
    obs = torch.rand(1, 4, 8, 8) + 0.1
    noise = torch.rand(1, 2, 8, 8) + 0.1
    valid = torch.ones(1, 8, 8, dtype=torch.bool)
    valid[0, :3] = False
    clean = covariance_nll(raw, obs, noise, valid)
    obs_bad, noise_bad = obs.clone(), noise.clone()
    obs_bad[:, :, :3] = float("nan")
    noise_bad[:, :, :3] = float("inf")
    raw_bad = raw.detach().clone()
    raw_bad[:, :, :3] = float("nan")
    raw_bad.requires_grad_(True)
    dirty = covariance_nll(raw_bad, obs_bad, noise_bad, valid)
    assert torch.isfinite(dirty) and dirty.item() == pytest.approx(clean.item(), rel=1e-6)
    dirty.backward()
    assert torch.isfinite(raw_bad.grad).all() and raw_bad.grad[:, :, :3].abs().sum() == 0


def _fit_constant(obs, noise, steps=500, lr=0.05):
    """ML fit of ONE covariance (4 raw numbers) to many single-look pixels."""
    p = torch.zeros(4, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        covariance_nll(p.view(1, 4, 1, 1).expand(1, 4, *obs.shape[2:]), obs, noise).backward()
        opt.step()
    c11, c22, re, im, g_re, g_im, g = raw_to_covariance(p.detach().view(1, 4, 1, 1))
    return c11.item(), c22.item(), complex(g_re.item(), g_im.item())


@pytest.mark.parametrize("phase", [0.0, 2.8, np.pi / 2])
def test_likelihood_recovers_known_covariance_including_sign_and_phase(phase):
    rng = np.random.default_rng(1)
    gam = 0.7 * np.exp(1j * phase)
    c11, c22 = 1.0, 0.5
    pair = simulate_look_pair(c11, c22, gam * np.sqrt(c11 * c22), shape=(150, 200), rng=rng)
    obs = torch.from_numpy(pair["c2_a"])[None]
    noise = torch.from_numpy(pair["noise_a"])[None]
    g11, g22, g = _fit_constant(obs, noise)
    assert g11 == pytest.approx(c11, rel=0.08) and g22 == pytest.approx(c22, rel=0.08)
    assert abs(g - gam) < 0.06                                    # complex error: magnitude AND phase


def test_noise_term_removes_the_coherence_bias_that_ignoring_noise_creates():
    rng = np.random.default_rng(2)
    c11, c22, n11, n22 = 1.0, 0.5, 0.1, 0.5                      # VH at 0 dB SNR
    gam = 0.7 * np.exp(1j * 2.8)
    pair = simulate_look_pair(c11, c22, gam * np.sqrt(c11 * c22), n11, n22, shape=(150, 200), rng=rng)
    obs = torch.from_numpy(pair["c2_a"])[None]
    noise = torch.from_numpy(pair["noise_a"])[None]
    _, _, g_aware = _fit_constant(obs, noise)
    _, _, g_blind = _fit_constant(obs, torch.zeros_like(noise))   # same data, noise ignored
    assert abs(g_aware - gam) < 0.07
    assert abs(g_blind) < 0.55 < 0.65 < abs(g_aware)              # blind fit sees ~0.47, aware fit ~0.70


def test_n2n_loss_is_finite_and_trains_the_tiny_unet():
    rng = np.random.default_rng(3)
    pair = simulate_look_pair(1.0, 0.5, 0.5 * np.sqrt(0.5), 0.05, 0.1, shape=(16, 32), rng=rng)
    batch = {k: torch.from_numpy(v)[None] for k, v in pair.items()}
    batch["valid"] = torch.ones(1, 16, 32, dtype=torch.bool)
    net = UNet(in_ch=6, base=8, depth=2)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first, _ = n2n_loss(net, batch)
    for _ in range(40):
        opt.zero_grad()
        loss, parts = n2n_loss(net, batch)
        loss.backward()
        opt.step()
    assert torch.isfinite(first) and loss.item() < first.item() and set(parts) == {"a2b", "b2a"}


def test_coherence_metrics_phase_error_wraps_correctly():
    raw = torch.zeros(1, 4, 1, 2)
    raw[0, 2, 0, 0], raw[0, 3, 0, 0] = -2.0, 0.05              # phase just below +pi
    raw[0, 2, 0, 1], raw[0, 3, 0, 1] = -2.0, -0.05             # phase just above -pi
    ref = torch.tensor([1.0, 1.0, -0.96, 0.0]).view(1, 4, 1, 1).expand(1, 4, 1, 2).contiguous()
    m = coherence_metrics(raw, ref)
    assert m["phase_abs_err_rad"].item() < 0.1                 # a naive difference would give ~2*pi