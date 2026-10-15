"""Train the tiny U-Net on a simulated scene with a KNOWN covariance and check what it learned.

Scene (32 x 96 looks): three zones, gamma = +0.8 | 0 | -0.8 (phase 0, no coherence, phase pi, as seen on the real
urban blocks and the water), with VH noise at about the signal level so the noise term matters. The zero zone
is essential: a network whose head starts at exactly zero never leaves gamma = 0 when it is present. A fresh pair of looks is
drawn every step (Noise2Noise: the target look never contains the input's speckle).
"""
import numpy as np
import torch

from sar_denoise.losses import coherence_metrics, n2n_loss
from sar_denoise.models import UNet, make_input, raw_to_covariance
from sar_denoise.simulate import simulate_look_pair

H, W = 32, 96
C11, C22, N11, N22 = 1.0, 0.4, 0.05, 0.3


def scene():
    gam = np.select([np.arange(W) < W // 3, np.arange(W) < 2 * W // 3], [0.8, 0.0], -0.8) * np.ones((H, 1))
    return gam, gam * np.sqrt(C11 * C22)


def batch(rng, n):
    _, c12 = scene()
    pairs = [simulate_look_pair(C11, C22, c12, N11, N22, shape=(H, W), rng=rng) for _ in range(n)]
    out = {k: torch.from_numpy(np.stack([p[k] for p in pairs])) for k in pairs[0]}
    out["valid"] = torch.ones(n, H, W, dtype=torch.bool)
    return out


def test_unet_learns_covariance_with_sign_from_noisy_looks():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = UNet(in_ch=6, base=8, depth=2)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(400):
        opt.zero_grad()
        loss, _ = n2n_loss(net, batch(rng, 8))
        loss.backward()
        opt.step()

    test = batch(np.random.default_rng(123), 4)
    with torch.no_grad():
        raw = net(make_input(test["c2_a"], test["noise_a"]))
    c11, c22, re, im, g_re, g_im, g = raw_to_covariance(raw)
    gam, _ = scene()
    third = W // 3
    zones = [slice(8, third - 8), slice(third + 8, 2 * third - 8), slice(2 * third + 8, W - 8)]
    gz = [complex(g_re[:, 4:-4, z].mean(), g_im[:, 4:-4, z].mean()) for z in zones]
    assert abs(gz[0] - 0.8) < 0.2 and abs(gz[1]) < 0.2 and abs(gz[2] + 0.8) < 0.2     # magnitude, zero AND sign
    assert abs(c11[:, 4:-4, 8:-8].mean().item() - C11) < 0.25 and abs(c22[:, 4:-4, 8:-8].mean().item() - C22) < 0.12

    ref = torch.zeros(4, 4, H, W)
    ref[:, 0], ref[:, 1] = C11, C22
    ref[:, 2] = torch.from_numpy(gam * np.sqrt(C11 * C22)).float()
    m = coherence_metrics(raw, ref)
    assert m["gamma_complex_rmse"].item() < 0.35