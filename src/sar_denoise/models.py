"""U-Net with a positive-semi-definite covariance head.

Input : one look, normalised by patches.normalise_pair, as real channels
            [C11, C22, Re C12, Im C12]  and (optionally) the two noise maps [noise_VV, noise_VH]
Output: raw 4-channel map (log C11, log C22, u, v). raw_to_covariance() turns it into a valid C2:
            C11 = exp(a),  C22 = exp(b)
            z = u + i v,   gamma = tanh(|z|) * z / |z|        (|gamma| < 1 always)
            C12 = gamma * sqrt(C11 C22)
so every output is a legal covariance matrix (C11, C22 > 0, |C12|^2 < C11 C22) and the coherence is a
bounded, signed (complex) quantity. The sign / phase of C12 is carried by z, never by a magnitude alone.

Normalisation layers are GroupNorm (not BatchNorm): batches are small and the patches are heavily
correlated, so batch statistics would be noisy and would leak between patches.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

HEAD_INIT_STD = 0.05
LOG_CLAMP = 15.0          # |log C| limit; data are normalised to unit mean so this is never active in practice


def _groups(ch: int, target: int = 8) -> int:
    g = min(target, ch)
    while ch % g:
        g -= 1
    return g


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(_groups(cout), cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(_groups(cout), cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Plain U-Net. Returns the RAW head output (B, 4, H, W); use raw_to_covariance() on it.

    The head has a small random init (std HEAD_INIT_STD) and a zero bias, so training starts near C11 = C22 = 1
    and gamma ~ 0. It must NOT be zero-initialised: with exactly zero weights the network sits on a saddle at
    gamma = 0 and, when the data contain incoherent areas, never leaves it (seen in simulation: the output
    stayed at gamma = 0.02 for 1500 steps, while std 0.05 recovered +0.8, -0.8 and 0 within 600).
    H and W need not be multiples of 2**depth: the input is padded (reflect) and the output cropped.
    """

    def __init__(self, in_ch: int = 6, base: int = 24, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.in_ch, self.depth = in_ch, depth
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([ConvBlock(in_ch if i == 0 else widths[i - 1], widths[i]) for i in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([ConvBlock(2 * widths[i], widths[i]) for i in range(depth)])
        self.head = nn.Conv2d(widths[0], 4, 1)
        nn.init.normal_(self.head.weight, std=HEAD_INIT_STD)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.enc[self.depth](x)
        for i in reversed(range(self.depth)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], dim=1))
        return self.head(x)[..., :h, :w]


def make_input(c2: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
    """Stack a look (B,4,H,W) and, if the model wants it, its noise maps (B,2,H,W) -> (B,6,H,W).

    Both are already on the common scale of normalise_pair, and nothing nonlinear is applied.
    """
    return c2 if noise is None else torch.cat([c2, noise], dim=1)


def raw_to_covariance(raw: torch.Tensor, eps: float = 1e-8):
    """Raw head output (B,4,H,W) -> (c11, c22, c12_re, c12_im, gamma_re, gamma_im, gamma_abs), each (B,H,W).

    gamma = tanh(|z|) z / |z| with |z| = sqrt(u^2 + v^2 + eps): smooth at z = 0, |gamma| < 1.
    """
    a = raw[:, 0].clamp(-LOG_CLAMP, LOG_CLAMP)
    b = raw[:, 1].clamp(-LOG_CLAMP, LOG_CLAMP)
    u, v = raw[:, 2], raw[:, 3]
    c11, c22 = torch.exp(a), torch.exp(b)
    m = torch.sqrt(u * u + v * v + eps)
    t = torch.tanh(m)
    g_re, g_im = t * u / m, t * v / m
    s = torch.sqrt(c11 * c22)
    return c11, c22, g_re * s, g_im * s, g_re, g_im, t


def covariance_stack(raw: torch.Tensor) -> torch.Tensor:
    """The denoised C2 in the same [C11, C22, Re C12, Im C12] layout as the data, (B,4,H,W)."""
    c11, c22, re, im, *_ = raw_to_covariance(raw)
    return torch.stack([c11, c22, re, im], dim=1)