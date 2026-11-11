"""Complex-valued U-Net (Trabelsi et al. 2018, "Deep Complex Networks") with the PSD covariance head.

Input : the two complex single-look pixels per pixel, S_VV and S_VH (normalised per patch, see complex_input),
        packed as a REAL tensor (B, 2*C, H, W) = [Re of the C complex channels, Im of the C complex channels].
        MPS has poor complex-dtype support, so every complex op is written as real ops on (re, im) pairs.
Output: raw (B, 4, H, W) = (log C11, log C22, u, v) -> models.raw_to_covariance (C11, C22 > 0, |gamma| < 1).

Building blocks (W = A + iB, x = x_r + i x_i):
  ComplexConv2d          W * x = (A*x_r - B*x_i) + i (B*x_r + A*x_i), plus a complex bias; computed as ONE real
                         conv with the block weight [[A, -B], [B, A]] (4x faster on MPS than separate calls)
  ComplexConvTranspose2d same algebra with transposed convolutions (upsampling)
  CReLU                  ReLU(x_r) + i ReLU(x_i)
  ComplexGroupNorm       per sample and channel group: subtract the complex mean, whiten the 2x2 covariance of
                         (re, im) (Trabelsi's complex BN, but with group statistics: batches are small), then a
                         per-channel 2x2 symmetric scale and a complex shift
  avg-pool               linear, so applying it to re and im separately IS the complex average pool
  head                   real 1x1 conv on [Re, Im] of the last features -> 4 raw outputs

Note on phase: a single-look C2 (the real U-Net input) is blind to a common phase e^{j phi} of S_VV and S_VH;
this network is not, so training applies a random common phase (augmentation) to teach it that this phase is
meaningless. The output covariance does not depend on it.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .models import HEAD_INIT_STD, _groups


def complex_input(s_vv_re, s_vv_im, s_vh_re, s_vh_im) -> torch.Tensor:
    """(B,H,W) parts -> (B, 4, H, W) = [Re S_VV, Re S_VH, Im S_VV, Im S_VH] (2 complex channels)."""
    return torch.stack([s_vv_re, s_vh_re, s_vv_im, s_vh_im], dim=1)


def split(x: torch.Tensor):
    c = x.shape[1] // 2
    return x[:, :c], x[:, c:]


def join(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    return torch.cat([re, im], dim=1)


class ComplexConv2d(nn.Module):
    """Complex convolution from cin to cout complex channels, implemented with two real convolutions."""

    transposed = False

    def __init__(self, cin: int, cout: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        conv = nn.ConvTranspose2d if self.transposed else nn.Conv2d
        self.a = conv(cin, cout, kernel_size, stride=stride, padding=padding, bias=False)   # real part of W
        self.b = conv(cin, cout, kernel_size, stride=stride, padding=padding, bias=False)   # imaginary part
        # complex Glorot-like init (Trabelsi): Var(A) = Var(B) = 1 / (2 * fan_in)   [ConvTranspose: weight is (cin, cout, k, k)]
        fan_in = cin * kernel_size * kernel_size
        for w in (self.a.weight, self.b.weight):
            nn.init.normal_(w, std=1.0 / math.sqrt(2.0 * fan_in))
        self.bias_re = nn.Parameter(torch.zeros(cout)) if bias else None
        self.bias_im = nn.Parameter(torch.zeros(cout)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # one real convolution with the block weight [[A, -B], [B, A]] on the packed [re; im] channels
        # (for a transposed conv the weight is (in, out, k, k), so the block is [[A, B], [-B, A]])
        a, b = self.a.weight, self.b.weight
        if self.transposed:
            w = torch.cat([torch.cat([a, b], 1), torch.cat([-b, a], 1)], 0)
            y = F.conv_transpose2d(x, w, stride=self.a.stride, padding=self.a.padding)
        else:
            w = torch.cat([torch.cat([a, -b], 1), torch.cat([b, a], 1)], 0)
            y = F.conv2d(x, w, stride=self.a.stride, padding=self.a.padding)
        if self.bias_re is not None:
            y = y + torch.cat([self.bias_re, self.bias_im])[:, None, None]
        return y


class ComplexConvTranspose2d(ComplexConv2d):
    transposed = True


class CReLU(nn.Module):
    def forward(self, x):
        return F.relu(x)          # ReLU on Re and Im separately == ReLU on the packed real tensor


class ComplexGroupNorm(nn.Module):
    """Complex whitening normalisation with group statistics (per sample), Trabelsi-style affine."""

    def __init__(self, channels: int, groups: int | None = None, eps: float = 1e-5):
        super().__init__()
        self.c, self.g, self.eps = channels, groups or _groups(channels), eps
        self.g_rr = nn.Parameter(torch.full((channels,), 1 / math.sqrt(2)))
        self.g_ii = nn.Parameter(torch.full((channels,), 1 / math.sqrt(2)))
        self.g_ri = nn.Parameter(torch.zeros(channels))
        self.b_re = nn.Parameter(torch.zeros(channels))
        self.b_im = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Written with bmm so the big tensor is touched only a few times (about 4x faster on MPS than elementwise):
        #   statistics: per (sample, group) mean m and 2x2 covariance V of (re, im) from one Gram matrix;
        #   whitening W = V^{-1/2} and the affine Gamma are folded into one per-(sample, channel) 2x2 matrix
        #   P = Gamma W and offset q = beta - P m, applied as y = P x + q with one baddbmm.
        b, c2, h, w = x.shape
        c, g = c2 // 2, self.g
        cg, hw = c // g, h * w
        xs = x.reshape(b, 2, g, cg * hw).transpose(1, 2).reshape(b * g, 2, cg * hw)
        m = xs.mean(-1, keepdim=True)                                            # (bg, 2, 1)
        v = torch.bmm(xs, xs.transpose(1, 2)) / (cg * hw) - m @ m.transpose(1, 2)
        vrr, vii, vri = v[:, 0, 0] + self.eps, v[:, 1, 1] + self.eps, v[:, 0, 1]
        # inverse square root of [[vrr, vri], [vri, vii]] in closed form
        s = torch.sqrt((vrr * vii - vri * vri).clamp_min(self.eps * self.eps))
        t = torch.sqrt(vrr + vii + 2 * s)
        k = 1.0 / (s * t)
        wm = torch.stack([torch.stack([(vii + s) * k, -vri * k], -1),
                          torch.stack([-vri * k, (vrr + s) * k], -1)], -2)       # (bg, 2, 2)
        wm = wm.view(b, g, 1, 2, 2).expand(b, g, cg, 2, 2).reshape(b, c, 2, 2)
        gam = torch.stack([torch.stack([self.g_rr, self.g_ri], -1),
                           torch.stack([self.g_ri, self.g_ii], -1)], -2)         # (c, 2, 2), symmetric
        p = gam @ wm                                                             # (b, c, 2, 2)
        mm = m.view(b, g, 1, 2, 1).expand(b, g, cg, 2, 1).reshape(b, c, 2, 1)
        q = torch.stack([self.b_re, self.b_im], -1)[..., None] - p @ mm          # (b, c, 2, 1)
        xc = x.reshape(b, 2, c, hw).transpose(1, 2).reshape(b * c, 2, hw)
        y = torch.baddbmm(q.reshape(b * c, 2, 1), p.reshape(b * c, 2, 2), xc)
        return y.view(b, c, 2, hw).transpose(1, 2).reshape(b, c2, h, w)


class ComplexBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            ComplexConv2d(cin, cout, 3, padding=1), ComplexGroupNorm(cout), CReLU(),
            ComplexConv2d(cout, cout, 3, padding=1), ComplexGroupNorm(cout), CReLU(),
        )

    def forward(self, x):
        return self.net(x)


def _cat_complex(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ar, ai = split(a)
    br, bi = split(b)
    return join(torch.cat([ar, br], 1), torch.cat([ai, bi], 1))


class ComplexUNet(nn.Module):
    """Complex U-Net. in_ch / base are COMPLEX channel counts. Returns the raw head output (B, 4, H, W)."""

    def __init__(self, in_ch: int = 2, base: int = 16, depth: int = 3):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.in_ch, self.depth = in_ch, depth
        widths = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([ComplexBlock(in_ch if i == 0 else widths[i - 1], widths[i]) for i in range(depth + 1)])
        self.up = nn.ModuleList([ComplexConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([ComplexBlock(2 * widths[i], widths[i]) for i in range(depth)])
        self.head = nn.Conv2d(2 * widths[0], 4, 1)
        nn.init.normal_(self.head.weight, std=HEAD_INIT_STD)     # not zero: see models.UNet
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 2 * self.in_ch:
            raise ValueError(f"expected {2 * self.in_ch} real channels (re, im of {self.in_ch} complex), got {x.shape[1]}")
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.enc[self.depth](x)
        for i in reversed(range(self.depth)):
            x = self.up[i](x)
            x = self.dec[i](_cat_complex(x, skips[i]))
        return self.head(x)[..., :h, :w]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
