import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sar_denoise.complex_models import (ComplexConv2d, ComplexConvTranspose2d, ComplexGroupNorm, ComplexUNet, CReLU,
                                        join, split)
from sar_denoise.evaluate import covariance_to_raw
from sar_denoise.models import raw_to_covariance
from sar_denoise.sim_losses import kl_loss, kl_map


def _to_c(x):
    r, i = split(x)
    return torch.complex(r, i)


def test_complex_conv_matches_native_complex_conv():
    torch.manual_seed(0)
    conv = ComplexConv2d(3, 5, 3, padding=1)
    with torch.no_grad():
        conv.bias_re.normal_()
        conv.bias_im.normal_()
    x = torch.randn(2, 6, 9, 11)
    w = torch.complex(conv.a.weight, conv.b.weight)
    bias = torch.complex(conv.bias_re, conv.bias_im)
    ref = F.conv2d(_to_c(x), w, bias, padding=1)
    torch.testing.assert_close(_to_c(conv(x)), ref, rtol=1e-4, atol=1e-5)


def test_complex_transposed_conv_matches_native():
    torch.manual_seed(1)
    conv = ComplexConvTranspose2d(4, 2, 2, stride=2, bias=False)
    x = torch.randn(1, 8, 5, 6)
    w = torch.complex(conv.a.weight, conv.b.weight)
    ref = F.conv_transpose2d(_to_c(x), w, stride=2)
    out = conv(x)
    assert out.shape == (1, 4, 10, 12)
    torch.testing.assert_close(_to_c(out), ref, rtol=1e-4, atol=1e-5)


def test_complex_conv_is_complex_linear():
    """Without bias, f(e^{j phi} x) = e^{j phi} f(x): it really is a complex-linear map, not just 2 real channels."""
    torch.manual_seed(2)
    conv = ComplexConv2d(2, 3, 3, padding=1, bias=False)
    x = _to_c(torch.randn(1, 4, 8, 8))
    rot = torch.exp(torch.tensor(1j * 0.7, dtype=torch.complex64))
    y = _to_c(conv(join(x.real, x.imag)))
    xr = x * rot
    y_rot = _to_c(conv(join(xr.real, xr.imag)))
    torch.testing.assert_close(y_rot, y * rot, rtol=1e-4, atol=1e-5)


def test_crelu_acts_on_real_and_imag_parts():
    x = join(torch.tensor([[[[-1.0, 2.0]]]]), torch.tensor([[[[3.0, -4.0]]]]))
    r, i = split(CReLU()(x))
    assert r.flatten().tolist() == [0.0, 2.0] and i.flatten().tolist() == [3.0, 0.0]


def test_complex_groupnorm_whitens():
    torch.manual_seed(3)
    n = ComplexGroupNorm(4, groups=1)
    re = 3 + 2 * torch.randn(2, 4, 16, 16)
    im = 0.5 * re + 0.1 * torch.randn(2, 4, 16, 16) - 1       # strongly correlated re/im, nonzero mean
    yr, yi = split(n(join(re, im)))
    for b in range(2):
        r, i = yr[b].flatten(), yi[b].flatten()
        assert abs(r.mean()) < 1e-4 and abs(i.mean()) < 1e-4
        assert abs((r * r).mean() - 0.5) < 1e-3 and abs((i * i).mean() - 0.5) < 1e-3   # init scale 1/sqrt 2
        assert abs((r * i).mean()) < 1e-3


@pytest.mark.parametrize("hw", [(32, 32), (30, 45)])
def test_complex_unet_shapes_and_valid_covariance(hw):
    torch.manual_seed(4)
    net = ComplexUNet(in_ch=2, base=4, depth=2)
    raw = net(torch.randn(2, 4, *hw))
    assert raw.shape == (2, 4, *hw)
    c11, c22, re, im, *_ = raw_to_covariance(raw)
    assert torch.all(c11 > 0) and torch.all(c22 > 0)
    assert torch.all(re ** 2 + im ** 2 < c11 * c22)


def test_complex_unet_rejects_wrong_channels():
    with pytest.raises(ValueError):
        ComplexUNet(in_ch=2, base=4, depth=1)(torch.randn(1, 3, 8, 8))


def _random_c(shape, seed):
    g = torch.Generator().manual_seed(seed)
    c11 = torch.exp(torch.randn(shape, generator=g))
    c22 = torch.exp(torch.randn(shape, generator=g))
    gam = 0.95 * torch.rand(shape, generator=g) * torch.exp(1j * 6.28 * torch.rand(shape, generator=g))
    c12 = gam * torch.sqrt(c11 * c22)
    return torch.stack([c11, c22, c12.real, c12.imag], dim=1)


def test_kl_zero_at_truth_positive_elsewhere():
    c = _random_c((2, 8, 8), 0)
    raw = covariance_to_raw(c.double()).float()
    assert kl_map(raw, c).abs().max() < 1e-3
    other = covariance_to_raw(_random_c((2, 8, 8), 1).double()).float()
    k = kl_map(other, c)
    assert torch.all(k > 0)
    # each part alone matters: wrong power, wrong |gamma|, wrong phase
    for idx, delta in ((0, 0.5), (2, 0.3), (3, -0.3)):
        r = raw.clone()
        r[:, idx] += delta
        assert kl_loss(r, c) > 1e-3


def test_kl_matches_matrix_formula():
    c = _random_c((1, 3, 3), 2).double()
    raw = covariance_to_raw(_random_c((1, 3, 3), 3).double())
    c11h, c22h, reh, imh, *_ = raw_to_covariance(raw)
    k = kl_map(raw, c)
    for (r, s) in [(0, 0), (1, 2), (2, 1)]:
        def mat(a11, a22, re, im):
            return torch.tensor([[a11, complex(re, im)], [complex(re, -im), a22]], dtype=torch.complex128)
        ct = mat(*(c[0, j, r, s].item() for j in range(4)))
        ch = mat(c11h[0, r, s].item(), c22h[0, r, s].item(), reh[0, r, s].item(), imh[0, r, s].item())
        p = torch.linalg.solve(ch, ct)
        ref = torch.trace(p).real - torch.log(torch.linalg.det(p).real) - 2
        assert abs(k[0, r, s].item() - ref.item()) < 1e-6


def test_kl_invariant_to_diagonal_scaling():
    """C -> D C D on both truth and estimate leaves KL unchanged (so normalisation / brightness do not matter)."""
    c = _random_c((1, 6, 6), 4).double()
    raw = covariance_to_raw(_random_c((1, 6, 6), 5).double())
    d1, d2 = 7.0, 0.01
    scale = torch.tensor([d1 ** 2, d2 ** 2, d1 * d2, d1 * d2], dtype=torch.float64)[None, :, None, None]
    raw2 = raw.clone()
    raw2[:, 0] += 2 * np.log(d1)
    raw2[:, 1] += 2 * np.log(d2)
    torch.testing.assert_close(kl_map(raw2, c * scale), kl_map(raw, c))


def test_kl_gradient_is_finite_for_extreme_heads():
    c = _random_c((1, 4, 4), 6)
    raw = torch.tensor([10.0, -10.0, 30.0, -30.0])[None, :, None, None].expand(1, 4, 4, 4).clone().requires_grad_()
    kl_loss(raw, c).backward()
    assert torch.isfinite(raw.grad).all()
