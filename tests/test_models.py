import pytest
import torch

from sar_denoise.models import UNet, covariance_stack, make_input, raw_to_covariance


def test_unet_shape_and_padding():
    net = UNet(in_ch=6, base=8, depth=3)
    for h, w in [(64, 512), (37, 101)]:          # the second is not a multiple of 2**depth
        out = net(torch.randn(2, 6, h, w))
        assert out.shape == (2, 4, h, w)


def test_head_starts_at_unit_power_and_zero_coherence():
    net = UNet(in_ch=6, base=8, depth=2)
    c11, c22, re, im, g_re, g_im, g = raw_to_covariance(net(torch.randn(1, 6, 16, 32)))
    assert torch.allclose(c11, torch.ones_like(c11)) and torch.allclose(c22, torch.ones_like(c22))
    assert g.max() < 1e-3 and re.abs().max() < 1e-3


def test_raw_to_covariance_is_always_a_valid_covariance():
    raw = torch.randn(3, 4, 20, 20) * 8.0                       # large values on purpose
    c11, c22, re, im, g_re, g_im, g = raw_to_covariance(raw)
    assert (c11 > 0).all() and (c22 > 0).all()
    assert (g < 1.0).all() or torch.allclose(g.max(), torch.tensor(1.0))     # tanh may round to 1.0 in float32
    assert ((c11 * c22 - re ** 2 - im ** 2) >= -1e-4 * c11 * c22).all()       # positive semi-definite


def test_phase_and_sign_are_carried_by_z():
    u = torch.tensor([[[[2.0, -2.0, 0.0, 0.0]]]])
    v = torch.tensor([[[[0.0, 0.0, 2.0, -2.0]]]])
    raw = torch.cat([torch.zeros_like(u), torch.zeros_like(u), u, v], dim=1)
    _, _, re, im, g_re, g_im, g = raw_to_covariance(raw)
    assert torch.allclose(torch.atan2(im, re), torch.atan2(v[:, 0], u[:, 0]), atol=1e-4)
    assert re[0, 0, 0] > 0 and re[0, 0, 1] < 0            # opposite signs of Re C12 survive
    assert torch.allclose(g, torch.tanh(torch.tensor(2.0)).expand_as(g), atol=1e-5)


def test_covariance_stack_layout_and_make_input():
    raw = torch.randn(2, 4, 8, 8)
    st = covariance_stack(raw)
    c11, c22, re, im, *_ = raw_to_covariance(raw)
    assert st.shape == (2, 4, 8, 8) and torch.equal(st[:, 2], re) and torch.equal(st[:, 1], c22)
    assert make_input(torch.zeros(1, 4, 4, 4)).shape[1] == 4
    assert make_input(torch.zeros(1, 4, 4, 4), torch.zeros(1, 2, 4, 4)).shape[1] == 6


def test_gradients_reach_every_parameter_after_one_step():
    net = UNet(in_ch=6, base=8, depth=2)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    raw = net(torch.randn(2, 6, 16, 32))
    raw.square().mean().backward()                  # only the head gets gradient on step 0 (zero-init head)
    opt.step()
    opt.zero_grad()
    net(torch.randn(2, 6, 16, 32)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())