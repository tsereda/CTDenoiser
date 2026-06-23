"""Tests for the self-supervised (N2V) and zero-shot (ZS-N2N) training methods."""

import pytest

torch = pytest.importorskip("torch")

from ctdenoiser.models import REDCNN
from ctdenoiser.selfsupervised import (
    make_blind_spot_mask,
    make_similarity_target,
    n2sim_training_step,
    n2v_training_step,
    replace_with_neighbors,
)
from ctdenoiser.zeroshot import (
    AttentionBilateralFilter,
    Filter2NoiseNetwork,
    ZSN2NNetwork,
    denoise_image,
    denoise_image_f2n,
    pair_downsampler,
    zsn2n_loss,
)


# ----- Noise2Void (masked self-supervision) -----

def test_blind_spot_mask_shape_and_nonempty():
    g = torch.Generator().manual_seed(0)
    mask = make_blind_spot_mask((4, 1, 32, 32), 0.05, generator=g)
    assert mask.shape == (4, 1, 32, 32)
    assert mask.dtype == torch.bool
    # at least one blind spot per image so the masked loss is always defined
    assert mask.reshape(4, -1).any(dim=1).all()


def test_blind_spot_mask_tiny_patch_still_has_a_spot():
    # mask_fraction tiny on an 8x8 patch -> the >=1 guarantee must kick in
    g = torch.Generator().manual_seed(1)
    mask = make_blind_spot_mask((2, 1, 8, 8), 0.0, generator=g)
    assert mask.reshape(2, -1).any(dim=1).all()


def test_replace_with_neighbors_only_changes_masked():
    img = torch.rand(2, 1, 16, 16)
    mask = make_blind_spot_mask(img.shape, 0.1)
    out = replace_with_neighbors(img, mask)
    assert out.shape == img.shape
    # unmasked pixels are untouched
    assert torch.equal(out[~mask], img[~mask])


def test_n2v_loss_runs_and_backprops():
    model = REDCNN(num_filters=8)
    low = torch.rand(2, 1, 32, 32)
    loss = n2v_training_step(model, low, mask_fraction=0.1)
    assert loss.ndim == 0
    assert loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_n2v_works_on_small_patch():
    model = REDCNN(num_filters=4)
    loss = n2v_training_step(model, torch.rand(1, 1, 8, 8), mask_fraction=0.2)
    loss.backward()  # must not raise


# ----- Noise2Sim (similarity-based self-supervision) -----

def test_similarity_target_shape_and_finite():
    noisy = torch.rand(2, 1, 16, 16)
    target = make_similarity_target(noisy, search_radius=3, patch_radius=1)
    assert target.shape == noisy.shape
    assert torch.isfinite(target).all()


def test_similarity_target_picks_from_image_values():
    # Every output value must be an actual pixel drawn from the search window,
    # never the pixel itself for a constant-plus-unique-center image.
    noisy = torch.zeros(1, 1, 8, 8)
    noisy[0, 0, 4, 4] = 5.0  # a lone outlier
    target = make_similarity_target(noisy, search_radius=2, patch_radius=0, num_similar=1)
    # the outlier's best match is a (different) background pixel -> 0, not 5
    assert target[0, 0, 4, 4].item() == 0.0


def test_similarity_target_num_similar_averages():
    noisy = torch.rand(1, 1, 16, 16)
    t1 = make_similarity_target(noisy, search_radius=3, num_similar=1)
    t4 = make_similarity_target(noisy, search_radius=3, num_similar=4)
    assert t1.shape == t4.shape
    # averaging more matches generally changes (smooths) the target
    assert not torch.equal(t1, t4)


def test_n2sim_loss_runs_and_backprops():
    model = REDCNN(num_filters=8)
    low = torch.rand(2, 1, 32, 32)
    loss = n2sim_training_step(model, low, search_radius=3)
    assert loss.ndim == 0
    assert loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_n2sim_works_on_small_patch():
    model = REDCNN(num_filters=4)
    loss = n2sim_training_step(model, torch.rand(1, 1, 8, 8), search_radius=2)
    loss.backward()  # must not raise


# ----- Zero-Shot Noise2Noise -----

def test_pair_downsampler_halves_resolution():
    d1, d2 = pair_downsampler(torch.rand(1, 1, 32, 32))
    assert d1.shape == (1, 1, 16, 16)
    assert d2.shape == (1, 1, 16, 16)


def test_zsn2n_network_preserves_shape():
    net = ZSN2NNetwork(num_channels=4)
    x = torch.rand(1, 1, 32, 32)
    assert net(x).shape == x.shape


def test_zsn2n_loss_scalar_and_differentiable():
    net = ZSN2NNetwork(num_channels=4)
    loss = zsn2n_loss(net, torch.rand(1, 1, 32, 32))
    assert loss.ndim == 0
    loss.backward()


def test_denoise_image_reduces_noise():
    g = torch.Generator().manual_seed(0)
    clean = torch.zeros(1, 1, 32, 32)
    clean[..., 8:24, 8:24] = 1.0
    noisy = (clean + 0.2 * torch.randn(1, 1, 32, 32, generator=g)).clamp(0.0, 1.0)
    out = denoise_image(noisy, num_iters=200, num_channels=8, seed=0)
    assert out.shape == noisy.shape
    assert torch.isfinite(out).all()
    # denoised image is closer to the clean signal than the noisy input
    assert ((out - clean) ** 2).mean() < ((noisy - clean) ** 2).mean()


def test_denoise_image_handles_odd_dims():
    noisy = torch.rand(1, 1, 31, 33)
    out = denoise_image(noisy, num_iters=5, num_channels=4)
    # odd dims cropped to even before downsampling
    assert out.shape == (1, 1, 30, 32)


# ----- Filter2Noise (attention-guided bilateral filtering) -----

def test_attention_bilateral_filter_preserves_shape():
    f = AttentionBilateralFilter(radius=2, num_channels=4)
    x = torch.rand(2, 1, 24, 24)
    out = f(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_filter2noise_network_stacks_and_backprops():
    net = Filter2NoiseNetwork(num_layers=2, radius=2, num_channels=4)
    x = torch.rand(1, 1, 32, 32)
    loss = zsn2n_loss(net, x)
    assert loss.ndim == 0
    loss.backward()
    assert any(p.grad is not None for p in net.parameters())


def test_denoise_image_f2n_reduces_noise():
    g = torch.Generator().manual_seed(0)
    clean = torch.zeros(1, 1, 32, 32)
    clean[..., 8:24, 8:24] = 1.0
    noisy = (clean + 0.2 * torch.randn(1, 1, 32, 32, generator=g)).clamp(0.0, 1.0)
    out = denoise_image_f2n(noisy, num_iters=200, num_layers=2, radius=2, num_channels=8, seed=0)
    assert out.shape == noisy.shape
    assert torch.isfinite(out).all()
    assert ((out - clean) ** 2).mean() < ((noisy - clean) ** 2).mean()


def test_denoise_image_f2n_handles_odd_dims():
    noisy = torch.rand(1, 1, 31, 33)
    out = denoise_image_f2n(noisy, num_iters=5, num_layers=1, radius=2, num_channels=4)
    assert out.shape == (1, 1, 30, 32)
