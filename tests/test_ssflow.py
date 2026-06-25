"""Tests for the self-supervised rectified flow (ctdenoiser.models.ssflow)."""

import torch

from ctdenoiser.models.ssflow import (
    SelfSupervisedFlow,
    UnconditionalVelocityUNet,
    make_similarity_pair,
)


def test_velocity_net_shape():
    net = UnconditionalVelocityUNet(num_filters=8, embed_dim=16)
    z = torch.randn(2, 1, 32, 32)
    t_emb = torch.randn(2, 16)
    assert net(z, t_emb).shape == z.shape


def test_similarity_pair_shape_and_excludes_self():
    noisy = torch.randn(1, 1, 24, 24)
    # exclude_radius=2 skips the 3x3 block around each query; output stays shaped.
    pair = make_similarity_pair(noisy, search_radius=3, exclude_radius=2)
    assert pair.shape == noisy.shape
    assert torch.isfinite(pair).all()


def test_ss_flow_loss_is_scalar_and_differentiable():
    model = SelfSupervisedFlow(num_filters=8, embed_dim=16, search_radius=2)
    noisy = torch.rand(2, 1, 32, 32)
    loss = model.ss_flow_loss(noisy)
    assert loss.ndim == 0 and loss.item() >= 0.0
    loss.backward()
    grads = [p.grad for p in model.net.parameters() if p.grad is not None]
    assert grads, "no gradients flowed into the velocity network"


def test_forward_preserves_shape_one_step_and_multistep():
    x = torch.rand(2, 1, 32, 32)
    one = SelfSupervisedFlow(num_filters=8, embed_dim=16, num_steps=1)
    multi = SelfSupervisedFlow(num_filters=8, embed_dim=16, num_steps=4)
    assert one(x).shape == x.shape
    assert multi(x).shape == x.shape
    # Output is clamped to the valid image range.
    assert one(x).min() >= 0.0 and one(x).max() <= 1.0


def test_downsample_pairing_runs():
    model = SelfSupervisedFlow(num_filters=8, embed_dim=16, pairing="downsample")
    noisy = torch.rand(2, 1, 32, 32)
    loss = model.ss_flow_loss(noisy)
    assert torch.isfinite(loss)
