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


def test_similarity_pairs_match_regression_target():
    """Flow and regression arms must see byte-identical pairs.

    The flow-vs-regression comparison (sweep_flow_vs_reg.yml) is only clean if
    the two arms differ *solely* in estimator, not in the similarity pairs they
    train on. ssflow.make_similarity_pair and selfsupervised.make_similarity_target
    share the same construction, so at matched (search_radius, patch_radius,
    num_similar, exclude_radius) they must produce the same tensor. This guards
    against silent drift between the two code paths.
    """
    from ctdenoiser.selfsupervised import make_similarity_target

    torch.manual_seed(0)
    noisy = torch.randn(2, 1, 24, 24)
    for exclude_radius in (1, 2, 3):
        for num_similar in (1, 2):
            flow = make_similarity_pair(
                noisy, search_radius=4, patch_radius=1,
                num_similar=num_similar, exclude_radius=exclude_radius,
            )
            reg = make_similarity_target(
                noisy, search_radius=4, patch_radius=1,
                num_similar=num_similar, exclude_radius=exclude_radius,
            )
            assert torch.equal(flow, reg), (
                f"pairing drift at exclude_radius={exclude_radius}, "
                f"num_similar={num_similar}"
            )


def test_ss_flow_loss_is_scalar_and_differentiable():
    model = SelfSupervisedFlow(num_filters=8, embed_dim=16, search_radius=2)
    noisy = torch.rand(2, 1, 32, 32)
    loss = model.ss_flow_loss(noisy)
    assert loss.ndim == 0 and loss.item() >= 0.0
    loss.backward()
    grads = [p.grad for p in model.net.parameters() if p.grad is not None]
    assert grads, "no gradients flowed into the velocity network"


def test_flow_loss_to_target_is_scalar_and_differentiable():
    # Used by the flowmatching + n2v/n2sim path: explicit endpoints, not _build_pair.
    model = SelfSupervisedFlow(num_filters=8, embed_dim=16)
    x0 = torch.rand(2, 1, 32, 32)
    x1 = torch.rand(2, 1, 32, 32)
    loss = model.flow_loss_to_target(x0, x1)
    assert loss.ndim == 0 and loss.item() >= 0.0
    loss.backward()
    grads = [p.grad for p in model.net.parameters() if p.grad is not None]
    assert grads, "no gradients flowed into the velocity network"


def test_ss_flow_loss_delegates_to_flow_loss_to_target():
    # Same endpoints -> same loss (ss_flow_loss is flow_loss_to_target on _build_pair).
    torch.manual_seed(0)
    model = SelfSupervisedFlow(num_filters=8, embed_dim=16, search_radius=2)
    noisy = torch.rand(2, 1, 32, 32)
    x0, x1 = model._build_pair(noisy)
    torch.manual_seed(123)
    a = model.flow_loss_to_target(x0, x1)
    torch.manual_seed(123)
    b = model.flow_loss_to_target(x0, x1)
    assert torch.allclose(a, b)


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
