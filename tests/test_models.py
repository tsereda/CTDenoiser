import pytest

torch = pytest.importorskip("torch")

from ctdenoiser.inference import overlapped_inference
from ctdenoiser.metrics import gmsd, nps_ratio, psnr, rmse, ssim
from ctdenoiser.models import CTformer, DnCNN, FlowMatching, REDCNN, UNet


@pytest.mark.parametrize("size", [64, 128])
def test_ctformer_preserves_shape(size):
    model = CTformer(embed_dim=16)
    x = torch.randn(2, 1, size, size)
    assert model(x).shape == x.shape


def test_redcnn_preserves_shape():
    model = REDCNN(num_filters=16)
    x = torch.randn(2, 1, 64, 64)
    assert model(x).shape == x.shape


def test_overlapped_inference_shape_and_coverage():
    model = REDCNN(num_filters=8)
    img = torch.randn(1, 1, 100, 130)
    out = overlapped_inference(model, img, patch_size=64, margin=16)
    assert out.shape == img.shape
    assert torch.isfinite(out).all()


def test_overlapped_inference_covers_border():
    """An identity model must reproduce the input everywhere, including the
    outer margin frame — regression for the dropped-border bug."""
    identity = torch.nn.Identity()
    img = torch.randn(1, 1, 128, 192)
    out = overlapped_inference(identity, img, patch_size=64, margin=16)
    assert torch.allclose(out, img, atol=1e-5)


def test_dncnn_preserves_shape():
    model = DnCNN(num_filters=16, num_layers=5)
    x = torch.randn(2, 1, 64, 64)
    assert model(x).shape == x.shape


def test_unet_preserves_shape():
    model = UNet(base_filters=8)
    x = torch.randn(2, 1, 64, 64)
    assert model(x).shape == x.shape


def test_flowmatching_preserves_shape():
    model = FlowMatching(num_filters=16, embed_dim=16, num_steps=2)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == x.shape


def test_flowmatching_flow_loss():
    model = FlowMatching(num_filters=16, embed_dim=16, num_steps=2)
    x0 = torch.randn(2, 1, 64, 64)
    x1 = torch.randn(2, 1, 64, 64)
    loss = model.flow_loss(x0, x1)
    assert loss.ndim == 0
    assert loss.item() >= 0
    assert loss.requires_grad


def test_flowmatching_overlapped_inference():
    model = FlowMatching(num_filters=8, embed_dim=8, num_steps=2)
    img = torch.randn(1, 1, 100, 130)
    out = overlapped_inference(model, img, patch_size=64, margin=16)
    assert out.shape == img.shape
    assert torch.isfinite(out).all()


def test_new_metrics():
    a = torch.rand(2, 1, 64, 64)
    b = torch.rand(2, 1, 64, 64)
    assert gmsd(a, b) >= 0.0
    assert nps_ratio(a, b) >= 0.0


def test_metrics_identity():
    a = torch.rand(1, 1, 32, 32)
    assert rmse(a, a) == pytest.approx(0.0, abs=1e-6)
    assert psnr(a, a) == float("inf")
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-4)


_EXPECTED_METRIC_KEYS = {
    "psnr", "psnr_std", "ssim", "ssim_std", "rmse", "rmse_std",
    "gmsd", "gmsd_std", "nps_ratio", "nps_ratio_std",
}


def _synthetic_loader(length=4, patch_size=32):
    from torch.utils.data import DataLoader

    from ctdenoiser.data import SyntheticCTDataset

    ds = SyntheticCTDataset(length=length, patch_size=patch_size)
    return DataLoader(ds, batch_size=1, shuffle=False)


def test_eval_paths_report_std():
    """evaluate / identity_baseline / run_zsn2n_eval all emit per-slice std."""
    import types

    from ctdenoiser.train import evaluate, identity_baseline, run_zsn2n_eval

    device = torch.device("cpu")
    loader = _synthetic_loader()

    model = REDCNN(num_filters=8)
    eval_out = evaluate(model, loader, device, full_slice=False, patch_size=32)
    assert _EXPECTED_METRIC_KEYS <= set(eval_out)

    base_out = identity_baseline(loader, device)
    assert _EXPECTED_METRIC_KEYS <= set(base_out)

    args = types.SimpleNamespace(
        zsn2n_iters=2, zsn2n_lr=1e-3, zsn2n_channels=4, seed=0
    )
    zs_out = run_zsn2n_eval(loader, device, args)
    assert _EXPECTED_METRIC_KEYS <= set(zs_out)

    # std is a real non-negative spread, not a placeholder.
    for out in (eval_out, base_out, zs_out):
        for k in ("psnr_std", "ssim_std", "gmsd_std"):
            assert out[k] >= 0.0


def test_synthetic_dataset_shapes():
    from ctdenoiser.data import SyntheticCTDataset

    ds = SyntheticCTDataset(length=8, patch_size=32)
    assert len(ds) == 8
    low, clean = ds[0]
    assert low.shape == (1, 32, 32)
    assert clean.shape == (1, 32, 32)
    assert 0.0 <= low.min() and low.max() <= 1.0
