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


def test_synthetic_dataset_shapes():
    from ctdenoiser.data import SyntheticCTDataset

    ds = SyntheticCTDataset(length=8, patch_size=32)
    assert len(ds) == 8
    low, clean = ds[0]
    assert low.shape == (1, 32, 32)
    assert clean.shape == (1, 32, 32)
    assert 0.0 <= low.min() and low.max() <= 1.0
