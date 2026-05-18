import pytest

torch = pytest.importorskip("torch")

from ctdenoiser.inference import overlapped_inference
from ctdenoiser.metrics import psnr, rmse, ssim
from ctdenoiser.models import CTformer, REDCNN


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


def test_metrics_identity():
    a = torch.rand(1, 1, 32, 32)
    assert rmse(a, a) == pytest.approx(0.0, abs=1e-6)
    assert psnr(a, a) == float("inf")
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-4)
