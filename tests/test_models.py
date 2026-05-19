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


def test_hdf5_dataset_split_and_shapes(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from ctdenoiser.data import HDF5CTDataset

    cache = tmp_path / "ldct_cache.h5"
    with h5py.File(cache, "w") as f:
        for pid in ["L001", "L002", "L003", "L004", "L005"]:
            for dose in ["low", "full"]:
                f.create_dataset(
                    f"{pid}_{dose}",
                    data=np.random.randn(4, 96, 110).astype("float32") * 500,
                )

    train_p, val_p = HDF5CTDataset.split_patients(cache, val_fraction=0.2, seed=0)
    assert set(train_p).isdisjoint(val_p)
    assert len(train_p) + len(val_p) == 5

    train_ds = HDF5CTDataset(cache, train_p, patch_size=64, train=True)
    low, full = train_ds[0]
    assert low.shape == (1, 64, 64) and full.shape == (1, 64, 64)
    assert 0.0 <= low.min() and low.max() <= 1.0

    val_ds = HDF5CTDataset(cache, val_p, patch_size=64, train=False)
    low, full = val_ds[0]
    assert low.shape == (1, 96, 110)  # full slice when train=False
