"""Tests for the natural-image and simulated-LDCT datasets and noise utils."""

import argparse

import numpy as np
import pytest
import torch

from ctdenoiser.data.dataset import (
    NaturalImageDenoisingDataset,
    SimulatedLowDoseCTDataset,
)
from ctdenoiser.data.noise import (
    gaussian_blur2d,
    sample_noise,
    simulate_low_dose,
)


# --------------------------------------------------------------------------- #
# noise utilities
# --------------------------------------------------------------------------- #
def test_sample_noise_matches_requested_std_both_regimes():
    g = torch.Generator().manual_seed(0)
    for mode in ("gaussian", "correlated"):
        n = sample_noise(128, 128, std=0.2, mode=mode,
                         correlation_sigma=2.0, generator=g)
        assert n.shape == (128, 128)
        # Both regimes are renormalised to the same overall noise power.
        assert n.std().item() == pytest.approx(0.2, rel=0.1)


def test_gaussian_blur_reduces_high_frequency_variance():
    g = torch.Generator().manual_seed(1)
    white = torch.randn(1, 1, 64, 64, generator=g)
    blurred = gaussian_blur2d(white, sigma=2.0)
    # Blurring a white field averages neighbours -> lower per-pixel variance.
    assert blurred.var().item() < white.var().item()


def test_correlated_noise_is_spatially_smoother_than_gaussian():
    g = torch.Generator().manual_seed(2)
    iid = sample_noise(96, 96, 0.1, "gaussian", generator=g)
    g = torch.Generator().manual_seed(2)
    corr = sample_noise(96, 96, 0.1, "correlated", correlation_sigma=2.0, generator=g)

    def lag1_corr(x):
        a = x[:, :-1].flatten()
        b = x[:, 1:].flatten()
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

    assert lag1_corr(corr) > lag1_corr(iid) + 0.2


def test_simulate_low_dose_is_signal_dependent():
    # A brighter (denser) region should pick up more noise than a dark one.
    clean = torch.zeros(64, 64)
    clean[:, 32:] = 0.9
    diffs = []
    for s in range(8):
        g = torch.Generator().manual_seed(s)
        low = simulate_low_dose(clean, generator=g)
        dark = (low[:, :32] - clean[:, :32]).std().item()
        bright = (low[:, 32:] - clean[:, 32:]).std().item()
        diffs.append(bright - dark)
    assert np.mean(diffs) > 0


# --------------------------------------------------------------------------- #
# NaturalImageDenoisingDataset
# --------------------------------------------------------------------------- #
def test_natural_procedural_clean_pair_shapes_and_range():
    items = NaturalImageDenoisingDataset.list_items(length=6)
    ds = NaturalImageDenoisingDataset(items, patch_size=32, train=True,
                                      samples_per_image=3, noise_std=0.1)
    assert len(ds) == 6 * 3
    low, full = ds[0]
    assert low.shape == (1, 32, 32)
    assert full.shape == (1, 32, 32)
    assert 0.0 <= low.min() and low.max() <= 1.0
    # low is a noised view of the clean target -> not identical.
    assert not torch.allclose(low, full)


def test_natural_noisy_pair_gives_two_distinct_noisy_views():
    items = list(range(4))
    ds = NaturalImageDenoisingDataset(items, train=False, pair_mode="noisy",
                                      noise_std=0.15)
    low, full = ds[0]
    # Both arms are noisy (neither equals the underlying clean image) and they
    # are independent draws, so they differ from each other.
    assert not torch.allclose(low, full)
    assert low.std() > 0 and full.std() > 0


def test_natural_eval_is_deterministic():
    items = list(range(3))
    kw = dict(train=False, noise_std=0.1, seed=7)
    a = NaturalImageDenoisingDataset(items, **kw)[1]
    b = NaturalImageDenoisingDataset(items, **kw)[1]
    assert torch.allclose(a[0], b[0])
    assert torch.allclose(a[1], b[1])


def test_natural_split_is_disjoint():
    train_items, val_items = NaturalImageDenoisingDataset.split_items(
        length=10, val_fraction=0.2, seed=0
    )
    assert set(train_items).isdisjoint(val_items)
    assert len(train_items) + len(val_items) == 10


def test_natural_rejects_bad_modes():
    with pytest.raises(ValueError, match="noise_mode"):
        NaturalImageDenoisingDataset([0], noise_mode="salt")
    with pytest.raises(ValueError, match="pair_mode"):
        NaturalImageDenoisingDataset([0], pair_mode="triplet")


# --------------------------------------------------------------------------- #
# SimulatedLowDoseCTDataset
# --------------------------------------------------------------------------- #
def test_sim_ldct_procedural_pairs():
    patients = SimulatedLowDoseCTDataset.list_patients(n_patients=4)
    assert len(patients) == 4
    ds = SimulatedLowDoseCTDataset(patients, train=True, patch_size=32,
                                   n_slices=5, image_size=64)
    assert len(ds) == 4 * 5
    low, full = ds[0]
    assert low.shape == (1, 32, 32)
    # full is the clean CT slice; low is the simulated low-dose arm.
    assert not torch.allclose(low, full)
    assert 0.0 <= low.min() and low.max() <= 1.0


def test_sim_ldct_eval_deterministic_and_split():
    train_p, val_p = SimulatedLowDoseCTDataset.split_patients(
        n_patients=6, val_fraction=0.34, seed=1
    )
    assert set(train_p).isdisjoint(val_p)
    ds1 = SimulatedLowDoseCTDataset(val_p, train=False, n_slices=3, image_size=48)
    ds2 = SimulatedLowDoseCTDataset(val_p, train=False, n_slices=3, image_size=48)
    a, b = ds1[2], ds2[2]
    assert torch.allclose(a[0], b[0])
    assert torch.allclose(a[1], b[1])


# --------------------------------------------------------------------------- #
# build_loaders wiring
# --------------------------------------------------------------------------- #
def _loader_args(**overrides):
    base = dict(
        h5_path=None, dicom_root=None, mmap_cache=None,
        natural=False, natural_root=None, natural_procedural=False,
        natural_len=8, samples_per_image=2,
        noise_std=0.1, noise_mode="gaussian", correlation_sigma=1.5,
        pair_mode="clean",
        sim_ldct=False, sim_source=None, sim_procedural=False, sim_patients=4,
        sim_base_std=0.03, sim_signal_std=0.06, sim_correlation_sigma=1.0,
        patch_size=32, val_fraction=0.25, seed=0,
        batch_size=4, num_workers=0, prefetch_factor=4, synthetic_len=8,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_loaders_natural():
    from ctdenoiser.train import build_loaders

    train_loader, val_loader, full_slice = build_loaders(_loader_args(natural=True))
    assert full_slice is True
    low, full = next(iter(train_loader))
    assert low.shape[-2:] == (32, 32)
    assert len(val_loader.dataset) >= 1


def test_build_loaders_sim_ldct():
    from ctdenoiser.train import build_loaders

    train_loader, val_loader, full_slice = build_loaders(_loader_args(sim_ldct=True))
    assert full_slice is True
    low, full = next(iter(train_loader))
    assert low.shape[0] == 4  # batch_size


# --------------------------------------------------------------------------- #
# CLI: explicit data source required (no silent synthetic fallback)
# --------------------------------------------------------------------------- #
def test_natural_requires_explicit_source():
    from ctdenoiser.train import main

    # --natural with neither --natural-root nor --natural-procedural: error out
    # rather than silently training on synthetic images.
    with pytest.raises(SystemExit):
        main(["--natural", "--device", "cpu"])


def test_natural_rejects_both_sources(tmp_path):
    from ctdenoiser.train import main

    with pytest.raises(SystemExit):
        main(["--natural", "--natural-root", str(tmp_path),
              "--natural-procedural", "--device", "cpu"])


def test_sim_ldct_requires_explicit_source():
    from ctdenoiser.train import main

    with pytest.raises(SystemExit):
        main(["--sim-ldct", "--device", "cpu"])
