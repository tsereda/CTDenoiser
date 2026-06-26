"""Tests for the hallucination-aware detectability evaluation (Phase 1).

The headline test is the analytic-Gaussian CHO sanity check: for a known signal
in white Gaussian noise the channelized Hotelling observer's detectability index
must match the closed-form ``d' = ||signal|| / sigma``. Without this, reviewers
(rightly) won't trust the CHO numbers.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from ctdenoiser.detectability import (
    cho_detectability,
    extract_rois,
    insert_signal,
    laguerre_gauss_channels,
    sample_flat_locations,
    signal_template,
)
from ctdenoiser.metrics import nps_ratio, residual_spectrum, uniform_nps


# ----- Signal insertion -----

def test_insert_signal_adds_exactly_the_template():
    clean = torch.rand(1, 1, 64, 64)
    s = signal_template(clean.shape[-2:], (32, 32), 4, contrast_hu=10, hu_scale=400)
    present = insert_signal(clean, (32, 32), 4, contrast_hu=10, hu_scale=400)
    # low_present - low == s, the identity the whole methodology rests on.
    assert torch.allclose(present - clean, s, atol=1e-6)


def test_disk_signal_amplitude_matches_hu_over_scale():
    s = signal_template((32, 32), (16, 16), 3, contrast_hu=20, hu_scale=400, profile="disk")
    # Centre pixel is inside the disk -> exactly contrast_hu / hu_scale.
    assert s[16, 16].item() == pytest.approx(20 / 400)
    # A far corner is outside the disk -> zero.
    assert s[0, 0].item() == 0.0


def test_gaussian_signal_peaks_at_center():
    s = signal_template((32, 32), (16, 16), 3, contrast_hu=20, hu_scale=400, profile="gaussian")
    assert s[16, 16].item() == pytest.approx(20 / 400, rel=1e-5)
    assert s[16, 16] > s[16, 20] > s[16, 25]  # monotone falloff


# ----- ROI selection -----

def test_sample_flat_locations_avoids_edges():
    # Flat-left / textured-right image: flat ROIs must land on the left half.
    img = torch.zeros(128, 128)
    img[:, 64:] = 0.5 + 0.3 * torch.rand(128, 64)  # noisy/textured right
    img[:, :64] = 0.5                                # perfectly flat left
    centers = sample_flat_locations(img, n_locations=5, roi_size=16, var_quantile=0.3)
    assert len(centers) > 0
    assert all(cx < 64 for _, cx in centers)


def test_extract_rois_shape_and_bounds():
    img = torch.rand(64, 64)
    rois = extract_rois(img, [(32, 32), (10, 10)], roi_size=16)
    assert rois.shape == (2, 16, 16)
    # An out-of-bounds centre is dropped, not clamped.
    rois = extract_rois(img, [(1, 1)], roi_size=16)
    assert rois.shape[0] == 0


# ----- Laguerre-Gauss channels -----

def test_lg_channels_orthonormal():
    u = laguerre_gauss_channels(24, n_channels=10)
    gram = u.T @ u
    assert torch.allclose(gram, torch.eye(10), atol=1e-5)


# ----- Channelized Hotelling Observer -----

def test_cho_matches_analytic_dprime_in_white_noise():
    """SKE/BKS d' must approach the closed-form ||signal|| / sigma."""
    torch.manual_seed(0)
    roi_size, sigma, n = 24, 1.0, 5000

    signal = signal_template(
        (roi_size, roi_size), (roi_size // 2, roi_size // 2),
        radius_px=3, contrast_hu=0.6, hu_scale=1.0, profile="gaussian",
    )
    u = laguerre_gauss_channels(roi_size, n_channels=12, dtype=torch.float64)

    # In-span ideal: the channels must actually capture the (radial) signal.
    s_vec = signal.reshape(-1).double()
    proj_norm = torch.linalg.vector_norm(u.double().T @ s_vec).item()
    assert proj_norm > 0.9 * torch.linalg.vector_norm(s_vec).item()
    ideal_dprime = proj_norm / sigma

    noise_p = sigma * torch.randn(n, roi_size, roi_size)
    noise_a = sigma * torch.randn(n, roi_size, roi_size)
    present = noise_p + signal[None]
    absent = noise_a

    out = cho_detectability(present, absent, signal=signal, channels=u)
    assert out["d_prime"] == pytest.approx(ideal_dprime, rel=0.15)
    assert 0.5 < out["auc"] < 1.0


def test_cho_null_case_is_chance():
    """Identical present/absent ensembles -> no detectability (d' ~ 0, AUC ~ 0.5)."""
    torch.manual_seed(1)
    rois = torch.randn(2000, 16, 16)
    out = cho_detectability(rois.clone(), rois.clone() + torch.randn(2000, 16, 16) * 0,
                            n_channels=8)
    # Same data on both sides -> empirical mean difference is ~0.
    assert out["d_prime"] < 0.5
    assert out["auc"] == pytest.approx(0.5, abs=0.05)


def test_cho_detects_stronger_signal_better():
    torch.manual_seed(2)
    roi_size, n = 20, 3000
    weak = signal_template((roi_size, roi_size), (10, 10), 3, 2.0, 1.0, "gaussian")
    strong = signal_template((roi_size, roi_size), (10, 10), 3, 6.0, 1.0, "gaussian")
    noise = lambda: torch.randn(n, roi_size, roi_size)
    d_weak = cho_detectability(noise() + weak[None], noise(), signal=weak)["d_prime"]
    d_strong = cho_detectability(noise() + strong[None], noise(), signal=strong)["d_prime"]
    assert d_strong > d_weak


# ----- Corrected NPS -----

def test_residual_spectrum_alias_matches_legacy_name():
    a, b = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)
    assert residual_spectrum(a, b) == nps_ratio(a, b)


def test_uniform_nps_peak_shifts_low_for_blotchy_noise():
    """Low-pass (blotchy) noise must shift the NPS peak to lower frequency."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    white = torch.randn(128, 128)
    # Blur to make spatially correlated ("waxy") noise.
    k = torch.ones(1, 1, 5, 5) / 25.0
    blotchy = F.conv2d(white[None, None], k, padding=2)[0, 0]

    centers = [(32, 32), (32, 96), (96, 32), (96, 96), (64, 64)]
    nps_white = uniform_nps(white, centers, roi_size=32)
    nps_blotchy = uniform_nps(blotchy, centers, roi_size=32)

    assert nps_white["radial_nps"].ndim == 1
    # Blotchy noise concentrates power at low frequency (lower spectral centroid)
    # and carries less total power. White noise has a ~flat spectrum, so the
    # centroid — not the (tie-prone) peak bin — is the robust discriminator.
    assert nps_blotchy["mean_freq"] < nps_white["mean_freq"]
    assert nps_blotchy["total_power"] < nps_white["total_power"]
