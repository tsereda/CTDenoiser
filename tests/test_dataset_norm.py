import numpy as np
import pytest

from ctdenoiser.data.dataset import (
    HU_OFFSET,
    HU_SCALE,
    normalize_hu,
    pair_noise_stats,
)


def test_default_window_is_soft_tissue():
    # Soft-tissue / abdomen window: level 40 HU, width 400 HU -> [-160, +240].
    assert HU_OFFSET == 160.0
    assert HU_SCALE == 400.0


def test_normalize_hu_maps_window_to_unit_range():
    vol = np.array([-160.0, 40.0, 240.0], dtype=np.float32)
    out = normalize_hu(vol)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)


def test_normalize_hu_clips_outside_window():
    vol = np.array([-1000.0, 5000.0], dtype=np.float32)
    out = normalize_hu(vol)
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_pair_noise_stats_returns_mean_abs_diff():
    low = np.zeros((2, 4, 4), dtype=np.float32)
    full = np.full((2, 4, 4), 0.1, dtype=np.float32)
    diff = pair_noise_stats("p1", low, full)
    assert diff == pytest.approx(0.1, abs=1e-6)


def test_pair_noise_stats_warns_on_identical(capsys):
    vol = np.full((2, 4, 4), 0.3, dtype=np.float32)
    diff = pair_noise_stats("p1", vol, vol.copy())
    assert diff == pytest.approx(0.0, abs=1e-9)
    assert "nearly identical" in capsys.readouterr().out


def test_wide_window_washes_out_noise():
    # A realistic ~30 HU dose difference is ~7.5% of a 400 HU window but only
    # ~1.5% of a 2000 HU window -- the regression this change fixes.
    full_hu = np.zeros((1, 8, 8), dtype=np.float32)
    low_hu = full_hu + 30.0
    soft = pair_noise_stats("soft", normalize_hu(low_hu), normalize_hu(full_hu))
    wide = pair_noise_stats(
        "wide",
        normalize_hu(low_hu, 1000.0, 2000.0),
        normalize_hu(full_hu, 1000.0, 2000.0),
    )
    assert soft > wide
    assert soft == pytest.approx(30.0 / 400.0, abs=1e-6)
