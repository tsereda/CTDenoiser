"""Per-anatomy windows, hardened pairing, and benchmark-report helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ctdenoiser.data.dataset import (
    ANATOMY_WINDOWS,
    HU_OFFSET,
    HU_SCALE,
    normalize_hu,
    window_for_anatomy,
)


# ── anatomy windows ───────────────────────────────────────────────────────────

def test_abdomen_preset_matches_default_window():
    assert window_for_anatomy("abdomen") == (HU_OFFSET, HU_SCALE)


def test_each_anatomy_has_a_distinct_window():
    windows = [window_for_anatomy(a) for a in ANATOMY_WINDOWS]
    assert len(set(windows)) == len(windows)


def test_chest_window_keeps_lung_tissue_in_range():
    # Lung parenchyma (~-800 HU) clips to 0 under the abdomen window but is
    # resolved under the chest/lung window -- the whole point of the preset.
    lung = np.array([-800.0], dtype=np.float32)
    off, scale = window_for_anatomy("chest")
    assert normalize_hu(lung, off, scale)[0] > 0.0
    assert normalize_hu(lung)[0] == 0.0  # default abdomen window clips it away


def test_unknown_anatomy_raises_with_hint():
    with pytest.raises(ValueError, match="unknown anatomy"):
        window_for_anatomy("thorax")


# ── hardened pairing ──────────────────────────────────────────────────────────

def _write_series(d: Path, pid: str, desc: str):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    d.mkdir(parents=True, exist_ok=True)
    ds = Dataset()
    ds.PatientID = pid
    ds.SeriesDescription = desc
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1000.0
    ds.Rows = ds.Columns = 4
    ds.BitsAllocated = 16
    ds.PixelData = np.zeros((4, 4), dtype=np.uint16).tobytes()
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = generate_uid()
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = fm
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(d / "1-001.dcm"), write_like_original=False)


def test_scan_pairs_low_and_full(tmp_path):
    pytest.importorskip("pydicom")
    from ctdenoiser.data.dicom import scan_paired_series

    _write_series(tmp_path / "s_low", "P1", "Low Dose Images")
    _write_series(tmp_path / "s_full", "P1", "Full Dose Images")
    mapping = scan_paired_series(str(tmp_path))
    assert set(mapping["P1"]) == {"low", "full"}


def test_scan_raises_on_duplicate_dose(tmp_path):
    pytest.importorskip("pydicom")
    from ctdenoiser.data.dicom import scan_paired_series

    _write_series(tmp_path / "s_low1", "P1", "Low Dose Images")
    _write_series(tmp_path / "s_low2", "P1", "Low Dose Images")
    _write_series(tmp_path / "s_full", "P1", "Full Dose Images")
    with pytest.raises(ValueError, match="two 'low dose' series"):
        scan_paired_series(str(tmp_path))


# ── benchmark report helpers ──────────────────────────────────────────────────

def _load_report_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_report.py"
    spec = importlib.util.spec_from_file_location("benchmark_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pareto_mask_picks_non_dominated():
    report = _load_report_module()
    # cost (lower better), quality (higher better)
    cost = [1.0, 2.0, 3.0, 2.0]
    qual = [10.0, 12.0, 11.0, 9.0]
    mask = report.pareto_mask(cost, qual)
    # point 0 (cheap), point 1 (best quality) optimal; 2 dominated by 1; 3 dominated by 1
    assert list(mask) == [True, True, False, False]


def test_pareto_mask_ignores_nan():
    report = _load_report_module()
    mask = report.pareto_mask([1.0, np.nan], [5.0, 9.0])
    assert list(mask) == [True, False]


def test_build_html_round_trips(tmp_path):
    report = _load_report_module()
    import pandas as pd

    df = pd.DataFrame([
        dict(State="finished", model="redcnn", **{"training-mode": "supervised"},
             anatomy="abdomen", param_count=1.8e6, model_size_mb=7.2,
             **{"val/latency_ms": 22.0, "val/psnr": 31.4, "val/psnr_std": 0.8,
                "val/ssim": 0.91, "val/rmse": 0.03, "val/gmsd": 0.02,
                "val/nps_ratio": 0.5, "baseline/psnr": 27.0,
                "baseline/psnr_std": 0.6}),
        dict(State="finished", model="ctformer", **{"training-mode": "supervised"},
             anatomy="abdomen", param_count=1.4e6, model_size_mb=5.6,
             **{"val/latency_ms": 48.0, "val/psnr": 32.1, "val/psnr_std": 0.7,
                "val/ssim": 0.92, "val/rmse": 0.028, "val/gmsd": 0.018,
                "val/nps_ratio": 0.48, "baseline/psnr": 27.0,
                "baseline/psnr_std": 0.5}),
    ])
    html = report.build_html(report.tidy(df), images_dir=None)
    assert "CTDenoiser benchmark" in html
    assert "ctformer" in html and "redcnn" in html
    assert "data:image/png;base64" in html  # at least one embedded plot
    assert "+4.40" in html or "+5.10" in html  # ΔPSNR over baseline rendered
