# CTDenoiser

Low-dose CT image denoising benchmark. Includes a **CTformer** transformer
architecture (Token2Token Dilation blocks with cyclic shifts) and a
**RED-CNN** convolutional baseline, plus training, evaluation, and
overlapped full-slice inference utilities.

## Layout

```
ctdenoiser/
  models/
    ctformer.py     # CTformer (T2TD / IT2TD + Transformer blocks)
    redcnn.py       # RED-CNN baseline
  data/
    dataset.py      # DICOMCTDataset + SyntheticCTDataset
    dicom.py        # DICOM series reader (HU conversion)
  metrics.py        # PSNR / SSIM / RMSE / GMSD / NPS-ratio
  inference.py      # overlapped full-slice inference
  train.py          # training loop / CLI
tests/
  test_models.py    # forward-pass shape sanity checks
```

## Install

```bash
pip install -e .
```

## Quick start

```bash
# Smoke test (synthetic data)
python -m ctdenoiser.train --model ctformer --epochs 1

# Train on DICOM data (patient-split, full-slice overlapped-inference eval)
python -m ctdenoiser.train --model ctformer \
    --dicom-root /data/ldct_dicom --epochs 50 --batch-size 16

# Run tests
pytest -q
```

## Data

### DICOM series directories

Point `--dicom-root` at a directory of TCIA LDCT SeriesInstanceUID
subdirectories, each containing `.dcm` files. Low/full dose is detected
automatically from the DICOM `SeriesDescription` header and paired by
`PatientID`.

```
ldct_dicom/
  1.2.840.113713.4.100.1.2.xxx/   # Low Dose Images
    1-001.dcm  1-002.dcm  ...
  1.2.840.113713.4.100.1.2.yyy/   # Full Dose Images
    1-001.dcm  1-002.dcm  ...
  ...
```

- Split is **by patient** (`--val-fraction`, `--seed`) to prevent leakage.
- Training uses random `--patch-size` crops.
- Validation runs **full-slice overlapped inference** (margin = patch/4)
  and reports PSNR / SSIM / RMSE / GMSD / NPS-ratio.
- HU → `[0,1]` via `clamp((hu + 1000) / 2000, 0, 1)`.

### Synthetic (smoke test)

Without `--dicom-root`, a synthetic noisy/clean dataset is generated so the
pipeline runs end to end.
