# CTDenoiser

Low-dose CT image denoising benchmark. Includes a **CTformer** transformer
architecture (Token2Token Dilation blocks with cyclic shifts) and a
**RED-CNN** convolutional baseline, plus training, evaluation, and
overlapped full-slice inference utilities.

## Layout

```
ctdenoiser/
  models/
    ctformer.py        # CTformer (T2TD / IT2TD + Transformer blocks)
    redcnn.py          # RED-CNN baseline
  data/
    dataset.py         # DICOMCTDataset + SyntheticCTDataset
    dicom.py           # DICOM series reader (HU conversion)
  selfsupervised.py    # Noise2Void blind-spot masking + loss
  zeroshot.py          # Zero-Shot Noise2Noise (per-image, data-free)
  metrics.py           # PSNR / SSIM / RMSE / GMSD / NPS-ratio
  inference.py         # overlapped full-slice inference
  train.py             # training loop / CLI
tests/
  test_models.py          # forward-pass shape sanity checks
  test_selfsupervised.py  # N2V + ZS-N2N checks
docs/
  future.md          # long-term research roadmap
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

## Self-supervised / zero-shot training

By default training is **supervised** (MSE / flow loss against the full-dose
reference). Two methods train *without* a clean target, using only the noisy
low-dose image — useful when paired LDCT/NDCT data are scarce. Both are still
evaluated against the clean `full` reference, so their PSNR / SSIM / RMSE / GMSD /
NPS-ratio are directly comparable to the supervised models.

```bash
# Noise2Void: blind-spot self-supervision (reuses any conv/transformer model)
python -m ctdenoiser.train --model redcnn --training-mode n2v \
    --dicom-root /data/ldct_dicom --epochs 50 --n2v-mask-fraction 0.02

# Zero-Shot Noise2Noise: a fresh tiny network is trained per image, no training data
python -m ctdenoiser.train --training-mode zsn2n \
    --dicom-root /data/ldct_dicom --zsn2n-iters 2000 --zsn2n-channels 48

# Smoke tests on synthetic data
python -m ctdenoiser.train --model unet --training-mode n2v --epochs 1
python -m ctdenoiser.train --training-mode zsn2n --synthetic-len 4 --zsn2n-iters 100
```

- `n2v` masks a fraction of pixels (`--n2v-mask-fraction`), replaces each with a
  random neighbour (`--n2v-neighbor-radius`), and trains the model to predict the
  original value at those blind spots. Produces a normal checkpoint.
  `flowmatching` is **not** compatible with `n2v` (it needs paired targets).
- `zsn2n` ignores `--model`: for each slice it trains a small 2-layer network from
  scratch on that image alone (`--zsn2n-iters`, `--zsn2n-channels`, `--zsn2n-lr`)
  and denoises it. No checkpoint is saved (per-image networks are discarded).

These are the self-supervised rows of the benchmark; see [`docs/future.md`](docs/future.md)
for how they fit the long-term research plan.

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
