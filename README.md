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
    dataset.py      # paired low-dose / full-dose patch dataset
  metrics.py        # PSNR / SSIM / RMSE
  inference.py      # overlapped full-slice inference
  train.py          # training loop / CLI
tests/
  test_models.py    # forward-pass shape sanity checks
```

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Train (synthetic data if --data-root is omitted, for smoke testing)
python -m ctdenoiser.train --model ctformer --epochs 1

# Run tests
pytest -q
```

## Data

`PairedCTDataset` expects two directories of `.npy` slices with matching
filenames:

```
data/
  low_dose/   slice_0001.npy ...
  full_dose/  slice_0001.npy ...
```

Pass `--data-root data/` to `train.py`. With no data root a synthetic
noisy/clean dataset is generated so the pipeline can be exercised end to end.
