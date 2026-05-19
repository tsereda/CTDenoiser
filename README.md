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

No `pip install -e .` needed — run everything with `python -m ctdenoiser.*`
from the repo root and Python's `-m` flag puts `.` on `sys.path` automatically.

## Quick start

```bash
# Train (synthetic data if --data-root is omitted, for smoke testing)
python -m ctdenoiser.train --model ctformer --epochs 1

# Run tests
pytest -q
```

## Google Colab

The active branch is `claude/init-project-setup-vWHBI`. After cloning you
need to check it out explicitly — Colab clones `main` by default:

```python
!git fetch origin
!git checkout claude/init-project-setup-vWHBI
!git pull origin claude/init-project-setup-vWHBI
!pip install -r requirements.txt
!pytest -q                                         # sanity check
!python -m ctdenoiser.train --model ctformer --epochs 1
```

If you prefer an editable install (optional), use `--no-build-isolation` to
avoid pip's sandboxing conflicting with Colab's system setuptools:

```python
!pip install --no-build-isolation -e .
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
