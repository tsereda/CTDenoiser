#!/usr/bin/env python3
"""Qualitative denoising figure: one CT slice denoised by several methods.

Renders a two-row panel --- top: full slices with a red ROI box; bottom: the
magnified ROI --- for the LDCT input, each trained method, and the full-dose
reference.  This is where the paper's numbers become visible: the supervised
``waxy'' over-smoothing, Noise2Sim's preserved texture, and (by listing the same
SSFlow checkpoint at several ``:steps``) the multi-step drift back toward noise
that the finite-step departure predicts.

Runs on the cluster where the h5 cache and checkpoints live -- it reuses the
project's real inference path (``overlapped_inference`` + each model's forward),
so what you see is exactly what the metrics scored.

    python scripts/figure_qualitative.py \
        --h5 /workspace/data.h5 --patient L067 --slice 120 \
        --models "LDCT input=input" \
                 "Noise2Sim=redcnn:checkpoints/redcnn_n2sim.pt" \
                 "Supervised=redcnn:checkpoints/redcnn_sup.pt" \
                 "SSFlow (1 step)=ssflow:checkpoints/ssflow.pt:1" \
                 "SSFlow (20 step)=ssflow:checkpoints/ssflow.pt:20" \
                 "Full dose=clean" \
        --roi 190,210,90,90 --out figures/qualitative.pdf

Each --models entry is ``LABEL=SPEC`` where SPEC is one of:
    input                      the noisy LDCT slice (no model)
    clean                      the full-dose reference
    MODEL:CKPT[:STEPS]         build MODELS[MODEL], load CKPT; STEPS sets
                               num_steps for flow models (default 1).
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from ctdenoiser.inference import overlapped_inference
from ctdenoiser.train import MODELS

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "figure.dpi": 200,
})


def load_slice(h5_path: str, patient: str | None, sl: int):
    with h5py.File(h5_path, "r") as f:
        pats = sorted(f["patients"].keys())
        pid = patient if patient in pats else pats[0]
        if patient and patient not in pats:
            print(f"patient {patient!r} not found; using {pid}. available: {pats[:8]}...")
        low = f[f"patients/{pid}/low"][sl].astype("float32")
        full = f[f"patients/{pid}/full"][sl].astype("float32")
    return pid, low, full


def denoise(spec: str, low_t: torch.Tensor, full_t: torch.Tensor,
            patch: int, device: str) -> np.ndarray:
    """Return an (H, W) denoised image in [0, 1] for a --models SPEC."""
    if spec == "input":
        return low_t.squeeze().cpu().numpy()
    if spec == "clean":
        return full_t.squeeze().cpu().numpy()
    parts = spec.split(":")
    name, ckpt = parts[0], parts[1]
    steps = int(parts[2]) if len(parts) > 2 else 1
    if name not in MODELS:
        raise SystemExit(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    model = MODELS[name]().to(device)
    state = torch.load(ckpt, map_location=device)
    state = state.get("state_dict", state) if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state)
    model.eval()
    if hasattr(model, "num_steps"):
        model.num_steps = steps
    out = overlapped_inference(model, low_t, patch_size=patch, margin=patch // 4)
    return out.clamp(0, 1).squeeze().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="paired low/full HDF5 cache")
    ap.add_argument("--patient", default=None, help="patient id (default: first)")
    ap.add_argument("--slice", type=int, default=100, dest="sl")
    ap.add_argument("--models", nargs="+", required=True,
                    help="LABEL=SPEC entries; see the module docstring")
    ap.add_argument("--roi", default=None,
                    help="x,y,w,h magnified ROI in pixels (default: centre 80x80)")
    ap.add_argument("--vmin", type=float, default=0.0, help="display window low")
    ap.add_argument("--vmax", type=float, default=1.0, help="display window high")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--out", default="figures/qualitative.pdf")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    pid, low, full = load_slice(a.h5, a.patient, a.sl)
    H, W = low.shape
    low_t = torch.from_numpy(low)[None, None].to(a.device)
    full_t = torch.from_numpy(full)[None, None].to(a.device)

    if a.roi:
        x, y, w, h = (int(v) for v in a.roi.split(","))
    else:
        w = h = 80
        x, y = (W - w) // 2, (H - h) // 2

    entries = []
    for m in a.models:
        label, _, spec = m.partition("=")
        img = denoise(spec, low_t, full_t, a.patch, a.device)
        entries.append((label, img))
        print(f"  rendered: {label}")

    n = len(entries)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.4),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    if n == 1:
        axes = axes[:, None]
    dw = dict(cmap="gray", vmin=a.vmin, vmax=a.vmax)
    for j, (label, img) in enumerate(entries):
        ax = axes[0, j]
        ax.imshow(img, **dw)
        ax.add_patch(plt.Rectangle((x, y), w, h, ec="#E23", fc="none", lw=1.2))
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        axz = axes[1, j]
        axz.imshow(img[y:y + h, x:x + w], **dw)
        axz.set_xticks([]); axz.set_yticks([])
        for s in axz.spines.values():
            s.set_edgecolor("#E23"); s.set_linewidth(1.2)
    axes[0, 0].set_ylabel("full slice", fontsize=8)
    axes[1, 0].set_ylabel("ROI (2×)", fontsize=8)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {a.out}  (patient {pid}, slice {a.sl}, ROI {x},{y},{w},{h})")


if __name__ == "__main__":
    main()
