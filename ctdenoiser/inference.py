"""Overlapped full-slice inference.

Transformer denoisers process locally structured token sets, so naively
stitching raw patches produces grid/boundary artifacts. Each patch is
denoised, its margin ``eta`` discarded, and the centers are blended on a
weight map.
"""

import torch


def _stops(extent, patch_size, stride):
    """Patch start positions covering ``extent``, incl. a final flush stop."""
    stops = list(range(0, extent - patch_size + 1, stride))
    if not stops or stops[-1] != extent - patch_size:
        stops.append(extent - patch_size)
    return stops


@torch.no_grad()
def overlapped_inference(model, full_img, patch_size=64, margin=16):
    """Evaluate a full CT slice ``(B, C, H, W)`` with overlapped inference."""
    model.eval()
    B, C, H, W = full_img.shape
    out_img = torch.zeros_like(full_img)
    weight_map = torch.zeros_like(full_img)

    stride = patch_size - 2 * margin
    if stride <= 0:
        raise ValueError("margin too large: patch_size - 2*margin must be > 0")

    for y in _stops(H, patch_size, stride):
        for x in _stops(W, patch_size, stride):
            patch = full_img[:, :, y : y + patch_size, x : x + patch_size]
            denoised = model(patch)
            center = denoised[:, :, margin:-margin, margin:-margin]
            ys, ye = y + margin, y + patch_size - margin
            xs, xe = x + margin, x + patch_size - margin
            out_img[:, :, ys:ye, xs:xe] += center
            weight_map[:, :, ys:ye, xs:xe] += 1.0

    return out_img / torch.clamp(weight_map, min=1.0)
