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

    stride = patch_size - 2 * margin
    if stride <= 0:
        raise ValueError("margin too large: patch_size - 2*margin must be > 0")

    # Pad by margin so the first and last patch centers cover the image edges.
    padded = torch.nn.functional.pad(
        full_img, (margin, margin, margin, margin), mode="reflect"
    )
    pH, pW = padded.shape[2], padded.shape[3]
    out_img = torch.zeros_like(padded)
    weight_map = torch.zeros_like(padded)

    for y in _stops(pH, patch_size, stride):
        for x in _stops(pW, patch_size, stride):
            patch = padded[:, :, y : y + patch_size, x : x + patch_size]
            denoised = model(patch)
            center = denoised[:, :, margin:-margin, margin:-margin]
            ys, ye = y + margin, y + patch_size - margin
            xs, xe = x + margin, x + patch_size - margin
            out_img[:, :, ys:ye, xs:xe] += center
            weight_map[:, :, ys:ye, xs:xe] += 1.0

    result = out_img / torch.clamp(weight_map, min=1.0)
    return result[:, :, margin : margin + H, margin : margin + W]
