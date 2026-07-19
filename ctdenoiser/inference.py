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
def overlapped_inference(model, full_img, patch_size=64, margin=16,
                         max_patch_batch=256):
    """Evaluate a full CT slice ``(B, C, H, W)`` with overlapped inference.

    All patches of a slice are collected and pushed through the model in
    batches of up to ``max_patch_batch`` rather than one launch per patch: a
    512x512 slice at patch 64 / margin 16 has 225 patches, and 225 sequential
    batch-1 forwards leave the GPU idle almost the whole eval (this runs per
    val slice, per epoch -- it dominated wall time and dragged sweep-wide GPU
    utilisation under the cluster's kill threshold). Per-patch results are
    identical to the one-at-a-time loop; only the batching changes.
    """
    model.eval()
    B, C, H, W = full_img.shape
    out_img = torch.zeros_like(full_img)
    weight_map = torch.zeros_like(full_img)

    stride = patch_size - 2 * margin
    if stride <= 0:
        raise ValueError("margin too large: patch_size - 2*margin must be > 0")

    coords = [
        (y, x)
        for y in _stops(H, patch_size, stride)
        for x in _stops(W, patch_size, stride)
    ]

    # (B, P, C, ps, ps) -> (B*P, C, ps, ps): every patch of every slice in one
    # tensor, then chunked forwards bound activation memory.
    patches = torch.stack(
        [full_img[:, :, y : y + patch_size, x : x + patch_size] for y, x in coords],
        dim=1,
    ).reshape(B * len(coords), C, patch_size, patch_size)
    denoised_chunks = [
        model(patches[i : i + max_patch_batch])
        for i in range(0, patches.shape[0], max_patch_batch)
    ]
    denoised_all = torch.cat(denoised_chunks, dim=0).reshape(
        B, len(coords), C, patch_size, patch_size
    )

    for j, (y, x) in enumerate(coords):
        denoised = denoised_all[:, j]
        # Discard the margin only on sides that abut another patch; keep it
        # against the image edge so the outer border is not dropped to zero.
        top = margin if y > 0 else 0
        left = margin if x > 0 else 0
        bottom = margin if y + patch_size < H else 0
        right = margin if x + patch_size < W else 0
        center = denoised[
            :, :,
            top : patch_size - bottom,
            left : patch_size - right,
        ]
        ys, ye = y + top, y + patch_size - bottom
        xs, xe = x + left, x + patch_size - right
        out_img[:, :, ys:ye, xs:xe] += center
        weight_map[:, :, ys:ye, xs:xe] += 1.0

    return out_img / torch.clamp(weight_map, min=1.0)
