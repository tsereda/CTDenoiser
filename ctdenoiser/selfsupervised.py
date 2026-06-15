"""Noise2Void blind-spot self-supervised denoising.

Trains any single-image denoiser (RED-CNN, DnCNN, U-Net, CTFormer, ...) using
only noisy images and no clean target. A random subset of pixels is replaced by
a randomly chosen neighbour value, the masked image is passed through the model,
and the loss is computed *only* at the masked ("blind-spot") locations against
the original noisy value. Because the network never sees the true value of a
blind-spot pixel in its receptive field, it cannot learn the identity and is
forced to predict the underlying signal from the surrounding context.

Reference: Krull, Buchholz & Jug, "Noise2Void - Learning Denoising from Single
Noisy Images", CVPR 2019.

Caveat for CT: the standard blind-spot assumption is that noise is spatially
independent. Real CT noise is spatially correlated, so N2V is expected to be a
weaker self-supervised baseline than methods that model correlation (e.g.
ZS-N2N in :mod:`ctdenoiser.zeroshot`); the contrast is itself a useful
benchmark result.
"""

import torch
import torch.nn.functional as F


def make_blind_spot_mask(shape, mask_fraction=0.02, device=None, generator=None):
    """Return a boolean blind-spot mask of ``shape`` ``(B, 1, H, W)``.

    Each pixel is independently selected with probability ``mask_fraction``.
    At least one pixel per image is guaranteed ``True`` so the masked loss is
    always defined, even for small training patches.
    """
    b, c, h, w = shape
    mask = torch.rand(shape, device=device, generator=generator) < mask_fraction

    # Guarantee >= 1 blind spot per image so the masked loss is never empty.
    flat = mask.reshape(b, -1)
    empty = ~flat.any(dim=1)
    if empty.any():
        rand = torch.rand(b, c * h * w, device=device, generator=generator)
        pick = rand.argmax(dim=1)
        flat[empty, pick[empty]] = True
    return mask


def replace_with_neighbors(img, mask, radius=2, generator=None):
    """Return a copy of ``img`` with each masked pixel replaced by a random
    neighbour value within a ``(2*radius+1)`` window (self included is possible
    but unlikely for radius >= 1).

    Vectorised: random integer ``(dy, dx)`` offsets are sampled for every pixel,
    applied only at masked locations, and the source coordinates are clamped to
    the image bounds so borders and small patches are handled safely.
    """
    b, c, h, w = img.shape
    device = img.device

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    yy = yy.expand(b, c, h, w)
    xx = xx.expand(b, c, h, w)

    dy = torch.randint(-radius, radius + 1, img.shape, device=device, generator=generator)
    dx = torch.randint(-radius, radius + 1, img.shape, device=device, generator=generator)
    src_y = (yy + dy).clamp(0, h - 1)
    src_x = (xx + dx).clamp(0, w - 1)

    # Flatten H,W to gather per (b, c) plane.
    flat = img.reshape(b, c, h * w)
    src_idx = (src_y * w + src_x).reshape(b, c, h * w)
    neighbors = torch.gather(flat, 2, src_idx).reshape(b, c, h, w)

    return torch.where(mask, neighbors, img)


def n2v_training_step(
    model,
    noisy,
    mask_fraction=0.02,
    neighbor_radius=2,
    generator=None,
):
    """Compute the Noise2Void blind-spot loss for a batch of noisy images.

    ``noisy`` is ``(B, 1, H, W)``. Returns a differentiable scalar MSE computed
    only at the blind-spot locations between the model prediction and the
    original noisy value.
    """
    mask = make_blind_spot_mask(
        noisy.shape, mask_fraction, device=noisy.device, generator=generator
    )
    masked_input = replace_with_neighbors(
        noisy, mask, radius=neighbor_radius, generator=generator
    )
    pred = model(masked_input)
    return F.mse_loss(pred[mask], noisy[mask])
