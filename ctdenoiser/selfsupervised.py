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


# ----------------------------------------------------------------------------
# Noise2Sim: similarity-based self-supervision
#
# Reference: Niu, Gao, Yu, Wang, "Noise2Sim - Similarity-based Self-Learning for
# Image Denoising", arXiv 2011.03384 / ICML 2021 workshop.
#
# Instead of a blind spot, Noise2Sim exploits non-local self-similarity: for
# each pixel it searches the image for a *similar* pixel (matched on the patch
# around it) and uses that pixel's value as the regression target. Two pixels
# that share the same underlying signal but carry statistically independent
# noise form a Noise2Noise pair, so regressing the full noisy image onto this
# per-pixel "similar" image drives the network to the clean signal.
#
# Unlike N2V, the model sees the *un-masked* noisy image: it cannot cheat by
# copying the input because the target is a different pixel whose noise is
# independent, so the MSE optimum is the conditional mean (the signal). This
# avoids N2V's blind-spot information loss while still needing no clean target.
# ----------------------------------------------------------------------------

@torch.no_grad()
def make_similarity_target(noisy, search_radius=4, patch_radius=1, num_similar=1,
                           exclude_radius=1):
    """Build the per-pixel similarity target for Noise2Sim.

    For every pixel of ``noisy`` ``(B, 1, H, W)`` we scan all spatial offsets in
    a ``(2*search_radius+1)`` window (excluding a ``(2*exclude_radius-1)`` box
    around the pixel itself), score each candidate by the mean squared patch
    difference over a ``(2*patch_radius+1)`` neighbourhood, and replace the pixel
    by the average of its ``num_similar`` best-matching candidate values. Returns
    a tensor of the same shape as ``noisy``.

    ``exclude_radius`` is the correlated-noise decorrelation knob (matching
    :func:`ctdenoiser.models.ssflow.make_similarity_pairs`): candidate offsets
    with ``max(|dy|, |dx|) < exclude_radius`` are skipped so the matched pixel
    lies beyond the FBP noise-correlation length. ``exclude_radius=1`` excludes
    only the pixel itself (the original Noise2Sim behaviour); ``2``/``3`` push
    matches past the near-neighbour correlation. This lets a direct regression
    estimator use the *same* decorrelated pairs as the SSFlow velocity net, so
    the flow and the regression can be compared at identical pairing.

    Fully vectorised: borders use refl/replicate padding so every pixel has a
    full candidate set, and the patch distance uses an average pool with
    ``count_include_pad=False`` so edge patches are scored fairly.
    """
    b, c, h, w = noisy.shape
    r, pr = search_radius, patch_radius
    padded = F.pad(noisy, (r, r, r, r), mode="replicate")

    dists, vals = [], []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if max(abs(dy), abs(dx)) < exclude_radius:
                continue  # skip the excluded box so the target is decorrelated
            shifted = padded[:, :, r + dy : r + dy + h, r + dx : r + dx + w]
            sq = (noisy - shifted) ** 2
            dist = F.avg_pool2d(
                sq, 2 * pr + 1, stride=1, padding=pr, count_include_pad=False
            )
            dists.append(dist)
            vals.append(shifted)

    dist_stack = torch.stack(dists, dim=0)  # (O, B, C, H, W)
    val_stack = torch.stack(vals, dim=0)
    k = min(num_similar, dist_stack.shape[0])
    # k smallest patch distances per pixel -> gather their pixel values, average.
    idx = dist_stack.topk(k, dim=0, largest=False).indices
    chosen = torch.gather(val_stack, 0, idx)
    return chosen.mean(dim=0)


def n2sim_training_step(
    model,
    noisy,
    search_radius=4,
    patch_radius=1,
    num_similar=1,
    exclude_radius=1,
):
    """Compute the Noise2Sim similarity loss for a batch of noisy images.

    ``noisy`` is ``(B, 1, H, W)``. A per-pixel similarity target is built with
    :func:`make_similarity_target` (no gradient) and the model regresses the
    full noisy image onto it. Returns a differentiable scalar MSE.
    ``exclude_radius`` (default ``1``, the original behaviour) selects the
    decorrelation distance of the paired target.
    """
    target = make_similarity_target(
        noisy,
        search_radius=search_radius,
        patch_radius=patch_radius,
        num_similar=num_similar,
        exclude_radius=exclude_radius,
    )
    pred = model(noisy)
    return F.mse_loss(pred, target)
