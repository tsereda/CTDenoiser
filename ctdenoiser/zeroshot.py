"""Zero-Shot Noise2Noise (ZS-N2N) test-time denoising.

For a single noisy image, a fresh tiny network is trained from scratch using a
pair of half-resolution images produced by a fixed checkerboard downsampler.
No external or clean data, no noise model, and no pretraining are required --
the network is discarded after denoising the image (hence "zero-shot").

The two downsampled views share signal but carry independent noise, so they act
as a Noise2Noise pair: the network learns to map one to the other (residual
term), regularised so that downsampling the denoised full-resolution image
agrees with the network's outputs on the pair (consistency term).

Reference: Mansour & Heckel, "Zero-Shot Noise2Noise: Efficient Image Denoising
without any Data", CVPR 2023.

Unlike Noise2Void's blind-spot scheme, the pair downsampler keeps both pixels of
each 2x2 block, which makes ZS-N2N more robust to the spatially correlated noise
found in CT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZSN2NNetwork(nn.Module):
    """Tiny 2-hidden-layer residual denoiser (CPU-friendly).

    ``Conv(1->C,3) -> ReLU -> Conv(C->C,3) -> ReLU -> Conv(C->1,1)`` with the
    output added to the input (residual learning). Default ``C=48`` follows the
    paper; tests use a much smaller width.
    """

    def __init__(self, num_channels=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, num_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, 1, 1),
        )

    def forward(self, x):
        return x + self.net(x)


def pair_downsampler(img):
    """Split ``(B, 1, H, W)`` into two half-resolution images.

    Two fixed 2x2 diagonal-averaging kernels are applied with stride 2, yielding
    ``(B, 1, H//2, W//2)`` views that share signal but carry independent noise.
    """
    k1 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], dtype=img.dtype, device=img.device)
    k2 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], dtype=img.dtype, device=img.device)
    img1 = F.conv2d(img, k1, stride=2)
    img2 = F.conv2d(img, k2, stride=2)
    return img1, img2


def zsn2n_loss(model, noisy):
    """Combined ZS-N2N residual + consistency loss for one image ``(B,1,H,W)``."""
    d1, d2 = pair_downsampler(noisy)

    # Residual term: each downsampled view denoises towards the other.
    pred1 = model(d1)
    pred2 = model(d2)
    loss_res = 0.5 * (F.mse_loss(pred1, d2) + F.mse_loss(pred2, d1))

    # Consistency term: downsampling the denoised image agrees with the
    # per-view denoised outputs.
    denoised = model(noisy)
    e1, e2 = pair_downsampler(denoised)
    loss_cons = 0.5 * (F.mse_loss(pred1, e1) + F.mse_loss(pred2, e2))

    return loss_res + loss_cons


def denoise_image(
    noisy,
    num_iters=2000,
    lr=1e-3,
    num_channels=48,
    device=None,
    seed=0,
):
    """Train a fresh :class:`ZSN2NNetwork` on the single image ``noisy`` and
    return the denoised result clamped to ``[0, 1]``.

    ``noisy`` is ``(1, 1, H, W)`` (or batch size 1). Odd spatial dimensions are
    cropped to even before downsampling. The network is self-contained: it has
    no external data and is not checkpointed.
    """
    device = device or noisy.device
    noisy = noisy.to(device)

    # pair_downsampler needs even H, W (stride-2 conv over 2x2 blocks).
    _, _, h, w = noisy.shape
    noisy = noisy[:, :, : h - (h % 2), : w - (w % 2)]

    # Reproducible weight init without disturbing the global RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ZSN2NNetwork(num_channels=num_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(num_iters):
        optimizer.zero_grad()
        loss = zsn2n_loss(model, noisy)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(noisy).clamp(0.0, 1.0)
    return out
