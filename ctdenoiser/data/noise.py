"""Synthetic-noise utilities for the non-CT / simulated datasets.

The denoising *theorem* the paper proves (one-step flow = regression, multi-step
Euler hurts) is not CT-specific, so these helpers manufacture the two noise
regimes that stress it outside real LDCT data:

* **i.i.d. Gaussian** -- the textbook white-noise case, and
* **spatially-correlated** -- Gaussian noise passed through a small Gaussian
  blur, mimicking the correlated grain that filtered-back-projection leaves in a
  reconstructed CT slice (and that the correlated-pairing experiments target).

Everything here is torch-only (no torchvision / scipy) so the datasets stay
importable in the minimal training image.
"""

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma, dtype=torch.float32, device=None):
    """A normalised 1-D Gaussian kernel with radius ``round(3*sigma)``."""
    radius = max(1, int(round(3.0 * float(sigma))))
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def gaussian_blur2d(x, sigma):
    """Separable reflect-padded Gaussian blur of an ``(N, C, H, W)`` tensor."""
    if sigma is None or sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma, x.dtype, x.device)
    n = k.numel()
    pad = n // 2
    channels = x.shape[1]
    kv = k.view(1, 1, n, 1).repeat(channels, 1, 1, 1)
    kh = k.view(1, 1, 1, n).repeat(channels, 1, 1, 1)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, kv, groups=channels)
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kh, groups=channels)
    return x


def sample_noise(h, w, std, mode="gaussian", correlation_sigma=1.5,
                 generator=None, dtype=torch.float32):
    """Return an ``(h, w)`` noise field with (approximately) the given std.

    ``mode='gaussian'`` is i.i.d. white noise; ``mode='correlated'`` blurs a
    white field and rescales it back to unit variance before applying ``std``,
    so the two regimes are matched in overall noise power and differ only in
    their spatial correlation.
    """
    field = torch.randn(1, 1, h, w, generator=generator, dtype=dtype)
    if mode == "correlated":
        field = gaussian_blur2d(field, correlation_sigma)
        # Blurring shrinks the variance; renormalise so the emitted level is the
        # requested std regardless of correlation_sigma (matched noise power).
        field = field / (field.std() + 1e-8)
    elif mode != "gaussian":
        raise ValueError(
            f"unknown noise mode {mode!r}; choose 'gaussian' or 'correlated'."
        )
    return field.view(h, w) * float(std)


def add_synthetic_noise(clean, std, mode="gaussian", correlation_sigma=1.5,
                        generator=None, clamp=True):
    """Add a fresh noise field to a clean ``(H, W)`` image tensor."""
    h, w = clean.shape[-2:]
    noisy = clean + sample_noise(
        h, w, std, mode=mode, correlation_sigma=correlation_sigma,
        generator=generator, dtype=clean.dtype,
    )
    return noisy.clamp(0.0, 1.0) if clamp else noisy


def simulate_low_dose(clean, base_std=0.03, signal_std=0.06,
                      correlation_sigma=1.0, generator=None, clamp=True):
    """Simulate a low-dose CT slice from a normalised full-dose slice.

    Reconstructed LDCT noise is (a) spatially correlated by the FBP ramp filter
    and (b) signal-dependent -- quantum noise grows with the attenuation the
    beam traverses, so denser (brighter, post-window) tissue is noisier. This is
    an image-domain approximation of that: a correlated unit field scaled by a
    per-pixel std ``base_std + signal_std * clean``. It needs no sinogram /
    Radon transform (hence no scipy/skimage) yet yields a second, independent
    paired low/full source whose noise is neither i.i.d. nor signal-flat.
    """
    h, w = clean.shape[-2:]
    unit = sample_noise(
        h, w, 1.0, mode="correlated", correlation_sigma=correlation_sigma,
        generator=generator, dtype=clean.dtype,
    )
    std_map = base_std + signal_std * clean
    noisy = clean + unit * std_map
    return noisy.clamp(0.0, 1.0) if clamp else noisy
