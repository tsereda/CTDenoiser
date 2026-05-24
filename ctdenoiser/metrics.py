"""Image-quality metrics for denoising evaluation."""

import torch
import torch.nn.functional as F

# Prewitt kernels for GMSD — registered once at module level
_PREWITT_X = torch.tensor(
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32
).view(1, 1, 3, 3)
_PREWITT_Y = _PREWITT_X.transpose(-1, -2).contiguous()


def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target)).item()


def psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return float("inf")
    return (10.0 * torch.log10(data_range**2 / mse)).item()


def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)


def ssim(pred, target, data_range=1.0, window_size=11, sigma=1.5):
    """Mean SSIM over a batch of ``(B, 1, H, W)`` tensors."""
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad)
    mu_t = F.conv2d(target, window, padding=pad)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t

    sigma_p2 = F.conv2d(pred**2, window, padding=pad) - mu_p2
    sigma_t2 = F.conv2d(target**2, window, padding=pad) - mu_t2
    sigma_pt = F.conv2d(pred * target, window, padding=pad) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
    )
    return ssim_map.mean().item()


def gmsd(pred, target, c=1e-6):
    """Gradient Magnitude Similarity Deviation — perceptual proxy.

    Lower is better. No pretrained weights required.
    """
    px = _PREWITT_X.to(pred.device, pred.dtype)
    py = _PREWITT_Y.to(pred.device, pred.dtype)

    def _gm(x):
        gx = F.conv2d(x, px, padding=1)
        gy = F.conv2d(x, py, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + c)

    gm_p = _gm(pred)
    gm_t = _gm(target)
    gms = (2 * gm_p * gm_t + c) / (gm_p ** 2 + gm_t ** 2 + c)
    return gms.std().item()


def nps_ratio(pred, target, eps=1e-8):
    """Mean noise power spectrum ratio via 2-D FFT.

    Captures spatial-frequency distribution of residual error.
    Lower is better; values near 0 indicate negligible residual energy.
    """
    residual = pred - target
    nps_res = torch.fft.fft2(residual).abs() ** 2
    nps_tgt = torch.fft.fft2(target).abs() ** 2
    return (nps_res.mean() / (nps_tgt.mean() + eps)).item()
