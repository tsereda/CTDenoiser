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


def residual_spectrum(pred, target, eps=1e-8):
    """Mean spectral energy of the residual relative to the reference.

    Captures the spatial-frequency distribution of the *residual error*
    ``pred - target`` via the 2-D FFT, normalised by the reference's spectral
    energy. Lower is better; values near 0 indicate negligible residual energy.

    Note: despite the historical name ``nps_ratio`` (kept as an alias for
    backward-compatible logging), this is **not** a noise power spectrum — it is
    a reference-relative residual spectrum, because it needs the clean target.
    The genuine, reference-free noise power spectrum of an image (and its shape /
    peak-frequency shift, the "waxy texture" radiologists distrust) is measured
    by :func:`uniform_nps`.
    """
    residual = pred - target
    nps_res = torch.fft.fft2(residual).abs() ** 2
    nps_tgt = torch.fft.fft2(target).abs() ** 2
    return (nps_res.mean() / (nps_tgt.mean() + eps)).item()


# Backward-compatible alias. Existing runs / CSVs / reports log this as
# ``nps_ratio``; keep the name working but treat :func:`residual_spectrum` as
# the honest one going forward (see docs/plan.md).
nps_ratio = residual_spectrum


def _radial_profile(power, n_bins=None):
    """Azimuthally average a 2-D power map ``(H, W)`` into a radial profile.

    The DC term sits at index 0 (assumes an ``fftshift``-ed, or here a
    centred-frequency, map). Returns a 1-D tensor of length ``n_bins`` indexed by
    increasing spatial frequency.
    """
    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(
        torch.arange(h, device=power.device, dtype=torch.float32) - cy,
        torch.arange(w, device=power.device, dtype=torch.float32) - cx,
        indexing="ij",
    )
    r = torch.sqrt(yy ** 2 + xx ** 2)
    n_bins = n_bins or int(r.max().item()) + 1
    r_idx = r.round().long().clamp(max=n_bins - 1).reshape(-1)
    flat = power.reshape(-1)
    sums = torch.zeros(n_bins, device=power.device, dtype=power.dtype)
    counts = torch.zeros(n_bins, device=power.device, dtype=power.dtype)
    sums.scatter_add_(0, r_idx, flat)
    counts.scatter_add_(0, r_idx, torch.ones_like(flat))
    return sums / counts.clamp_min(1.0)


def uniform_nps(image, rois, roi_size=64):
    """Reference-free 2-D noise power spectrum, ensemble-averaged over flat ROIs.

    Unlike :func:`residual_spectrum` this needs *no clean target*: it measures
    the texture of the noise in the image itself, which is what reveals a
    denoiser shifting power toward low frequencies (the blotchy / "waxy" look
    that PSNR and SSIM are blind to).

    Parameters
    ----------
    image : torch.Tensor
        ``(H, W)`` or ``(1, 1, H, W)`` slice.
    rois : sequence[tuple[int, int]]
        ``(y, x)`` centres of flat ROIs (e.g. from
        :func:`ctdenoiser.detectability.sample_flat_locations`). Each ROI is
        detrended (mean removed) before the FFT so only the *noise* texture,
        not the slowly varying anatomy, contributes.
    roi_size : int
        Side length of each square ROI.

    Returns
    -------
    dict
        ``{"radial_nps", "peak_freq", "mean_freq", "total_power"}``.
        ``radial_nps`` is the azimuthally-averaged power vs. frequency-bin index;
        ``peak_freq`` is the (normalised, 0..1) frequency at which the NPS peaks;
        ``mean_freq`` is the power-weighted spectral centroid — a robust summary
        of where the noise energy sits, so that a denoiser pushing power to low
        frequency (blotchy / "waxy" texture) shows up as a *drop* in ``mean_freq``
        even when the spectrum has no sharp peak; ``total_power`` is the mean
        noise variance over the ROIs.
    """
    img = image.reshape(image.shape[-2:])
    h, w = img.shape
    half = roi_size // 2
    spectra = []
    total_power = 0.0
    n = 0
    for cy, cx in rois:
        y0, x0 = cy - half, cx - half
        if y0 < 0 or x0 < 0 or y0 + roi_size > h or x0 + roi_size > w:
            continue
        roi = img[y0 : y0 + roi_size, x0 : x0 + roi_size]
        roi = roi - roi.mean()  # detrend so anatomy gradient does not leak in
        spec = torch.fft.fftshift(torch.fft.fft2(roi)).abs() ** 2 / (roi_size ** 2)
        spectra.append(spec)
        total_power += float((roi ** 2).mean().item())
        n += 1
    if n == 0:
        raise ValueError("no in-bounds ROIs for uniform_nps")
    mean_spec = torch.stack(spectra, 0).mean(0)
    radial = _radial_profile(mean_spec)
    nyquist = len(radial) - 1
    # Drop the DC bin (index 0) for both summaries; it carries no texture info.
    ac = radial[1:]
    freqs = torch.arange(1, len(radial), device=radial.device, dtype=radial.dtype)
    peak_freq = (int(torch.argmax(ac).item()) + 1) / nyquist
    mean_freq = float((freqs * ac).sum() / (ac.sum().clamp_min(1e-12))) / nyquist
    return {
        "radial_nps": radial,
        "peak_freq": peak_freq,
        "mean_freq": mean_freq,
        "total_power": total_power / n,
    }
