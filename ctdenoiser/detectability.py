"""Task-based, hallucination-aware detectability evaluation.

This module operationalises Phase 1 of ``docs/future.md``: instead of asking
only "how close is the denoised image to the reference" (PSNR/SSIM/RMSE), it
asks the clinically meaningful question — *is a known low-contrast signal
preserved, erased, or fabricated by the denoiser?*

Key design insight (see ``docs/plan.md``): no CT forward projector is needed.
The dataset has real paired low/full acquisitions, so the real, spatially
correlated FBP noise is simply ``n = low - full``. To synthesise a
signal-present low-dose image we add a known lesion ``s`` to the clean image and
re-attach that same real noise::

    low_present = full + s + n = (low - full) + full + s = low + s
    low_absent  = low                     # the real acquisition, untouched

So the signal-present image is *literally* ``low + s`` — real correlated CT
noise carrying a lesion of known location / size / contrast. The standard
low-contrast signal-insertion assumption (the small signal does not perturb the
noise field) is documented and respected by keeping the contrast low.

Detectability is then quantified with a **Channelized Hotelling Observer (CHO)**
for the SKE/BKS task (Signal Known Exactly, Background Known Statistically),
which yields a detectability index ``d'`` and the corresponding AUC. Running the
present/absent ensembles through a denoiser and comparing ``d'`` of the
input / denoised / clean images measures whether the denoiser keeps the lesion
detectable (preserved), washes it out (erased), or invents structure where none
exists (fabricated).

References
----------
- Barrett & Myers, *Foundations of Image Science* (Hotelling observer, LG channels).
- ICRU / AAPM TG-233 task-based image-quality assessment.
"""

import math

import torch

# --------------------------------------------------------------------------- #
# Signal insertion
# --------------------------------------------------------------------------- #

def insert_signal(clean, center, radius_px, contrast_hu, hu_scale, profile="disk"):
    """Add a low-contrast lesion to a ``[0, 1]``-normalised CT slice.

    Parameters
    ----------
    clean : torch.Tensor
        ``(..., H, W)`` slice(s) normalised to ``[0, 1]``. Not modified in place.
    center : tuple[int, int]
        ``(y, x)`` pixel coordinate of the lesion centre.
    radius_px : float
        Lesion radius in pixels. For ``profile="gaussian"`` this is the standard
        deviation; for ``profile="disk"`` it is the hard radius.
    contrast_hu : float
        Lesion contrast in Hounsfield units (HU). Positive = brighter than
        background. Converted to a normalised amplitude by ``contrast_hu /
        hu_scale`` (e.g. ``10 HU / 400 = 0.025``), matching the dataset's
        ``normalize_hu`` window so the same HU contrast means the same thing for
        every anatomy preset.
    hu_scale : float
        HU width of the normalisation window (the anatomy preset's ``hu_scale``).
    profile : {"disk", "gaussian"}
        ``"disk"`` adds a flat disk; ``"gaussian"`` adds a radially Gaussian
        bump (smoother, closer to a real low-contrast lesion).

    Returns
    -------
    torch.Tensor
        ``clean + signal`` with the same shape/dtype/device as ``clean``. The
        added signal itself is available via :func:`signal_template`.
    """
    sig = signal_template(
        clean.shape[-2:], center, radius_px, contrast_hu, hu_scale,
        profile=profile, device=clean.device, dtype=clean.dtype,
    )
    return clean + sig


def signal_template(shape, center, radius_px, contrast_hu, hu_scale,
                    profile="disk", device=None, dtype=torch.float32):
    """Return the additive lesion ``s`` itself as an ``(H, W)`` tensor.

    Same parameters as :func:`insert_signal`; exposed separately because the CHO
    template and the contrast-recovery measurement both need the signal alone,
    not an image with the signal added.
    """
    h, w = shape
    amplitude = contrast_hu / hu_scale
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    cy, cx = center
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    if profile == "disk":
        s = (r2 <= radius_px ** 2).to(dtype) * amplitude
    elif profile == "gaussian":
        s = torch.exp(-r2 / (2.0 * radius_px ** 2)) * amplitude
    else:
        raise ValueError(f"unknown profile {profile!r}; use 'disk' or 'gaussian'")
    return s


# --------------------------------------------------------------------------- #
# ROI selection
# --------------------------------------------------------------------------- #

def body_mask(image, threshold=0.05):
    """Boolean ``(H, W)`` mask of the patient body (foreground) in a ``[0,1]`` slice.

    Air normalises near 0; a low threshold separates body from background. Used
    to keep inserted lesions and flat ROIs inside tissue rather than in air.
    """
    img = image if image.dim() == 2 else image.reshape(image.shape[-2:])
    return img > threshold


def sample_flat_locations(image, n_locations, roi_size, mask=None,
                          var_quantile=0.5, min_separation=None, seed=0):
    """Sample ``(y, x)`` ROI centres on flat background inside the body.

    A lesion must be inserted on a *flat* region (no edges/vessels) so that the
    only structured signal in the ROI is the one we added — the BKS assumption.
    Candidate centres are ranked by local variance and the flattest are kept.

    Parameters
    ----------
    image : torch.Tensor
        ``(H, W)`` or ``(1, 1, H, W)`` clean reference slice.
    n_locations : int
        Number of lesion centres to return.
    roi_size : int
        Side length of the square ROI (used to keep ROIs in-bounds and to size
        the local-variance window).
    mask : torch.Tensor, optional
        Boolean body mask; defaults to :func:`body_mask`.
    var_quantile : float
        Keep candidates whose local variance is below this quantile (flattest
        first). Lower = flatter / stricter.
    min_separation : int, optional
        Minimum centre-to-centre distance (pixels) so ROIs do not overlap;
        defaults to ``roi_size``.
    seed : int
        RNG seed for the random pick among the flat candidates.

    Returns
    -------
    list[tuple[int, int]]
        ``(y, x)`` integer centres, length ``<= n_locations``.
    """
    import torch.nn.functional as F

    img = image.reshape(image.shape[-2:])
    h, w = img.shape
    half = roi_size // 2
    min_separation = roi_size if min_separation is None else min_separation
    if mask is None:
        mask = body_mask(img)

    # Local variance via box filter: E[x^2] - E[x]^2 over a roi-sized window.
    x = img[None, None]
    k = max(3, roi_size | 1)  # odd kernel
    pad = k // 2
    ones = torch.ones(1, 1, k, k, device=img.device, dtype=img.dtype) / (k * k)
    mean = F.conv2d(x, ones, padding=pad)
    mean2 = F.conv2d(x * x, ones, padding=pad)
    local_var = (mean2 - mean ** 2).reshape(h, w).clamp_min(0.0)

    # Valid centres: ROI fully in-bounds, centre inside body, flat enough.
    valid = torch.zeros(h, w, dtype=torch.bool, device=img.device)
    valid[half : h - half, half : w - half] = True
    valid &= mask
    var_in = local_var[valid]
    if var_in.numel() == 0:
        return []
    thresh = torch.quantile(var_in, var_quantile)
    valid &= local_var <= thresh

    ys, xs = torch.nonzero(valid, as_tuple=True)
    if ys.numel() == 0:
        return []
    g = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(ys.numel(), generator=g)

    chosen: list[tuple[int, int]] = []
    for idx in order.tolist():
        cy, cx = int(ys[idx]), int(xs[idx])
        if all((cy - py) ** 2 + (cx - px) ** 2 >= min_separation ** 2
               for py, px in chosen):
            chosen.append((cy, cx))
            if len(chosen) >= n_locations:
                break
    return chosen


def extract_rois(image, centers, roi_size):
    """Stack the ``roi_size`` square patches centred on ``centers``.

    Returns ``(N, roi_size, roi_size)``. Centres whose ROI would fall out of
    bounds are skipped, so the returned count may be ``< len(centers)``.
    """
    img = image.reshape(image.shape[-2:])
    h, w = img.shape
    half = roi_size // 2
    rois = []
    for cy, cx in centers:
        y0, x0 = cy - half, cx - half
        if y0 < 0 or x0 < 0 or y0 + roi_size > h or x0 + roi_size > w:
            continue
        rois.append(img[y0 : y0 + roi_size, x0 : x0 + roi_size])
    if not rois:
        return torch.empty(0, roi_size, roi_size, device=img.device, dtype=img.dtype)
    return torch.stack(rois, dim=0)


# --------------------------------------------------------------------------- #
# Channelized Hotelling Observer
# --------------------------------------------------------------------------- #

def _laguerre(n, x):
    """Laguerre polynomial ``L_n(x)`` via the three-term recurrence."""
    if n == 0:
        return torch.ones_like(x)
    lm1, l = torch.ones_like(x), 1.0 - x
    for k in range(1, n):
        lm1, l = l, ((2 * k + 1 - x) * l - k * lm1) / (k + 1)
    return l


def laguerre_gauss_channels(roi_size, n_channels=10, spread=None,
                            device=None, dtype=torch.float32):
    """Orthonormal Laguerre-Gauss channel matrix ``U`` of shape ``(P, C)``.

    ``P = roi_size**2`` flattened pixels, ``C = n_channels``. LG channels are
    rotationally symmetric (matched to compact, roughly circular lesions) and
    span a low-dimensional efficient subspace for the Hotelling observer, which
    is what makes the channelized observer tractable and its covariance
    estimable from a modest number of ROIs.

    The raw channels are orthonormalised (QR) so ``U.T @ U = I``; this keeps the
    ``d'`` computation numerically stable and makes the white-noise case
    analytically clean (``d' = ||signal|| / sigma`` when the signal lies in the
    channel span), which is exactly the property the unit test checks.
    """
    spread = roi_size / 4.0 if spread is None else spread
    half = (roi_size - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(roi_size, device=device, dtype=dtype) - half,
        torch.arange(roi_size, device=device, dtype=dtype) - half,
        indexing="ij",
    )
    r2 = (yy ** 2 + xx ** 2) / (spread ** 2)
    gauss = torch.exp(-r2 / 2.0)
    cols = []
    for n in range(n_channels):
        cols.append((_laguerre(n, r2) * gauss).reshape(-1))
    u = torch.stack(cols, dim=1)  # (P, C)
    # Orthonormalise so U.T @ U = I (reduced QR; sign-agnostic for the observer).
    q, _ = torch.linalg.qr(u, mode="reduced")
    return q


def cho_detectability(present_rois, absent_rois, signal=None, n_channels=10,
                      channels=None, ridge=1e-6):
    """Channelized Hotelling Observer detectability for an SKE/BKS task.

    Parameters
    ----------
    present_rois, absent_rois : torch.Tensor
        ``(N, roi, roi)`` (or ``(N, P)``) ensembles of signal-present and
        signal-absent ROIs. Need not be the same size.
    signal : torch.Tensor, optional
        The known signal ROI ``(roi, roi)`` (SKE template). If given, the
        Hotelling template direction uses the *exact* signal channel response
        (lower variance, the canonical SKE observer); otherwise the empirical
        present/absent channel-mean difference is used (the BKS-only estimate).
    n_channels : int
        Number of Laguerre-Gauss channels when ``channels`` is not supplied.
    channels : torch.Tensor, optional
        Precomputed ``(P, C)`` channel matrix; overrides ``n_channels``.
    ridge : float
        Diagonal loading on the channel covariance for invertibility.

    Returns
    -------
    dict
        ``{"d_prime": float, "auc": float, "n_present": int, "n_absent": int}``.
        ``d'`` is the detectability index; ``auc`` is the area under the ROC of
        the resulting linear discriminant, ``Phi(d'/sqrt(2))``.
    """
    p = present_rois.reshape(present_rois.shape[0], -1).double()
    a = absent_rois.reshape(absent_rois.shape[0], -1).double()
    n_pix = p.shape[1]
    roi_size = int(round(math.sqrt(n_pix)))

    if channels is None:
        channels = laguerre_gauss_channels(
            roi_size, n_channels=n_channels, device=p.device, dtype=torch.float64
        )
    u = channels.double()  # (P, C)

    # Channelise both ensembles: (N, C).
    vp = p @ u
    va = a @ u

    # Template direction: exact signal (SKE) if available, else empirical means.
    if signal is not None:
        delta = (signal.reshape(-1).double() @ u)  # (C,)
    else:
        delta = vp.mean(0) - va.mean(0)

    # Pooled intra-class channel covariance (each class centred on its own mean).
    cp = torch.cov(vp.T) if vp.shape[0] > 1 else torch.zeros(u.shape[1], u.shape[1])
    ca = torch.cov(va.T) if va.shape[0] > 1 else torch.zeros(u.shape[1], u.shape[1])
    cov = 0.5 * (cp + ca)
    cov = cov + ridge * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)

    # Hotelling: w = K^{-1} delta;  d'^2 = delta^T K^{-1} delta.
    w = torch.linalg.solve(cov, delta)
    d2 = float(torch.dot(delta, w).clamp_min(0.0))
    d_prime = math.sqrt(d2)
    auc = 0.5 * (1.0 + math.erf(d_prime / 2.0))
    return {
        "d_prime": d_prime,
        "auc": auc,
        "n_present": int(p.shape[0]),
        "n_absent": int(a.shape[0]),
    }
