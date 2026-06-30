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


# --------------------------------------------------------------------------- #
# End-to-end evaluation over a dataloader
# --------------------------------------------------------------------------- #

def run_detectability_eval(denoise, loader, device, *, hu_scale,
                           contrast_hu=(40.0, 80.0, 160.0), radius_px=4.0,
                           profile="gaussian", roi_size=32, sites_per_slice=6,
                           var_quantile=0.3, n_channels=10, max_slices=0, seed=0):
    """Score a denoiser's task-based detectability over a ``(low, full)`` loader.

    ``denoise`` is any callable mapping a noisy ``(1, 1, H, W)`` slice to a
    denoised one (identity, a trained model wrapped in overlapped inference, ...)
    — keeping this model-agnostic lets ``train.py`` (live model) and
    ``scripts/evaluate_detectability.py`` (loaded checkpoint / identity) share it.

    ``contrast_hu`` may be a single contrast or a sequence of them: passing a
    sequence sweeps a **contrast-detail curve** (the standard CD analysis) in one
    pass, so a denoiser's effect on detectability can be read across the
    sub-threshold→detectable range rather than at one arbitrary operating point.
    The lesion is otherwise fixed (location/size/profile), and the contrast-free
    work — sampling sites, ``denoise(low)``, the NPS — is shared across contrasts.

    For each slice we sample flat lesion sites on the clean reference, insert the
    same known lesion at every (well-separated) site, and build the input,
    denoised and clean present/absent ROI ensembles per the ``low_present =
    low + s`` identity. A CHO ``d'`` is computed per stage and contrast; the
    ``input -> denoised -> clean`` progression shows whether the denoiser
    preserves, erases, or fabricates the lesion. A reference-free NPS centroid is
    also tracked (input vs denoised) to flag blotchy / "waxy" texture.

    Alongside erosion (does a *present* lesion survive?) the eval also reports a
    contrast-free **fabrication index** (does the denoiser *invent* lesion-scale
    structure where the truth has none?): ``d_prime_fabrication`` is a CHO ``d'``
    discriminating the denoiser's signal-absent output from the ground-truth
    clean image, with ``d_prime_fabrication_input`` (the raw noise vs clean) as
    the ~0 floor. Together they map the full erosion <-> fabrication axis that the
    perception--distortion / hallucination literature warns generative denoisers
    trade along.

    Returns a flat dict of scalars for W&B / CSV logging. Per-contrast metrics are
    keyed ``c{hu}/d_prime_*`` etc.; the highest (most detectable) contrast is also
    emitted un-prefixed (``d_prime_*``, ``detectability_preserved``, ...) as the
    headline operating point. NPS / noise-power, fabrication and ``n_slices`` are
    contrast-free.
    """
    from .metrics import uniform_nps

    # Normalise to a sorted ascending list; the last = most detectable headline.
    scalar_in = isinstance(contrast_hu, (int, float))
    contrasts = [float(contrast_hu)] if scalar_in else sorted(float(c) for c in contrast_hu)

    # SKE templates: the lesion as it sits at the centre of an ROI, per contrast.
    templates = [
        signal_template(
            (roi_size, roi_size), (roi_size // 2, roi_size // 2), radius_px,
            c, hu_scale, profile=profile, device=device, dtype=torch.float32,
        )
        for c in contrasts
    ]

    stages = ("input", "denoised", "clean")
    # Absent ensembles are contrast-free (no lesion); present ones are per contrast.
    absent = {s: [] for s in stages}
    present = {(s, ci): [] for s in stages for ci in range(len(contrasts))}
    nps_in, nps_out = [], []

    n_slices = 0
    for low, full in loader:
        low, full = low.to(device), full.to(device)
        clean = full[0, 0]
        sites = sample_flat_locations(
            clean, sites_per_slice, roi_size, var_quantile=var_quantile,
            seed=seed + n_slices,
        )
        if not sites:
            continue

        # Contrast-free work, shared across the whole CD curve.
        den_low = denoise(low)
        absent["input"].append(extract_rois(low[0, 0], sites, roi_size))
        absent["denoised"].append(extract_rois(den_low[0, 0], sites, roi_size))
        absent["clean"].append(extract_rois(full[0, 0], sites, roi_size))
        nps_in.append(uniform_nps(low[0, 0], sites, roi_size=roi_size))
        nps_out.append(uniform_nps(den_low[0, 0], sites, roi_size=roi_size))

        for ci, c in enumerate(contrasts):
            # One signal map with a lesion at every (well-separated) site.
            s_map = torch.zeros_like(clean)
            for cy, cx in sites:
                s_map = s_map + signal_template(
                    clean.shape, (cy, cx), radius_px, c, hu_scale,
                    profile=profile, device=device, dtype=clean.dtype,
                )
            s_map = s_map[None, None]
            present[("input", ci)].append(extract_rois((low + s_map)[0, 0], sites, roi_size))
            present[("denoised", ci)].append(extract_rois(denoise(low + s_map)[0, 0], sites, roi_size))
            present[("clean", ci)].append(extract_rois((full + s_map)[0, 0], sites, roi_size))

        n_slices += 1
        if max_slices and n_slices >= max_slices:
            break

    if n_slices == 0:
        raise RuntimeError("no slices yielded usable flat ROIs")

    results = {"n_slices": n_slices}
    abs_cat = {s: torch.cat(absent[s], 0) for s in stages}
    headline_ci = len(contrasts) - 1

    # Fabrication / hallucination index (contrast-free; the symmetric counterpart
    # to detectability erosion). Erosion asks "is a *present* lesion washed out?";
    # fabrication asks "is structure *invented* where the truth has none?". We
    # answer it with the same Hotelling machinery on the signal-ABSENT ensembles:
    # discriminate the denoiser's output ``denoise(low)`` from the ground-truth
    # clean ``full`` over the flat (lesion-free) ROIs, in the lesion-scale
    # Laguerre-Gauss subspace. The BKS template (empirical channel-mean
    # difference, ``signal=None``) is used deliberately so the fabricated
    # structure need not be a centred lesion. A faithful denoiser removes only
    # zero-mean noise, so its absent output is indistinguishable from truth and
    # ``d'_fab -> 0``; a denoiser that paints blotchy / "waxy" or lesion-like
    # structure becomes separable from truth and ``d'_fab > 0``. Residual noise is
    # zero-mean and so loads the covariance, not the mean difference, isolating
    # *systematic* fabricated structure (the hallucination-bias component). The
    # input-vs-clean value is reported alongside as the noise floor (~0).
    fab = cho_detectability(abs_cat["denoised"], abs_cat["clean"], n_channels=n_channels)
    fab_floor = cho_detectability(abs_cat["input"], abs_cat["clean"], n_channels=n_channels)
    results["d_prime_fabrication"] = fab["d_prime"]
    results["auc_fabrication"] = fab["auc"]
    results["d_prime_fabrication_input"] = fab_floor["d_prime"]
    for ci, c in enumerate(contrasts):
        per = {}
        for st in stages:
            cho = cho_detectability(
                torch.cat(present[(st, ci)], 0), abs_cat[st],
                signal=templates[ci], n_channels=n_channels,
            )
            per[f"d_prime_{st}"] = cho["d_prime"]
            per[f"auc_{st}"] = cho["auc"]
            per[f"n_rois_{st}"] = cho["n_present"]
        d_in = per["d_prime_input"]
        per["detectability_preserved"] = (
            per["d_prime_denoised"] / d_in if d_in > 0 else float("nan")
        )
        # Per-contrast keys; plus un-prefixed headline at the most detectable point.
        results.update({f"c{c:g}/{k}": v for k, v in per.items()})
        if ci == headline_ci:
            results.update(per)

    results["nps_mean_freq_input"] = sum(x["mean_freq"] for x in nps_in) / n_slices
    results["nps_mean_freq_denoised"] = sum(x["mean_freq"] for x in nps_out) / n_slices
    results["noise_power_input"] = sum(x["total_power"] for x in nps_in) / n_slices
    results["noise_power_denoised"] = sum(x["total_power"] for x in nps_out) / n_slices
    return results
