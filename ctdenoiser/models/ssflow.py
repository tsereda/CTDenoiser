"""Self-supervised rectified flow with similarity-based noisy pairs.

A label-free denoiser built on rectified flow. Two independent noisy
observations of the same scene are manufactured from a single noisy slice and
used as the flow endpoints; an *unconditional* velocity network is trained on
them. Because the network sees only the interpolant ``z_t`` (and the time
``t``), and not the noisy input as a conditioning channel, the minimiser of the
flow-matching loss is the marginal velocity
``v*(z, t) = E[x1 - x0 | z_t = z]``. Evaluated at ``t = 0`` this gives the
Noise2Noise posterior mean ``x0 + v*(x0, 0) = E[s | x0]`` (the clean signal),
so the flow denoises without any clean target.

The crucial difference from :class:`ConditionalFlowMatching` is the *absence* of
the ``x0`` conditioning: feeding ``x0`` alongside ``z_t`` would let the network
solve ``x1 = (z_t - (1 - t) x0) / t`` exactly and reproduce the target noise,
destroying the averaging that cancels it.

Pair construction:
  * ``similarity`` (v2, default): Noise2Sim-style non-local self-similar patches,
    excluding candidates within ``exclude_radius`` so the paired noise is
    decorrelated -- the variant intended for spatially correlated CT noise.
  * ``downsample`` (v1): Neighbor2Neighbor / ZS-N2N diagonal downsampling pairs,
    valid only when the noise correlation length is below the pixel pitch.

References: Lipman et al. (flow matching, 2023); Liu et al. (rectified flow,
2023); Lehtinen et al. (Noise2Noise, 2018); Niu & Wang (Noise2Sim, 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flowmatching import SinusoidalTimeEmbedding
from ..zeroshot import pair_downsampler


@torch.no_grad()
def make_similarity_pair(
    noisy, search_radius=4, patch_radius=1, num_similar=1, exclude_radius=2
):
    """Build a non-local "second noisy look" ``x1`` for each pixel of ``noisy``.

    For every pixel we scan all spatial offsets in a ``(2*search_radius+1)``
    window, score each candidate by the mean squared patch difference over a
    ``(2*patch_radius+1)`` neighbourhood, and replace the pixel by the average of
    its ``num_similar`` best matches. Offsets with ``max(|dy|, |dx|) <
    exclude_radius`` are skipped so the matched pixel lies beyond the noise
    correlation length and carries (approximately) independent noise; this is the
    key correlated-noise knob. Returns a tensor shaped like ``noisy``
    ``(B, 1, H, W)``.

    With ``exclude_radius=1`` only the pixel itself is excluded, recovering the
    Noise2Sim target.
    """
    b, c, h, w = noisy.shape
    r, pr = search_radius, patch_radius
    padded = F.pad(noisy, (r, r, r, r), mode="replicate")

    dists, vals = [], []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if max(abs(dy), abs(dx)) < exclude_radius:
                continue  # too close: shares correlated noise with the query
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
    idx = dist_stack.topk(k, dim=0, largest=False).indices
    chosen = torch.gather(val_stack, 0, idx)
    return chosen.mean(dim=0)


class UnconditionalVelocityUNet(nn.Module):
    """Velocity net for unconditional rectified flow: ``v(z_t, t)``.

    Same U-Net body as :class:`ConditionalFlowMatching`'s velocity network, but
    the input is the single-channel interpolant only -- there is no noisy-image
    conditioning channel. That omission is what makes the learned marginal
    velocity a Noise2Noise estimator (see module docstring).
    """

    def __init__(self, num_filters=96, embed_dim=64):
        super().__init__()
        # Input is [B, 1, H, W]: the interpolant z_t alone (no condition).
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, num_filters, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, num_filters, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, num_filters, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, num_filters, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, 1, 5, padding=2),
        )
        self.t_proj1 = nn.Linear(embed_dim, num_filters)
        self.t_proj2 = nn.Linear(embed_dim, num_filters)

    def forward(self, z, t_emb):
        # z: [B,1,H,W]; t_emb: [B, embed_dim]
        e1 = self.enc1(z)
        e1 = e1 + self.t_proj1(t_emb)[:, :, None, None]

        e2 = self.enc2(e1)
        e2 = e2 + self.t_proj2(t_emb)[:, :, None, None]

        b = self.bottleneck(e2)

        d2 = self.dec2(torch.cat([b, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))
        return out


class SelfSupervisedFlow(nn.Module):
    """Unconditional rectified flow trained on self-supervised noisy pairs.

    Training calls :meth:`ss_flow_loss(noisy)`; inference calls
    :meth:`forward(x)`, preserving the standard ``[B,1,H,W] -> [B,1,H,W]``
    contract used by the harness. ``num_steps <= 1`` uses the one-step posterior
    mean ``x + v(x, 0)`` (equivalent to Noise2Noise/Noise2Sim regression);
    larger ``num_steps`` runs an Euler ODE solve from the noisy input, which is
    the multi-step refinement the method's central ablation tests.
    """

    def __init__(
        self,
        num_filters=96,
        embed_dim=64,
        num_steps=1,
        pairing="similarity",
        search_radius=4,
        patch_radius=1,
        num_similar=1,
        exclude_radius=2,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.pairing = pairing
        self.search_radius = search_radius
        self.patch_radius = patch_radius
        self.num_similar = num_similar
        self.exclude_radius = exclude_radius
        self.time_emb = SinusoidalTimeEmbedding(embed_dim)
        self.net = UnconditionalVelocityUNet(num_filters=num_filters, embed_dim=embed_dim)

    def _build_pair(self, noisy):
        """Manufacture two independent noisy observations ``(x0, x1)``."""
        if self.pairing == "downsample":
            # v1: Neighbor2Neighbor / ZS-N2N half-resolution pair.
            return pair_downsampler(noisy)
        # v2: similarity-based non-local pair at full resolution.
        x1 = make_similarity_pair(
            noisy,
            search_radius=self.search_radius,
            patch_radius=self.patch_radius,
            num_similar=self.num_similar,
            exclude_radius=self.exclude_radius,
        )
        return noisy, x1

    def _velocity(self, z, t_scalar):
        n = z.size(0)
        t = torch.full((n,), t_scalar, device=z.device, dtype=z.dtype)
        return self.net(z, self.time_emb(t))

    def ss_flow_loss(self, noisy):
        """Flow-matching loss on self-supervised endpoints. ``noisy``: [B,1,H,W]."""
        x0, x1 = self._build_pair(noisy)
        n = x0.size(0)
        t = torch.rand(n, 1, 1, 1, device=x0.device, dtype=x0.dtype)
        z_t = (1 - t) * x0 + t * x1
        v_target = x1 - x0  # constant rectified-flow velocity
        v_pred = self.net(z_t, self.time_emb(t.view(n)))
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def forward(self, x):
        """Denoise ``x`` ([B,1,H,W]). One-step posterior mean or k-step Euler."""
        if self.num_steps is None or self.num_steps <= 1:
            return (x + self._velocity(x, 0.0)).clamp(0.0, 1.0)
        y = x.clone()
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            y = y + dt * self._velocity(y, i * dt)
        return y.clamp(0.0, 1.0)
