"""Conditional flow matching (rectified flow) for CT denoising."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, t):
        # t: [B] or scalar -> [B, embed_dim]
        t = t.view(-1)
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # [B, embed_dim]
        return self.mlp(emb)


class VelocityUNet(nn.Module):
    """U-Net-like velocity network conditioned on time embedding."""

    def __init__(self, num_filters=96, embed_dim=64):
        super().__init__()
        # Input is [B, 2, H, W]: x_t concatenated with noisy condition
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, num_filters, 5, padding=2),
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
        # Project time embedding to per-encoder-block bias
        self.t_proj1 = nn.Linear(embed_dim, num_filters)
        self.t_proj2 = nn.Linear(embed_dim, num_filters)

    def forward(self, x_t, cond, t_emb):
        # x_t, cond: [B,1,H,W]; t_emb: [B, embed_dim]
        inp = torch.cat([x_t, cond], dim=1)  # [B,2,H,W]

        e1 = self.enc1(inp)
        e1 = e1 + self.t_proj1(t_emb)[:, :, None, None]

        e2 = self.enc2(e1)
        e2 = e2 + self.t_proj2(t_emb)[:, :, None, None]

        b = self.bottleneck(e2)

        d2 = self.dec2(torch.cat([b, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))
        return out


class ConditionalFlowMatching(nn.Module):
    """Rectified-flow model: learns straight-line ODE from noisy to clean CT.

    Training calls flow_loss(x0, x1); inference calls forward(x) which runs
    the Euler ODE solve, preserving the standard [B,1,H,W]->[B,1,H,W] contract.
    """

    def __init__(self, num_filters=96, embed_dim=64, num_steps=20):
        super().__init__()
        self.num_steps = num_steps
        self.time_emb = SinusoidalTimeEmbedding(embed_dim)
        self.net = VelocityUNet(num_filters=num_filters, embed_dim=embed_dim)

    def _velocity(self, y, cond, t_scalar):
        B = y.size(0)
        t = torch.full((B,), t_scalar, device=y.device, dtype=y.dtype)
        t_emb = self.time_emb(t)
        return self.net(y, cond, t_emb)

    def flow_loss(self, x0, x1):
        """CFM training loss. x0=noisy CT, x1=clean CT."""
        B = x0.size(0)
        t = torch.rand(B, 1, 1, 1, device=x0.device, dtype=x0.dtype)
        x_t = (1 - t) * x0 + t * x1
        v_target = x1 - x0  # constant velocity for rectified flow

        t_flat = t.view(B)
        t_emb = self.time_emb(t_flat)
        v_pred = self.net(x_t, x0, t_emb)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def forward(self, x):
        """Euler ODE solve: noisy CT -> denoised CT."""
        y = x.clone()
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t_scalar = i * dt
            y = y + dt * self._velocity(y, x, t_scalar)
        return y.clamp(0.0, 1.0)
