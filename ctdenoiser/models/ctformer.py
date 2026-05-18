"""CTformer: Token2Token Dilation transformer for low-dose CT denoising.

Modularized so the latent representation can be edited or the standard
Multi-Head Attention swapped for a Performer variant if training memory
becomes a bottleneck.
"""

import torch
import torch.nn as nn


class CyclicShift(nn.Module):
    def __init__(self, shift_size):
        super().__init__()
        self.shift_size = shift_size

    def forward(self, x):
        # x: (B, C, H, W)
        return torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = res + x

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        return res + x


class T2TDilation(nn.Module):
    """Dilated Token2Token re-tokenization with a cyclic shift.

    Spatial dimensions are preserved (stride=1, padding=dilation), so the
    same module serves as the symmetric inverse (IT2TD) by negating
    ``shift_size``.
    """

    def __init__(self, in_dim, out_dim, kernel_size, dilation, shift_size):
        super().__init__()
        self.cyclic_shift = CyclicShift(shift_size)
        self.unfold = nn.Unfold(
            kernel_size=kernel_size, dilation=dilation, padding=dilation, stride=1
        )
        self.proj = nn.Linear(in_dim * kernel_size**2, out_dim)

    def forward(self, x, spatial_size):
        B, N, C = x.shape
        H, W = spatial_size

        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.cyclic_shift(x)
        x = self.unfold(x)            # (B, C*K*K, L)
        x = x.transpose(1, 2)         # (B, L, C*K*K)
        x = self.proj(x)
        return x, (H, W)


class CTformer(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64):
        super().__init__()

        self.tokenization = nn.Conv2d(
            in_channels, embed_dim, kernel_size=7, stride=2, padding=3
        )

        # Encoder: Transformer block + dilated T2T
        self.enc_tb1 = TransformerBlock(dim=embed_dim)
        self.t2td1 = T2TDilation(embed_dim, embed_dim, kernel_size=3, dilation=2, shift_size=2)

        self.enc_tb2 = TransformerBlock(dim=embed_dim)
        self.t2td2 = T2TDilation(embed_dim, embed_dim, kernel_size=3, dilation=1, shift_size=2)

        self.bottleneck_tb = TransformerBlock(dim=embed_dim)

        # Decoder: inverse T2T + Transformer block
        self.it2td1 = T2TDilation(embed_dim, embed_dim, kernel_size=3, dilation=1, shift_size=-2)
        self.dec_tb1 = TransformerBlock(dim=embed_dim)

        self.it2td2 = T2TDilation(embed_dim, embed_dim, kernel_size=3, dilation=2, shift_size=-2)
        self.dec_tb2 = TransformerBlock(dim=embed_dim)

        self.detokenization = nn.ConvTranspose2d(
            embed_dim, in_channels, kernel_size=7, stride=2, padding=3, output_padding=1
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.tokenization(x)                       # (B, D, H/2, W/2)
        spatial_shape = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)               # (B, N, D)

        x = self.enc_tb1(x)
        x, spatial_shape = self.t2td1(x, spatial_shape)
        x = self.enc_tb2(x)
        x, spatial_shape = self.t2td2(x, spatial_shape)

        x = self.bottleneck_tb(x)

        x, spatial_shape = self.it2td1(x, spatial_shape)
        x = self.dec_tb1(x)
        x, spatial_shape = self.it2td2(x, spatial_shape)
        x = self.dec_tb2(x)

        x = x.transpose(1, 2).view(B, -1, spatial_shape[0], spatial_shape[1])
        x = self.detokenization(x)
        return x
