"""U-Net baseline: 3-level encoder-decoder with skip connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1 = _double_conv(in_channels, f)
        self.enc2 = _double_conv(f, f * 2)
        self.enc3 = _double_conv(f * 2, f * 4)
        self.bottleneck = _double_conv(f * 4, f * 8)
        self.dec3 = _double_conv(f * 8 + f * 4, f * 4)
        self.dec2 = _double_conv(f * 4 + f * 2, f * 2)
        self.dec1 = _double_conv(f * 2 + f, f)
        self.out_conv = nn.Conv2d(f, in_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = self.dec3(torch.cat([F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False), e1], dim=1))
        return self.out_conv(d1)
