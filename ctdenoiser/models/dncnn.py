"""DnCNN baseline (Zhang et al., 2017): flat residual CNN with BN."""

import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_channels=1, num_layers=17, num_filters=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, num_filters, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(num_filters, in_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)
