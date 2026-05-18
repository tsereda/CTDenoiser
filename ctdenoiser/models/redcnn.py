"""RED-CNN baseline (Chen et al., 2017): residual encoder-decoder CNN."""

import torch.nn as nn


class REDCNN(nn.Module):
    def __init__(self, in_channels=1, num_filters=96):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, 5, padding=2)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, padding=2)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, padding=2)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, padding=2)
        self.conv5 = nn.Conv2d(num_filters, num_filters, 5, padding=2)

        self.deconv1 = nn.ConvTranspose2d(num_filters, num_filters, 5, padding=2)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, padding=2)
        self.deconv3 = nn.ConvTranspose2d(num_filters, num_filters, 5, padding=2)
        self.deconv4 = nn.ConvTranspose2d(num_filters, num_filters, 5, padding=2)
        self.deconv5 = nn.ConvTranspose2d(num_filters, in_channels, 5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))

        out = self.deconv1(out)
        out = self.relu(out + residual_3)
        out = self.relu(self.deconv2(out))
        out = self.deconv3(out)
        out = self.relu(out + residual_2)
        out = self.relu(self.deconv4(out))
        out = self.deconv5(out)
        out = self.relu(out + residual_1)
        return out
