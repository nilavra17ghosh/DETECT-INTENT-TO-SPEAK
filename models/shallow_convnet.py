"""
ShallowConvNet: Shallow Convolutional Network for EEG Decoding
==============================================================

Reference: Schirrmeister et al., Human Brain Mapping, 2017.

Output: single logit for BCEWithLogitsLoss.
"""

import torch
import torch.nn as nn

class SquareLayer(nn.Module):
    def forward(self, x):
        return x ** 2

class LogLayer(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-7))

class ShallowConvNet(nn.Module):
    """
    ShallowConvNet — single-logit output for binary BCE loss.

    Learns log band-power features: log(AvgPool(x²) + ε).
    """

    def __init__(self, n_channels=4, n_samples=128,
                 n_filters=40, kernel_length=25,
                 pool_length=75, pool_stride=15,
                 dropout_rate=0.5):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, n_filters, (1, kernel_length), bias=False)
        self.spatial_conv = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.square = SquareLayer()
        self.pool = nn.AvgPool2d((1, pool_length), stride=(1, pool_stride))
        self.log = LogLayer()
        self.dropout = nn.Dropout(dropout_rate)

        self._feat = self._get_feat(n_channels, n_samples)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self._feat, 1))

    def _get_feat(self, C, T):
        x = torch.zeros(1, 1, C, T)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.log(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.log(x)
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    m = ShallowConvNet(n_channels=4, n_samples=128)
    x = torch.randn(8, 4, 128)
    print(f"Input:  {x.shape}")
    print(f"Output: {m(x).shape}")
    print(f"Params: {m.get_num_params():,}")
