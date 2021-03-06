import torch
from torch import nn
from torch.nn import functional as F


class ASPPPooling3d(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling3d, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-3:]  # Spatial shape
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)


class ASPPConv3d(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv3d, self).__init__(*modules)


class ASPP3d(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP3d, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv3d(in_channels, out_channels, rate1))
        modules.append(ASPPConv3d(in_channels, out_channels, rate2))
        modules.append(ASPPConv3d(in_channels, out_channels, rate3))
        modules.append(ASPPPooling3d(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

        self.last_channel = out_channels

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
