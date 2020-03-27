from torch import nn


class Conv3dBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(Conv3dBNReLU, self).__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU6(inplace=True)
        )
