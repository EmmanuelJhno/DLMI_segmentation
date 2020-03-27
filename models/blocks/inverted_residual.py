from torch import nn

from models.blocks import Conv3dBNReLU


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # point-wise convolution
            layers.append(Conv3dBNReLU(in_channels, hidden_dim, kernel_size=1))
        layers.extend([
            # depth-wise convolution
            Conv3dBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # point-wise convolution
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
