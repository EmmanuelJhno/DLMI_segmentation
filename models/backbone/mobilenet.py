import torch
from torch import nn

from models.backbone import Backbone
from models.blocks import Conv3dBNReLU, InvertedResidual
from models.utils import _make_divisible


class MobileNet3d(Backbone):
    def __init__(self,
                 in_channels,
                 width_mult=1.0,
                 round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            in_channels (int): Number of input channels
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNet3d, self).__init__(in_channels=in_channels)

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        self.inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [Conv3dBNReLU(in_channels, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(Conv3dBNReLU(input_channel, last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.ModuleList(features)
        self.llf_channels = features[3].out_channels
        self.hlf_channels = last_channel

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for layer in self.features[:4]:
            x = layer(x)
        low_level_feats = x
        for layer in self.features[4:]:
            x = layer(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        # x = nn.functional.adaptive_avg_pool3d(x, 1).reshape(x.shape[0], -1)
        return low_level_feats, x
