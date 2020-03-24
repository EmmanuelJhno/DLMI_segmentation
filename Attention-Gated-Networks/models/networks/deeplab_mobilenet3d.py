import math
import torch.nn as nn
from .utils import ASPP3d, InvertedResidual, _make_divisible, Conv3dBNReLU
import torch.nn.functional as F
from models.networks_other import init_weights


class _MobileNet3d(nn.Module):
    def __init__(self,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet3d
        """
        super(_MobileNet3d, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [Conv3dBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(Conv3dBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

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
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        # x = nn.functional.adaptive_avg_pool3d(x, 1).reshape(x.shape[0], -1)
        return x


class deeplab_mobilenet3d(nn.Module):

    def __init__(self, n_classes=21, in_channels=3, atrous_rates=None, width_mult=1.0):
        super(deeplab_mobilenet3d, self).__init__()
        self.in_channels = in_channels
        if atrous_rates is None:
            atrous_rates = [6, 12, 18]
        self.atrous_rates = atrous_rates
        self.width_mult = width_mult

        # Feature extraction through MobileNet
        self.mobilenet = _MobileNet3d(width_mult=width_mult)

        # A-trou Spatial Pyramid Pooling
        self.aspp = ASPP3d(in_channels=self.mobilenet.last_channel,
                           atrous_rates=self.atrous_rates)

        # Final convolution
        self.final = nn.Conv3d(in_channels=self.aspp.last_channel,
                               out_channels=n_classes,
                               kernel_size=1)

        # initialise weigths
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        features = self.mobilenet(inputs)
        aspp = self.aspp(features)
        final = self.final(aspp)
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p
