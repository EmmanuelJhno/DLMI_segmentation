from .mobilenet3d import MobileNet3d, Conv3dBNReLU
from .deeplab3d import ASPP3d
import torch
from torch import nn
import torch.nn.functional as F

class DeeplabMobilenet3d(nn.Module):

    def __init__(self, n_classes=21, in_channels=3, atrous_rates=None, width_mult=1.0):
        super(DeeplabMobilenet3d, self).__init__()
        self.in_channels = in_channels
        if atrous_rates is None:
            atrous_rates = [6, 12, 18]
        self.atrous_rates = atrous_rates
        self.width_mult = width_mult

        # Feature extraction through MobileNet
        self.mobilenet = MobileNet3d(width_mult=width_mult, in_channels=in_channels)

        # A-trou Spatial Pyramid Pooling
        self.aspp = ASPP3d(in_channels=self.mobilenet.last_channel,
                           atrous_rates=self.atrous_rates)

        # Low-level features decoder
        self.low_conv = Conv3dBNReLU(in_channels=self.mobilenet.low_level_feats_channels,
                                     out_channels=self.aspp.last_channel,
                                     kernel_size=1)

        # Final convolution
        self.final_conv = nn.Sequential(
            Conv3dBNReLU(in_channels=2*self.aspp.last_channel,
                         out_channels=256,
                         kernel_size=3),
            Conv3dBNReLU(in_channels=256,
                         out_channels=256,
                         kernel_size=3),
            nn.Conv3d(in_channels=256,
                      out_channels=n_classes,
                      kernel_size=3),
        )

        # initialise weigths
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        high_level_feats, low_level_feats = self.mobilenet(inputs)

        # ASPP and upsampling
        aspp = self.aspp(high_level_feats)
        aspp = F.interpolate(aspp, size=low_level_feats.size()[2:],
                             mode='trilinear', align_corners=True)

        # Low-level feature extraction
        low_level_feats = self.low_conv(low_level_feats)

        # Concat and decode
        x = torch.cat([low_level_feats, aspp], dim=1)
        x = self.final_conv(x)
        x = F.interpolate(x, size=inputs.size()[2:],
                          mode='trilinear', align_corners=True)
        return x

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p
