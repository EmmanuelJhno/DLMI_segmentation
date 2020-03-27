import torch
from torch import nn
from torch.nn import functional as F

from models.backbone import MobileNet3d
from models.blocks import ASPP3d, Conv3dBNReLU

BACKBONES = {"mobilenet": MobileNet3d}


class DeepLab3D(nn.Module):
    """
    Class for DeepLab (V3+) framework for semantic segmentation.

    Args:
        config: Must contain following attributes:
            num_classes (int): Number of output classes in the mask;
            in_channels (int): Number of channels in the input image;
            backbone (str, optional): the backbone to use. e.g. "mobilenet";
            backbone_kwargs (dict, optional): keyword args to pass to the backbone constructor
        loss_type (str)

    Attributes:
        num_classes (int): Number of classes in the output mask
        in_channels (int): Number of channels in the input imahge
        atrous_rates (list of int): Dilatation factor for a-trou convolutions in the
            A-trous Spatial Pyramid Pooling block. Defaults to [6, 12, 18]
        backbone_name (str): name of the backbone used (eg "mobilenet")
        backbone (models.backbone.Backbone): The actual backbone network
        aspp (models.blocks.ASPP3d): A-trous Spatial Pyramid Pooling block
        low_level_decoder (nn.Sequential): Convolutional decoder for low-level features
        decoder (nn.Sequential): Convolutional decoder for the concatenated decoded low-level features
            and the ASPP features
        loss_type (str, optional)
    """

    def __init__(self, config, loss_type = None):
        super(DeepLab3D, self).__init__()
        assert hasattr(config, "num_classes")
        assert hasattr(config, "in_channels")

        if not hasattr(config, "atrous_rates"):
            print("atrous rates not specified in config, setting to default [6, 12, 18]")
            setattr(config, "atrous_rates", [6, 12, 18])

        self.num_classes = config.num_classes
        self.in_channels = config.in_channels
        self.atrous_rates = config.atrous_rates
        self.loss_type = loss_type

        if hasattr(config, "backbone"):
            backbone_name = config.backbone
        else:
            backbone_name = "mobilenet"

        if not (hasattr(config, "backbone_kwargs") and isinstance(config.backbone_kwargs, dict)):
            setattr(config, "backbone_kwargs", {})

        self.backbone_name = backbone_name

        if self.backbone_name in BACKBONES:
            self.backbone = BACKBONES[self.backbone_name](in_channels=config.in_channels,
                                                          **config.backbone_kwargs)
        else:
            print("Unknown backbone; setting to default MobileNet3D")
            self.backbone = BACKBONES["mobilenet"](in_channels=config.in_channels,
                                                   **config.backbone_kwargs)

        self.aspp = ASPP3d(in_channels=self.backbone.hlf_channels,
                           atrous_rates=self.atrous_rates)

        self.low_level_decoder = nn.Sequential(
            Conv3dBNReLU(in_channels=self.backbone.llf_channels,
                         out_channels=self.aspp.last_channel,
                         kernel_size=1))
        self.decoder = nn.Sequential(
            Conv3dBNReLU(in_channels=2 * self.aspp.last_channel,
                         out_channels=256,
                         kernel_size=3),
            nn.Conv3d(in_channels=256,
                      out_channels=self.num_classes,
                      kernel_size=3)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        # Feature extraction with backbone
        low_level_feats, high_level_feats = self.backbone(inputs)

        # A-trous Spatial Pyramid Pooling on HL feats and upsampling
        aspp = self.aspp(high_level_feats)
        aspp = F.interpolate(aspp,
                             size=low_level_feats.size()[2:],
                             mode="trilinear",
                             align_corners=True)

        # Decoding LL feats
        low_level_feats = self.low_level_decoder(low_level_feats)

        # Concatenate and decode
        x = torch.cat([low_level_feats, aspp], dim=1)
        x = self.decoder(x)

        # Upsample
        x = F.interpolate(x,
                          size=inputs.size()[2:],
                          mode="trilinear",
                          align_corners=True)

        if self.loss_type is None:
            return x
        else:
            raise NotImplementedError('Unknow loss type')