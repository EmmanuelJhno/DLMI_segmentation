import torch
from torch import nn
from torch.nn import functional as F

from models.blocks import UnetConv3, MultiAttentionBlock, UnetGridGatingSignal3, UnetUp3_CT, UnetDsv3


class Unet3D(nn.Module):
    """
        Unet3D for semantic segmentation.

        Args:
            config: Must contain following attributes:
                num_classes (int): Number of output classes in the mask;
                in_channels (int): Number of channels in the input image;
                feature_scale (int, optional): factor by which to scale down the number of filters / channels in each block;
                is_deconv (bool, optional): whether to use DeConvolutions;
                is_batchnorm (bool, optional): whether to use Batch Normalization;

        Attributes:
            num_classes (int): Number of classes in the output mask
            in_channels (int): Number of channels in the input image
            is_batchnorm (bool)
            is_deconv (bool)
            feature_scale (int)
        """

    def __init__(self, config):
        super(Unet3D, self).__init__()
        assert hasattr(config, "num_classes")
        assert hasattr(config, "in_channels")

        if not hasattr(config, "feature_scale"):
            print("feature_scale not specified in config, setting to default 4")
            config.feature_scale = 4

        if not hasattr(config, "is_deconv"):
            print("is_deconv not specified in config, setting to default True")
            config.is_deconv = True

        if not hasattr(config, "is_batchnorm"):
            print("is_batchnorm not specified in config, setting to default True")
            config.is_batchnorm = True

        self.num_classes = config.num_classes
        self.in_channels = config.in_channels

        self.is_deconv = config.is_deconv
        self.is_batchnorm = config.is_batchnorm
        self.feature_scale = config.feature_scale

        nonlocal_mode = 'concatenation'
        attention_dsample = (2, 2, 2)

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1),
                                            is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], self.is_deconv)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=self.num_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=self.num_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=self.num_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=self.num_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(self.num_classes * 4, self.num_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.BatchNorm3d):
                classname = m.__class__.__name__
                # print(classname)
                if classname.find('Conv') != -1:
                    nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
                elif classname.find('Linear') != -1:
                    nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
                elif classname.find('BatchNorm') != -1:
                    nn.init.normal(m.weight.data, 1.0, 0.02)
                    nn.init.constant(m.bias.data, 0.0)

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        pred = F.softmax(final, dim=1)
        return pred

#     @staticmethod
#     def apply_argmax_softmax(pred):
#         log_p = F.softmax(pred, dim=1)

#         return log_p
