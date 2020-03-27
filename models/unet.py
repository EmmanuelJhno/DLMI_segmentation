import torch
from torch import nn
from torch.nn import functional as F

from models.blocks import UnetConv3, UnetUp3


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
            loss_type (str)

        Attributes:
            num_classes (int): Number of classes in the output mask
            in_channels (int): Number of channels in the input image
            is_batchnorm (bool)
            is_deconv (bool)
            feature_scale (int)
            loss_type (str, optional)
        """

    def __init__(self, config, loss_type=None):
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
        self.loss_type = loss_type

        self.is_deconv = config.is_deconv
        self.is_batchnorm = config.is_batchnorm
        self.feature_scale = config.feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3], self.is_deconv, self.is_batchnorm)
        self.up_concat3 = UnetUp3(filters[3], filters[2], self.is_deconv, self.is_batchnorm)
        self.up_concat2 = UnetUp3(filters[2], filters[1], self.is_deconv, self.is_batchnorm)
        self.up_concat1 = UnetUp3(filters[1], filters[0], self.is_deconv, self.is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], self.num_classes, 1)

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
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        if self.loss_type is None:
            return final
        else:
            raise NotImplementedError('Unknow loss type')

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
