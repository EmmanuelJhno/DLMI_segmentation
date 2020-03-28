import torch
from torch import nn
from torch.nn import functional as F

from models.backbone.backbone import Backbone
from models.blocks.resnet import BasicBlock, Bottleneck, conv1x1, conv3x3


class ResNet3D(Backbone):
    DEPTHS = {
        18: (BasicBlock, [2, 2, 2, 2]),
        34: (BasicBlock, [3, 4, 6, 3]),
        50: (Bottleneck, [3, 4, 6, 3]),
        101: (Bottleneck, [3, 4, 23, 3]),
    }

    def __init__(self, in_channels, depth=18,
                 groups=1, width_per_group=64):
        """
        MobileNet V2 main class
        Args:
            in_channels (int): Number of input channels
        """
        super(ResNet3D, self).__init__(in_channels=in_channels)
        self.depth = depth
        if self.depth in self.DEPTHS:
            block, layers = self.DEPTHS[self.depth]
        else:
            print("Unknown depth; defaulting to 18")
            block, layers = self.DEPTHS[18]

        self.inplanes = 64

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(self.in_channels, self.inplanes,
                               kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.llf_channels = 128*BasicBlock.expansion
        self.hlf_channels = 512*BasicBlock.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, channels * block.expansion, stride),
                nn.BatchNorm3d(channels * block.expansion),
            )

        layers = [block(self.inplanes, channels, stride=stride, downsample=downsample, groups=self.groups,
                        base_width=self.base_width)]
        self.inplanes = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, channels, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        low_level_feats = x

        x = self.layer3(x)
        x = self.layer4(x)

        return low_level_feats, x
