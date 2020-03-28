import torch
from torch import nn
from torch.nn import functional as F


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                 init_stride=(1, 1, 1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True), )

        # initialise the blocks
        for m in self.children():
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
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))
        else:
            self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
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

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
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
        outputs = self.conv1(inputs)
        return outputs


class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
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

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)
