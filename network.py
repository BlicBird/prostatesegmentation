import os
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import models
import h5py


class block(torch.nn.Module):
    def __init__(self, channels, expansion=4):
        super(block, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels * expansion, kernel_size=3, padding=1, bias=False)
        self.batch1 = torch.nn.BatchNorm2d(channels * expansion)
        self.conv2 = torch.nn.Conv2d(channels * expansion, channels, kernel_size=3, padding=1, bias=False)
        self.batch2 = torch.nn.BatchNorm2d(channels)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        intial = x
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = x + intial
        final = self.relu(x)

        return final


class ResNet(torch.nn.Module):
    def __init__(self, ch_in, ch_res):
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(ch_in, ch_res, kernel_size=3, padding=1, bias=False)
        self.batch1 = torch.nn.BatchNorm2d(ch_res)
        self.res_block_1 = block(ch_res)
        self.res_block_2 = block(ch_res)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)

        return x


class UNet(torch.nn.Module):

    def __init__(self, init_feat=64):
        super(UNet, self).__init__()

        self.encoder1 = ResNet(1, init_feat)
        self.encoder2 = ResNet(init_feat, init_feat * 2)
        self.encoder3 = ResNet(init_feat * 2, init_feat * 4)
        self.encoder4 = ResNet(init_feat * 4, init_feat * 8)
        self.bottleneck = ResNet(init_feat * 8, init_feat * 16)
        self.pool_2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_conv1 = torch.nn.ConvTranspose2d(init_feat * 16, init_feat * 8, kernel_size=2, stride=2)
        self.up_conv2 = torch.nn.ConvTranspose2d(init_feat * 8, init_feat * 4, kernel_size=2, stride=2)
        self.up_conv3 = torch.nn.ConvTranspose2d(init_feat * 4, init_feat * 2, kernel_size=2, stride=2)
        self.up_conv4 = torch.nn.ConvTranspose2d(init_feat * 2, init_feat, kernel_size=2, stride=2)
        self.decoder1 = ResNet(init_feat * 16, init_feat * 8)
        self.decoder2 = ResNet(init_feat * 8, init_feat * 4)
        self.decoder3 = ResNet(init_feat * 4, init_feat * 2)
        self.decoder4 = ResNet(init_feat * 2, init_feat)
        self.decoder5 = torch.nn.Conv2d(init_feat, 1, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.encoder1(x)
        # print(x1.size())
        x2 = self.encoder2(self.pool_2x2(x1))
        # print(x2.size())
        x3 = self.encoder3(self.pool_2x2(x2))
        # print(x3.size())
        x4 = self.encoder4(self.pool_2x2(x3))
        # print(x4.size())
        x5 = self.bottleneck(self.pool_2x2(x4))
        # print(x5.size())

        # decoder
        x6 = self.up_conv1(x5)
        x6 = torch.cat((TF.resize(x6, size=x4.shape[2:]), x4), dim=1)
        x6 = self.decoder1(x6)
        # print(x6.size())
        x7 = self.up_conv2(x6)
        x7 = torch.cat((TF.resize(x7, size=x3.shape[2:]), x3), dim=1)
        x7 = self.decoder2(x7)
        # print(x7.size())
        x8 = self.up_conv3(x7)
        x8 = torch.cat((TF.resize(x8, size=x2.shape[2:]), x2), dim=1)
        x8 = self.decoder3(x8)
        # print(x8.size())
        x9 = self.up_conv4(x8)
        x9 = torch.cat((TF.resize(x9, size=x1.shape[2:]), x1), dim=1)
        x9 = self.decoder4(x9)
        # print(x9.size())

        return torch.sigmoid(self.decoder5(x9))

    






