# models/modules/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.sync_batchnorm import SynchronizedBatchNorm2d

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid_channel(avg_out + max_out)
        x = x * scale
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid_spatial(self.conv_after_concat(concat))
        return x * scale

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm(out_channels)
        self.se = SEBlock(out_channels)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.residual_bn = BatchNorm(out_channels)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.residual_conv is not None:
            residual = self.residual_bn(self.residual_conv(residual))
        out += residual
        return self.relu(out)

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'efficientnet_b5':
            low_level_inplanes = [24, 40, 64, 176]
            bottleneck_channels = 256
        elif backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.upconv4 = nn.ConvTranspose2d(bottleneck_channels, 256, kernel_size=2, stride=2)
        self.cbam5 = CBAM(low_level_inplanes[3])
        self.conv4 = ResidualConvBlock(256 + low_level_inplanes[3], 256, BatchNorm)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.cbam4 = CBAM(low_level_inplanes[2])
        self.conv3 = ResidualConvBlock(128 + low_level_inplanes[2], 128, BatchNorm)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.cbam3 = CBAM(low_level_inplanes[1])
        self.conv2 = ResidualConvBlock(64 + low_level_inplanes[1], 64, BatchNorm)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.cbam2 = CBAM(low_level_inplanes[0])
        self.conv1 = ResidualConvBlock(32 + low_level_inplanes[0], 32, BatchNorm)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x, skip_feats):
        x = self.upconv4(x)
        x = torch.cat((x, self.cbam5(skip_feats[3])), dim=1)
        x = self.conv4(x)

        x = self.upconv3(x)
        x = torch.cat((x, self.cbam4(skip_feats[2])), dim=1)
        x = self.conv3(x)

        x = self.upconv2(x)
        x = torch.cat((x, self.cbam3(skip_feats[1])), dim=1)
        x = self.conv2(x)

        x = self.upconv1(x)
        x = torch.cat((x, self.cbam2(skip_feats[0])), dim=1)
        x = self.conv1(x)

        return self.final_conv(x)

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
