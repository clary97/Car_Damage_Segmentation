# models/backbones/xception.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modules.sync_batchnorm import SynchronizedBatchNorm2d

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1))  # fixed padding for kernel=3
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skipbn(self.skip(inp))
        else:
            skip = inp
        return x + skip

class AlignedXception(nn.Module):
    def __init__(self, output_stride, BatchNorm, in_channels=3, pretrained=False):
        super(AlignedXception, self).__init__()
        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        self.block1 = Block(64, 128, 2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block2 = Block(128, 256, 2, stride=2, BatchNorm=BatchNorm)
        self.block3 = Block(256, 728, 2, stride=entry_block3_stride, BatchNorm=BatchNorm, is_last=True)

        self.middle_blocks = nn.Sequential(*[
            Block(728, 728, 3, stride=1, dilation=middle_block_dilation, BatchNorm=BatchNorm)
            for _ in range(16)
        ])

        self.block20 = Block(728, 1024, 2, stride=1, dilation=exit_block_dilations[0], BatchNorm=BatchNorm, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048)

        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.middle_blocks(x)
        x = self.block20(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pass  # 구현 생략 또는 모델 허브에서 불러오기
