# models/DeepLab_V3_Plus_Effi_USE_Trans2.py
import torch
import torch.nn as nn
from backbones.efficientnet import EfficientNetB5
from models.modules.aspp import build_aspp
from models.modules.decoder import build_decoder
from models.modules.sync_batchnorm import SynchronizedBatchNorm2d

class DeepLab_V3_Plus_Effi_USE_Trans2(nn.Module):
    def __init__(self, num_classes=6, backbone='efficientnet_b5', output_stride=16, sync_bn=True, freeze_bn=False):
        super(DeepLab_V3_Plus_Effi_USE_Trans2, self).__init__()
        BatchNorm = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d

        self.backbone = EfficientNetB5(in_channels=3, output_stride=output_stride, BatchNorm=BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, skip_feats = self.backbone(input)  # x = p6, skip_feats = [p2, p3, p4, p5]
        x = self.aspp(x)
        x = self.decoder(x, skip_feats)
        x = nn.functional.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        for m in self.backbone.modules():
            for p in m.parameters():
                if p.requires_grad:
                    yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for module in modules:
            for m in module.modules():
                for p in m.parameters():
                    if p.requires_grad:
                        yield p

def build_deeplab(num_classes=6, backbone='efficientnet_b5', output_stride=16, sync_bn=True, freeze_bn=False):
    return DeepLab_V3_Plus_Effi_USE_Trans2(num_classes, backbone, output_stride, sync_bn, freeze_bn)