# models/backbones/efficientnet.py
import torch
import torch.nn as nn
import timm

class EfficientNetB5(nn.Module):
    """
    EfficientNet-B5 백본을 사용하여 (x, skip_feats) 반환
    skip_feats: [p2, p3, p4, p5] - decoder에서 skip connection으로 사용됨
    """
    def __init__(self, in_channels, output_stride, BatchNorm, pretrained=True):
        super(EfficientNetB5, self).__init__()
        self.backbone = timm.create_model(
            'efficientnet_b5',
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )
        self.feat_channels = self.backbone.feature_info.channels

    def forward(self, x):
        features = self.backbone(x)
        p2 = features[0]  # stride 4
        p3 = features[1]  # stride 8
        p4 = features[2]  # stride 16
        p5 = features[3]  # stride 32
        p6 = features[4]  # stride 32 (ASPP 입력)
        return p6, [p2, p3, p4, p5]
