import torch
import torch.nn as nn
import torchvision.models as models


class ChannelGate(nn.Module):
    # channel attention: tells the model which feature channels are important
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        avg_out = self.mlp(x.mean(dim=[2, 3]))
        max_out = self.mlp(x.amax(dim=[2, 3]))
        scale = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        return x * scale


class SpatialGate(nn.Module):
    # spatial attention: tells the model where to focus in the image
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        spatial = torch.cat([avg_map, max_map], dim=1)
        scale = torch.sigmoid(self.bn(self.conv(spatial)))
        return x * scale


class CBAM(nn.Module):
    # applies channel attention then spatial attention
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel = ChannelGate(in_channels, reduction)
        self.spatial = SpatialGate()

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class BioLiteNet(nn.Module):
    def __init__(self, num_classes=23, dropout_rate=0.4, use_cbam=True,
                 cbam_stages=None, cbam_reduction=16, pretrained=True):
        super().__init__()

        if cbam_stages is None:
            cbam_stages = [3, 6, 12]

        self.use_cbam = use_cbam
        self.cbam_stages = set(cbam_stages)

        # load pretrained MobileNetV3-Large as backbone
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)

        # split backbone into individual stages so we can insert CBAM between them
        self.stages = nn.ModuleList()
        self.cbams  = nn.ModuleDict()

        for i, stage in enumerate(backbone.features.children()):
            self.stages.append(stage)
            if use_cbam and i in self.cbam_stages:
                out_ch = self._get_out_channels(stage)
                if out_ch is not None:
                    self.cbams[str(i)] = CBAM(out_ch, reduction=cbam_reduction)

        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(960, num_classes)

    @staticmethod
    def _get_out_channels(module):
        # find the last conv layer to get the output channel count
        last_conv = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            return last_conv.out_channels
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                return m.num_features
        return None

    def forward(self, x):
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.use_cbam and str(i) in self.cbams:
                x = self.cbams[str(i)](x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cbam_p    = sum(p.numel() for p in self.cbams.parameters())
        return {
            "total_M":     round(total / 1e6, 2),
            "trainable_M": round(trainable / 1e6, 2),
            "cbam_M":      round(cbam_p / 1e6, 3),
        }


if __name__ == "__main__":
    model = BioLiteNet(num_classes=23, use_cbam=True)
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print("Output shape:", out.shape)
    print("Parameters:", model.count_parameters())
