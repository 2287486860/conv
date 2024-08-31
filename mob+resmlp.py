import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MLPBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)

class CustomModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        mobilenet = list(mobilenet.children())[0]  # 保留 MobileNetV2 的特征提取部分
        self.mobilenet_head = nn.Sequential(*mobilenet[:7])

        self.mlp_block = MLPBlock(16, 64, 24)

        self.mobilenet_tail = nn.Sequential(*mobilenet[7:])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.mobilenet_head(x)

        n, c, h, w = x.size()
        x = x.view(n, c, h * w)
        x = self.mlp_block(x)
        x = x.view(n, -1, h, w)

        x = self.mobilenet_tail(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = CustomModel()
x = torch.randn(128, 3, 32, 32)
output = model(x)
print(output.size())
