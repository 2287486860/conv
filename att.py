import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import mobilenet_v2
from torchinfo import summary


class SelfAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, h * w)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(b, -1, h * w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out


class MobileNetSelfAttn(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetSelfAttn, self).__init__()
        # 定义模型的特征提取部分
        self.features = nn.Sequential(
            # 第一个卷积层，输入通道数为3，输出通道数为32，卷积核大小为3x3，步长为2，填充为1
            nn.Conv2d(3, int(32 * width_mult), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * width_mult)),
            nn.ReLU(inplace=True),
            # 深度可分离卷积，输入通道数为32，输出通道数为64，卷积核大小为3x3，步长为1，分组数为输入通道数，
            # 填充为1，减少模型参数量
            nn.Conv2d(int(32 * width_mult), int(64 * width_mult), kernel_size=3, stride=1, padding=1,
                      groups=int(32 * width_mult), bias=False),
            nn.BatchNorm2d(int(64 * width_mult)),
            nn.ReLU(inplace=True),
            # 自注意力机制，用于增强特征的重要性，提高模型性能
            SelfAttention(int(64 * width_mult)),
            # 深度可分离卷积，输入通道数为64，输出通道数为128，卷积核大小为3x3，步长为2，分组数为输入通道数，
            # 填充为1
            nn.Conv2d(int(64 * width_mult), int(128 * width_mult), kernel_size=3, stride=2, padding=1,
                      groups=int(64 * width_mult), bias=False),
            nn.BatchNorm2d(int(128 * width_mult)),
            nn.ReLU(inplace=True),
            SelfAttention(int(128 * width_mult)),
            nn.Conv2d(int(128 * width_mult), int(128 * width_mult), kernel_size=3, stride=1, padding=1,
                      groups=int(128 * width_mult), bias=False),
            nn.BatchNorm2d(int(128 * width_mult)),
            nn.ReLU(inplace=True),
            SelfAttention(int(128 * width_mult)),
            nn.Conv2d(int(128 * width_mult), int(256 * width_mult), kernel_size=3, stride=2, padding=1,
                      groups=int(128 * width_mult), bias=False),
            nn.BatchNorm2d(int(256 * width_mult)),
            nn.ReLU(inplace=True),
            SelfAttention(int(256 * width_mult)),
            nn.Conv2d(int(256 * width_mult), int(256 * width_mult), kernel_size=3, stride=1, padding=1,
                      groups=int(256 * width_mult), bias=False),
            nn.BatchNorm2d(int(256 * width_mult)),
            nn.ReLU(inplace=True),
            SelfAttention(int(256 * width_mult)),
            nn.Conv2d(int(256 * width_mult), int(512 * width_mult), kernel_size=3, stride=2, padding=1,
                      groups=int(256 * width_mult), bias=False),
            nn.BatchNorm2d(int(512 * width_mult)),
            nn.ReLU(inplace=True),
            SelfAttention(int(512 * width_mult)),
            nn.Conv2d(int(512 * width_mult), int(512 * width_mult), kernel_size=3, stride=1, padding=1,
                      groups=int(512 * width_mult), bias=False),
            nn.BatchNorm2d(int(512 * width_mult)),
            nn.ReLU(inplace=True),
            SelfAttention(int(512 * width_mult)),
            nn.Conv2d(int(512 * width_mult), int(512 * width_mult), kernel_size=3, stride=1, padding=1,
                      groups=int(512 * width_mult), bias=False),
            nn.BatchNorm2d(int(512 * width_mult)),
            nn.ReLU(inplace=True),
            SelfAttention(int(512 * width_mult)),
            nn.Conv2d(int(512 * width_mult), int(1024 * width_mult), kernel_size=3, stride=2, padding=1,
                      groups=int(512 * width_mult), bias=False),
            nn.BatchNorm2d(int(1024 * width_mult)),
            nn.ReLU(inplace=True),
            SelfAttention(int(1024 * width_mult)),
            nn.Conv2d(int(1024 * width_mult), int(1024 * width_mult), kernel_size=3, stride=1, padding=1,
                      groups=int(1024 * width_mult), bias=False),
            nn.BatchNorm2d(int(1024 * width_mult)),
            nn.ReLU(inplace=True),
            SelfAttention(int(1024 * width_mult)),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(int(192 * width_mult), num_classes)

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MultiResMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiResMLP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = torch.cat((x1, x2, x3), dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class MobileNetSEMultiResMLP(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetSEMultiResMLP, self).__init__()
        self.mobilenet = MobileNetSelfAttn(num_classes=num_classes, width_mult=width_mult)
        self.multi_res_mlp = MultiResMLP(int(1024 * width_mult), 64)

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.multi_res_mlp(x)
        x = x.view(x.size(0), -1)
        x = self.mobilenet.classifier(x)
        return x
model=MobileNetSEMultiResMLP(num_classes=10)
x = torch.randn(128, 3, 32, 32)
y = model(x)
print(y.shape)
summary(model,(128,3,32,32),depth=5)