import torch
import torch.nn as nn
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

    # C1表示输入通道为1，输出通道为6，核大小为5*5的 2D 卷积操作；
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
    # R1是ReLU非线性激活函数；
        self.R1 = nn.ReLU()
    # S2是max-pooling 操作；
        self.S2 = nn.MaxPool2d(kernel_size=2)
    # C3, R2, S4 分别是新一层的 2D 卷积和 ReLU 激活函数以及 Max-Pooling 操作，
        self.C3 = nn.Conv2d(6, 16, 5, 1, 0)
        self.R2 = nn.ReLU()
        self.S4 = nn.MaxPool2d(2)
        self.C5 = nn.Conv2d(16, 120, 5, 1, 0)
        self.R3 = nn.ReLU()
    # F6 是全连接层，OUT 输出层。
        self.F6 = nn.Linear(in_features=120, out_features=84)
        self.R4 = nn.ReLU()
        self.OUT = nn.Linear(84, 10)

    def forward(self, x):
    # 前向计算方法
        x = self.C1(x)  # 先经过卷积层C1处理x
        x = self.R1(x)  # 经过ReLu非线性激活函数R1
        x = self.S2(x)  # 然后进行max-pooling处理S2
        x = self.C3(x)  # 再通过卷积层C3
        x = self.R2(x)  # 经过ReLu非线性激活函数R2
        x = self.S4(x)  # 进行max-pooling处理S4
        x = self.C5(x)  # 输入卷积层C5
        x = self.R3(x)  # 进行ReLu非线性激活函数R3
        x = x.view(x.size(0), -1)  #  reshape 成 (batch_size, -1)
        x = self.F6(x)  # 输出全连接层F6
        x = self.R4(x)  # 再次进行ReLu非线性激活函数R4
        x = self.OUT(x) # 输出层
        return x
if __name__ == "__main__":
# 定义一个LeNet实例model
    model = LeNet()
# 随机生成输入张量a，大小为（[1, 1, 28, 28]）
    a = torch.randn(1, 1, 28, 28)
# 对模型传入上述张量，计算输出张量b
    b = model(a)
# 打印输出张量b
    print(b)