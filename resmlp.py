import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from torchinfo import summary
# 定义一个仿射变换层
class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 定义可学习的缩放和平移参数
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        # 对输入张量进行缩放和平移操作
        x = x * self.alpha + self.beta
        return x

# 定义一个前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()

        # 定义前馈神经网络的结构
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 定义一个 MLP 块
class MLPblock(nn.Module):

    def __init__(self, dim, num_patch, mlp_dim, dropout = 0., init_values=1e-4):
        super().__init__()

        # 定义 MLP 块的结构
        self.pre_affine = Aff(dim)
        self.token_mix = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patch, num_patch),
            Rearrange('b d n -> b n d'),
        )
        self.ff = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
        )
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # 对输入张量进行非线性变换
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x

# 定义 ResMLP 模型
class ResMLP(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, mlp_dim):
        super().__init__()

        # 检查输入图像的大小是否能够被补丁大小整除
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算图像中补丁的数量
        self.num_patch =  (image_size// patch_size) ** 2

        # 定义将输入图像转换为补丁嵌入张量的卷积层和 Rearrange 函数
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        # 创建多个 MLP 块，并将它们添加到 nn.ModuleList 中
        self.mlp_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(dim, self.num_patch, mlp_dim))

        # 定义一个仿射变换层和一个 MLP 头
        self.affine = Aff(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # 将输入图像转换为补丁嵌入张量
        x = self.to_patch_embedding(x)

        # 对输入张量进行多个 MLP 块的非线性变换
        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)

        # 对输出张量进行缩放和平移操作
        x = self.affine(x)

        # 对输出张量的第二个维度进行平均池化，并将其输入到 MLP 头中进行分类
        x = x.mean(dim=1)

        return self.mlp_head(x)
model = ResMLP(in_channels=3, image_size=32, patch_size=16, num_classes=38,
                     dim=384, depth=12, mlp_dim=384*4)
# 测试代码
if __name__ == "__main__":
    img = torch.ones([128, 3, 32, 32])

    # 创建 ResMLP 模型
    model = ResMLP(in_channels=3, image_size=32, patch_size=16, num_classes=38,
                     dim=384, depth=12, mlp_dim=384*4)
    summary(model, (1, 3, 32, 32))
    # 计算模型中可训练参数的数量
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    # 将输入张量输入到 ResMLP 模型中，并打印输出张量的形状
    # out_img = model(img)
    # print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]