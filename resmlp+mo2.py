import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MLPBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.mlp_block = MLPBlock(in_channels, hidden_channels, out_channels, kernel_size, stride, padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        residual = x
        x = self.mlp_block(x)
        x += self.conv(residual)
        return x

class ResMLP(nn.Module):
    def __init__(self, image_size=32, patch_size=16, num_classes=10, hidden_dim=384, depth=12):
        super(ResMLP, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2
        self.hidden_dim = hidden_dim

        # Patch embedding
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim // 2, hidden_dim) for _ in range(depth)
        ])

        # Classifier
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)

        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embedding
        x = x + self.pos_embedding

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Classifier
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

model = ResMLP()

x = torch.randn(128, 3, 32, 32)
output = model(x)
print(output.shape)
