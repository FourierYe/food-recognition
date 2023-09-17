import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import models
from senet import *
# helpers

class SpatialAttention(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=C,out_channels=1,kernel_size=1, stride=1)
        self.conv13_1 = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=13, padding=6, stride=1)
        self.conv13_2 = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=13, padding=6, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv13_1(x))
        x = self.sigmoid(self.conv13_2(x))
        return x

class ChannelAttention(nn.Module):
    def __init__(self, C, r):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv1_1 = nn.Conv2d(in_channels=C, out_channels=C//r, kernel_size=1, stride=1)
        self.conv1_2 = nn.Conv2d(in_channels=C//r, out_channels=C, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.gap(x)
        x = self.relu(self.conv1_1(x))
        x = self.sigmoid(self.conv1_2(x))
        return x

class PixelAttention(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.spatial_attention = SpatialAttention(C)
        self.channel_attention = ChannelAttention(C, 16)

    def forward(self, x):
        ms = self.spatial_attention(x)
        mc = self.channel_attention(x)
        mp = torch.mul(ms, mc)
        return torch.mul(mp, x)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MyModel(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.backbone = senet154(num_classes=1000, pretrained=None)
        self.layer0 = self.backbone.layer0
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.layer1_attention = PixelAttention(C=256)
        self.layer2_attention = PixelAttention(C=512)
        self.layer3_attention = PixelAttention(C=1024)

        self.fc = nn.Linear(in_features=3840, out_features=500, bias=True)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        global_feature = self.transformer(x4.view(-1, 2048, 49))
        global_feature = global_feature.mean(dim=2)

        x1_after_attention = self.layer1_attention(x1) + x1
        x2_after_attention = self.layer2_attention(x2) + x2
        x3_after_attention = self.layer3_attention(x3) + x3

        x1_feature = self.gap(x1_after_attention)
        x2_feature = self.gap(x2_after_attention)
        x3_feature = self.gap(x3_after_attention)

        local_feature = torch.cat((x1_feature, x2_feature,x3_feature), dim=1).squeeze()
        all_features = torch.cat((local_feature, global_feature), dim=1)

        out = self.fc(all_features)
        return out

if __name__ == "__main__":

    model = MyModel(num_classes=1000, dim=49, depth=2, heads=2, mlp_dim=768, pool='mean', dim_head=64)
    X = torch.randn(2, 4, 224, 224)
    X = model(X)
    print(X.shape)