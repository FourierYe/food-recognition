import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch.nn as nn
from torchvision import models
import torch
import pretrainedmodels
import math
import torch.nn.functional as F
from vit_pytorch.vit import ViT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from volo import *


def load_model(model):
    if model == "senet154":
        return SeNet154()
    elif model == 'vit':
        return ViT(image_size=224, patch_size=16, num_classes=500, dim=768, depth=12, heads=12, mlp_dim=3072)
    elif model == "senet_vit":
        return SenetVit()
    elif model == "volo":
        return Volo(num_classes=500)
    elif model == "senet_volo":
        return SeNetVol()
    elif model == 'swin':
        model = models.swin_b()
        model.head = nn.Linear(in_features=1024, out_features=500, bias=True)
        return model


class Volo(nn.Module):
    def __init__(self, **kwargs):
        super(Volo, self).__init__()
        self.model = volo_d5(**kwargs)

    def forward(self, x):
        if self.training:
            x = self.model(x)[0]
        else:
            x = self.model(x)
        return x


# baseline model
class SeNet154(nn.Module):
    def __init__(self):
        super(SeNet154, self).__init__()
        self.model = self._load_model()

    def _load_model(self):
        model_name = 'senet154'
        # could be fbresnet152 or inceptionresnetv2
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model.eval()
        model.last_linear = nn.Linear(2048, 500, bias=True)
        return model

    def forward(self, x):
        x = self.model(x)
        return x


class SenetVit(nn.Module):

    def __init__(self):
        super(SenetVit, self).__init__()
        self.senet154 = self._load_senet154()
        self.vit = self._load_vit()
        self.fc = nn.Linear(1000, 500)

    def _load_senet154(self):
        model_name = 'senet154'

        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model.eval()
        model.last_linear = nn.Linear(2048, 500, bias=True)
        return model

    def _load_vit(self):
        return ViT(image_size=224, patch_size=16, num_classes=500, dim=768, depth=12, heads=12, mlp_dim=3072)

    def forward(self, x):
        # gobal features
        x1 = self.senet154(x)
        # local features
        x2 = self.vit(x)

        x = torch.cat([x1, x2], 1)
        x = self.fc(x)
        return x


class SeNetVol(nn.Module):

    def __init__(self):
        super(SeNetVol, self).__init__()
        self.senet154 = self._load_senet154()

    def _load_senet154(self):
        model_name = 'senet154'

        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model.eval()
        model.last_linear = nn.Linear(2048, 500, bias=True)
        return model

    def forward(self, x):
        # gobal features
        x1 = self.senet154(x)
        # local features
        x2 = self.resnet(x)

        x = torch.cat([x1, x2], 1)
        x = self.fc(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


if __name__ == '__main__':
    # C = 10
    # X = torch.randn(1, 3, 10, 10)
    # model = OutlookAttention(dim=C, num_heads=1)
    # X = model(X)
    # print(X.shape)

    model = load_model('swin')
    print(model)
    X = torch.randn(20, 3, 224, 224)
    X = model(X)
    print(X)