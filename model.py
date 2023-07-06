import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch.nn as nn
from torchvision import models
import torch
import pretrainedmodels
import math
import torch.nn.functional as F
from vit_pytorch.vit import ViT

def load_model(model):
    if model == "senet154":
        return SeNet154()
    elif model == 'vit':
        return ViT(image_size=224, patch_size=16, num_classes=500, dim=768, depth=12, heads=12, mlp_dim=3072)
    elif model == "senet_volo":
        return SeNetVol()


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


class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim ** -0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v)
        v = v.reshape(B, self.num_heads, C // self.num_heads,
                      self.kernel_size * self.kernel_size,
                      h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
               self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


if __name__ == '__main__':
    # C = 10
    # X = torch.randn(1, 3, 10, 10)
    # model = OutlookAttention(dim=C, num_heads=1)
    # X = model(X)
    # print(X.shape)

    model = load_model('vit')
    print(model)
    imgs = torch.randn(1, 3, 224, 224)
    preds = model(imgs)
    print(preds)