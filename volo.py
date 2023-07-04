import torch
import torch.nn as nn

unfold = nn.Unfold(kernel_size=3, stride=1)
img = torch.arange(1, 26).reshape(1, 5, 5)
print(img)
print(img.shape)
features = unfold(img.to(torch.float32))
print(features)
print(features.shape)