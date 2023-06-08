import torch
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from model import FoodClassification
import torch.nn as nn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置全局参数
modellr = 1e-4
BATCH_SIZE = 16
EPOCH = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])

# DATASET
dataset_train = datasets.ImageFolder('data/train', transform)

# BATCH LOADER
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

model = FoodClassification()
model.to(DEVICE)

# define loss
criterion = nn.CrossEntropyLoss().to(DEVICE)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=modellr)

# train model
for epoch in range(EPOCH):
    sum_loss = 0
    for batch_idx, (img, target) in enumerate(train_loader):
        img = img.to(DEVICE)
        target = target.to(DEVICE)
        pred = model(img)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(img), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
            ave_loss = sum_loss / len(train_loader)
            print('epoch:{},ave loss:{}'.format(epoch, ave_loss))

# save model
torch.save(model, "model.pt")