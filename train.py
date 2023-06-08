import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from model import load_model
from data import load_dataset
import os
import argparse

import torch
from torchvision import transforms

def evaluteTop1(loader):

    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        if args.multi_gpu:
            if torch.cuda.is_available():
                x = x.cuda(device=device_ids[0])
                y = y.cuda(device=device_ids[0])
        else:
            x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

def evaluteTop5(loader):

    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        if args.multi_gpu:
            if torch.cuda.is_available():
                x = x.cuda(device=device_ids[0])
                y = y.cuda(device=device_ids[0])
        else:
            x, y = x.to(device), y.to(device)
        logits = model(x)
        maxk = max((1, 5))
        y_resize = y.view(-1, 1)
        _, pred = logits.topk(maxk, 1, True, True)
        correct += torch.eq(pred, y_resize).sum().float().item()
    return correct / total

def load_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    N = 256
    train_transforms = transforms.Compose([
         transforms.RandomHorizontalFlip(p=0.5),  # default value is 0.5
         transforms.Resize((N, N)),
         transforms.RandomCrop((224, 224)),
         transforms.ToTensor(),
         normalize
     ])

    test_transforms = transforms.Compose([
         transforms.Resize((N, N)),
         transforms.CenterCrop((224, 224)),
         transforms.ToTensor(),
         normalize
     ])
    return train_transforms, test_transforms

# train model
def train_model(epoch):
    print('Training Network.....')
    model.train()

    # keep track of training and validation loss
    correct_top1acc = 0
    correct_top5acc = 0
    total_sample = len(train_dataset)
    for batch_index, (images, target) in enumerate(train_loader):

        if args.multi_gpu:
            if torch.cuda.is_available():
                images = images.cuda(device=device_ids[0])
                target = target.cuda(device=device_ids[0])
        else:
            images, target = images.to(device), target.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        logits = model(images)
        # calculate the batch loss
        loss = criterion(logits, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        # top1 acc
        pred1 = logits.argmax(dim=1)
        correct_top1acc += torch.eq(pred1, target).sum().float().item()

        # top5 acc
        maxk = max((1, 5))
        y_resize = target.view(-1, 1)
        _, pred5 = logits.topk(maxk, 1, True, True)
        correct_top5acc += torch.eq(pred5, y_resize).sum().float().item()

        if batch_index % 50 == 0:
            trained_samples = batch_index * BATCH_SIZE + len(images)
            print(f"Training Epoch: {epoch + 1}, [{trained_samples}/{total_sample}]\t "
                  f"Loss:{loss.item():.4f}")
            writer.add_scalar("loss", loss.item(), batch_index + 1
                              + epoch * len(train_loader))

    train_top1acc = correct_top1acc / total_sample
    train_top5acc = correct_top5acc / total_sample
    print(f"Training Epoch: {epoch + 1}, training top1acc: {train_top1acc:.4f}, "
          f"training top5acc:{train_top5acc:.4f}")
    writer.add_scalar("training top1acc", train_top1acc, epoch)
    writer.add_scalar("training top5acc", train_top5acc, epoch)


# eval model
@torch.no_grad()
def eval_test_dataset(epoch, loader):
    print('Evaluating Network.....')
    start = time.time()
    model.eval()
    top1_acc = evaluteTop1(loader)
    top5_acc = evaluteTop5(loader)
    finish = time.time()
    print(f"Evaluating Test dataset: Epoch: {epoch}, top1: {top1_acc}, top5: {top5_acc}, time consumed:{finish - start:.2f}s")
    writer.add_scalar("test top1acc", top1_acc, epoch)
    writer.add_scalar("test top5acc", top5_acc, epoch)
    return top1_acc


# parse parameters
def parse_args():
    parser = argparse.ArgumentParser('parameters for training model')
    parser.add_argument('--model', type=str, default='senet154')
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0,1])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--step_lr', type=int, default=10)

    args, unparsed = parser.parse_known_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    # 是否使用多GPU训练
    if args.multi_gpu:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        # 可用GPU
        device_ids = args.gpus

    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    model_name = args.model

    train_dataset, test_dataset = load_dataset(BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=2)
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = load_model(model_name)

    if args.multi_gpu:
        if torch.cuda.is_available():
            # 指定要用到的设备
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            model = model.cuda(device=device_ids[0])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    step_size = args.step_lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False)

    # start training
    # init tensorboard
    writer = SummaryWriter('./log')
    best_top1_acc = 0
    for epoch in range(EPOCH):
        # train model one epoch
        train_model(epoch)
        # Evaluation on test dataset
        top1_acc = eval_test_dataset(epoch, test_loader)
        # save best check point
        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            torch.save(model.state_dict(), f'{model_name}_pretrained.pth')

