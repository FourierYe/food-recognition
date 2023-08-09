import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from model import load_model
from data import load_dataset
import os
import argparse
import torchmetrics
import torch
from torchvision import transforms

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

    if args.multi_gpu:
        if torch.cuda.is_available():
            top1acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=500, top_k=1).cuda(device=device_ids[0])
            top5acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=500, top_k=5).cuda(device=device_ids[0])
    else:
        top1acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=500, top_k=1).to(device)
        top5acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=500, top_k=5).to(device)

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

        # top1 acc and top5 acc
        top1acc = top1acc_metric(logits, target)
        top5acc = top5acc_metric(logits, target)

        if batch_index % 50 == 0:
            trained_samples = batch_index * BATCH_SIZE + len(images)
            print(f"Training Epoch: {epoch}, [{trained_samples}/{total_sample}]\t "
                  f"Loss:{loss.item():.4f}, \t top1acc:{top1acc.item():.4f},\t top5acc:{top5acc.item():.4f}")
            writer.add_scalar("loss", loss.item(), batch_index + epoch * len(train_loader))
            writer.add_scalar("training one batch top1acc", top1acc, batch_index + epoch * len(train_loader))
            writer.add_scalar("training one batch top5acc", top5acc, batch_index + epoch * len(train_loader))

    top1acc = top1acc_metric.compute()
    top5acc = top5acc_metric.compute()
    print(f"Training Epoch: {epoch}, training top1acc: {top1acc:.4f}, "
          f"training top5acc:{top5acc:.4f}")
    writer.add_scalar("training one epoch top1acc", top1acc, epoch)
    writer.add_scalar("training one epoch top5acc", top5acc, epoch)

    top1acc_metric.reset()
    top5acc_metric.reset()

# eval model
@torch.no_grad()
def eval_test_dataset(epoch, loader):
    print('Evaluating Network.....')
    start = time.time()
    model.eval()
    if args.multi_gpu:
        if torch.cuda.is_available():
            top1acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=500, top_k=1).cuda(device=device_ids[0])
            top5acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=500, top_k=5).cuda(device=device_ids[0])
    else:
        top1acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=500, top_k=1).to(device)
        top5acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=500, top_k=5).to(device)

    for imgs, target in loader:
        if args.multi_gpu:
            if torch.cuda.is_available():
                imgs = imgs.cuda(device=device_ids[0])
                target = target.cuda(device=device_ids[0])
        else:
            imgs, target = imgs.to(device), target.to(device)
        logits = model(imgs)

        loss = criterion(logits, target)

        # top1 accuracy and top5 accuracy
        top1acc = top1acc_metric(logits, target)
        top5acc = top5acc_metric(logits, target)

    top1acc = top1acc_metric.compute()
    top5acc = top5acc_metric.compute()
    finish = time.time()
    print(f"Evaluating Test dataset: Epoch: {epoch}, top1: {top1acc}, top5: {top5acc}, time consumed:{finish - start:.2f}s")
    writer.add_scalar("test loss", loss.item(), epoch)
    writer.add_scalar("test top1acc", top1acc, epoch)
    writer.add_scalar("test top5acc", top5acc, epoch)

    top1acc_metric.reset()
    top5acc_metric.reset()

    return top1acc, top5acc


# parse parameters
def parse_args():
    parser = argparse.ArgumentParser('parameters for training model')
    parser.add_argument('--model', type=str, default='senet154')
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0,1])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--step_lr', type=int, default=10)
    parser.add_argument('--train_dataset_path', type=str, default="Food500_train_path")
    parser.add_argument('--test_dataset_path', type=str, default="Food500_test_path")
    parser.add_argument('--focal_loss', type=bool, default=False)
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
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path

    train_dataset, test_dataset = load_dataset(train_dataset_path, test_dataset_path, BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=2)

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
    best_top5_acc = 0
    for epoch in range(EPOCH):
        # train model one epoch
        train_model(epoch)
        # Evaluation on test dataset
        top1_acc, top5_acc = eval_test_dataset(epoch, test_loader)
        # save best check point
        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            best_top5_acc = top5_acc
            torch.save(model.state_dict(), f'{model_name}_pretrained.pth')

    print(f"best top1 acc {best_top1_acc}; best top 5 acc {best_top5_acc}")
