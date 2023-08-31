import torch
import numpy as np
import PIL
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0. # beta分布超参数
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch, target):
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        # 建立one-hot标签
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        # 判断是否进行mixup
        if torch.rand(1).item() >= self.p:
            return batch, target

        # 这里将batch数据平移一个单位，产生mixup的图像对，这意味着每个图像与相邻的下一个图像进行mixup
        # timm实现是通过flip来做的，这意味着第一个图像和最后一个图像进行mixup
        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # 随机生成组合系数
        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)  # 得到mixup后的图像

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)  # 得到mixup后的标签

        return batch, target

def load_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    N = 256
    train_transforms = transforms.Compose([
         transforms.RandomHorizontalFlip(p=0.5),  # default value is 0.5
         transforms.Resize((N, N)),
         transforms.RandomCrop((224, 224)),
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
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

def My_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, labels_file_path, images_file_path, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(labels_file_path, 'r')
        imgs = []

        for line in data_txt:
            line = line.strip()
            words = line.split()

            imgs.append((words[0], int(words[1])))

        self.images_file_path = images_file_path
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]

        try:
            img = self.loader(os.path.join(self.images_file_path, img_name))
            if self.transform is not None:
                img = self.transform(img)
        except:
            img = np.zeros((256, 256, 3), dtype=float)
            img = Image.fromarray(np.uint8(img))
            if self.transform is not None:
                img = self.transform(img)
            print('erro picture:', img_name)
        return img, label

def load_dataset(train_dataset_path, test_dataset_path, batch_size):

    TRAIN_LABELS_FILE = train_dataset_path

    TEST_LABELS_FILE = test_dataset_path
    IMAGES_FILE_PATH = 'data/isiafood500/ISIA_Food500/images'

    train_transforms, test_transforms = load_transforms()

    train_dataset = MyDataset(TRAIN_LABELS_FILE, IMAGES_FILE_PATH, transform=train_transforms)
    # val_dataset = MyDataset(VAL_LABELS_FILE, IMAGES_FILE_PATH, transform=train_transforms)
    test_dataset = MyDataset(TEST_LABELS_FILE, IMAGES_FILE_PATH, transform=test_transforms)

    # return train_dataset, val_dataset, test_dataset
    return train_dataset, test_dataset


def load_class_weight():
    total_num = 0
    label_num = {}
    with open("data/isiafood500/metadata_ISIAFood_500/train_full.txt") as file:
        for line in file:
            cols = line.split()
            label = int(cols[-1])
            if label in label_num:
                label_num[label] = label_num[label] + 1
            else:
                label_num[label] = 1
            total_num += 1

    label_num = dict(sorted(label_num.items(), key=lambda d: d[0]))

    imgs_num = 0
    for key in label_num:
        imgs_num += label_num[key]

    class_num = len(label_num)
    average_weight = imgs_num / class_num

    label_weight = []
    for key in label_num:
        label_weight.append(average_weight / label_num[key])

    return torch.tensor(label_weight)

if __name__ == "__main__":
    # train_loader, val_loader, test_loader = load_dataloader(80)
    # train_loader, test_loader = load_dataset(80)
    # print(test_loader)
    # for imgs, lables in train_loader:
    #     print(lables)
    label_weight = load_class_weight()
    print(label_weight)