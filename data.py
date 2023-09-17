import torch
import numpy as np
import PIL
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

class LBPFusion(object):
    def __call__(self, img):
        height = img.shape[0]
        width = img.shape[1]

        dst = img.copy()

        lbp_value = np.zeros((1, 8), dtype=np.uint8)

        neighbors = np.zeros((1, 8), dtype=np.uint8)

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                center = img[y, x]

                neighbors[0, 0] = img[y - 1, x - 1]
                neighbors[0, 1] = img[y - 1, x]
                neighbors[0, 2] = img[y - 1, x + 1]
                neighbors[0, 3] = img[y, x - 1]
                neighbors[0, 4] = img[y, x + 1]
                neighbors[0, 5] = img[y + 1, x - 1]
                neighbors[0, 6] = img[y + 1, x]
                neighbors[0, 7] = img[y + 1, x + 1]

                for i in range(8):
                    if neighbors[0, i] > center:
                        lbp_value[0, i] = 1
                    else:
                        lbp_value[0, i] = 0

                # uint8 八位二进制数来存放0-1序列 巧妙！
                lbp = 0
                for i in range(8):
                    lbp += lbp_value[0, i] * 2 ** i

                dst[y, x] = lbp

        return dst


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
    train_loader, val_loader, test_loader = load_dataloader(80)
    train_loader, test_loader = load_dataset(80)
    print(test_loader)
    for imgs, lables in train_loader:
        print(lables)