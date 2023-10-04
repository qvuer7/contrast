from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms as transforms_v1

from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np


def get_dictionary(path='/home/andrii/sup_contrast/data/welding_defects_datasets/12_classes'):
    classes = os.listdir(path)
    data_dict = {}
    clas2idx = {clas: idx for idx, clas in enumerate(classes)}
    idx2clas = {idx: clas for idx, clas in enumerate(classes)}
    for clas in classes:
        clas_path = os.path.join(path, clas)
        images = os.listdir(clas_path)
        for image in images:
            image_path = os.path.join(clas_path, image)
            data_dict[image_path] = clas2idx[clas]
    return data_dict, clas2idx, idx2clas

def split_dict(data_dict, train_ratio, test_ratio, val_ratio):
    train_dict = {}
    test_dict = {}
    val_dict = {}
    for i in set(data_dict.values()):
        r = np.where(np.array(list(data_dict.values()))  == i)
        tr = round(len(r[0]) * train_ratio)
        ts = round(len(r[0]) * test_ratio)
        vs = round(len(r[0]) * val_ratio)
        for j in r[0][:tr]:
            train_dict[list(data_dict.keys())[j]] = i
        for j in r[0][tr:tr + ts]:
            test_dict[list(data_dict.keys())[j]] = i
        for j in r[0][tr + ts:]:
            val_dict[list(data_dict.keys())[j]] = i
    return train_dict, test_dict, val_dict


class contrastiveClassDataset(Dataset):
    def __init__(self, data_dict, transforms=None):
        self.data_dict = data_dict
        self.data_list = list(self.data_dict.items())
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, target = self.data_list[idx]
        image = Image.open(image_path)

        if image.size[0] == 4080:
            transposed_image = np.transpose(image, (1, 0, 2))
            image = Image.fromarray(transposed_image)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_data_loaders(train_r = 0.7, test_r = 0.2, val_r = 0.1, path = None):
    data_dict, clas2idx, idx2clas = get_dictionary(path = path )
    for path, clas in data_dict.items():
        print(f'{path}  :   {idx2clas[clas]}')
    train_data, test_data, val_data = split_dict(data_dict, train_r, test_r, val_r)
    train_transforms = transforms_v2.Compose(
        [
            transforms_v1.Resize((768, 768), transforms_v1.InterpolationMode.BILINEAR),
            transforms_v1.ColorJitter(contrast=0.5),
            transforms_v1.Grayscale(num_output_channels=3),
            transforms_v1.GaussianBlur(kernel_size=5),
            transforms_v1.RandomHorizontalFlip(p=0.5),
            transforms_v1.RandomVerticalFlip(p=0.5),
            transforms_v1.RandomAffine(degrees=40, shear=(0.5, 0.5)),
            transforms_v1.ToTensor()
        ]
    )

    test_transforms = transforms_v2.Compose(
        [
            transforms_v1.Resize((768, 768), transforms_v1.InterpolationMode.BILINEAR),
            transforms_v1.ToTensor(),
        ])

    train_dataset = contrastiveClassDataset(train_data, TwoCropTransform(train_transforms))
    test_dataset = contrastiveClassDataset(test_data, TwoCropTransform(test_transforms))
    val_dataset = contrastiveClassDataset(val_data, TwoCropTransform(test_transforms))

    train_batch_size = 3
    test_batch_size = 1
    val_batch_size = 1

    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, test_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=True)

    return train_loader, test_loader, val_loader



