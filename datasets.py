import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
import glob
from PIL import Image


def image_to_array(path):
    img = Image.open(path)
    img = np.array(img)
    return img


def crop_and_save(img, scale, id):
    height = img.shape[0]
    width = img.shape[1]

    w_count = width // scale
    h_count = height // scale

    imgs = []

    for j in range(h_count):
        for i in range(w_count):
            imgs.append(img[j*scale:(j+1) * scale, i*scale:(i+1) * scale, :])
    # for j in range(0, height, scale):
    #     for i in range(0, width, scale):
    #         imgs.append(img[j:j+scale, i:i+scale, :])

    for i, img in enumerate(imgs):
        img = Image.fromarray(img)
        # img.save('SRDB/HR_patch/%07d_%07d.PNG' % (id, i))
        img.save('SRDB/LR_patch/%07d_%07d.PNG' % (id, i))


def total_patch():
    # folder_path = 'SRDB/DIV2K_train_HR'
    folder_path = 'SRDB/DIV2K_train_LR_bicubic/X2'
    images = sorted(glob.glob(folder_path + '/*'))
    scale = 32

    # transform image to array, crop and save
    for i, img in enumerate(images[:50]):
        img = image_to_array(img)
        crop_and_save(img, scale, i)


class CustomDataset(Dataset):
    def __init__(self, hr_root='SRDB/HR_patch', lr_root='SRDB/LR_patch'):
        super(CustomDataset, self).__init__()

        # data load
        self.hr_images = sorted(glob.glob(hr_root + '/*'))
        self.lr_images = sorted(glob.glob(lr_root + '/*'))

        # transform
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((90, 90)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # array to image
        hr_img = Image.open(self.hr_images[index])
        lr_img = Image.open(self.lr_images[index])

        # transform
        hr_img = self.transform(hr_img)
        lr_img = self.transform(lr_img)

        return {
            'hr': hr_img,
            'lr': lr_img
        }

    def __len__(self):
        return len(self.hr_images)


if __name__ == '__main__':
    print('datasets')
