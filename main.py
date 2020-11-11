import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from datasets import CustomDataset
from model import Discriminator, SRresnet


def train():
    cuda = torch.cuda.is_available()

    batch_size = 32

    # Import Model
    net = SRresnet()
    net.train()
    discriminator = Discriminator()

    # DataLoader
    dataset = CustomDataset()
    customImageLoader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    epochs = 10

    # Criterion, Optimizer, Scheduler
    G_optimizer = optim.Adam(net.parameters(), lr=0.001)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    criterion = nn.L1Loss()

    scheduler = optim.lr_scheduler.StepLR(
        optimizer=G_optimizer, step_size=1, gamma=0.1)

    if cuda:
        net.cuda()
        discriminator.cuda()
        criterion.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    for epoch in range(epochs):
        for i, imgs in enumerate(customImageLoader):
            hr_img = Variable(imgs['hr'], requires_grad=False).type(Tensor)
            lr_img = Variable(imgs['lr']).type(Tensor)

            # Adversarial ground truths
            valid = Variable(
                Tensor(np.ones((lr_img.size(0), 1))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((lr_img.size(0), 1))),
                            requires_grad=False)

            # Train Generator
            gen_hr = net(lr_img)
            loss = criterion(discriminator(gen_hr), valid)

            G_optimizer.zero_grad()
            loss.backward()
            G_optimizer.step()

            # Train Discriminator
            D_optimizer.zero_grad()
            loss_real = criterion(discriminator(hr_img), valid)
            loss_fake = criterion(discriminator(gen_hr.detach()), fake)

            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            D_optimizer.step()

            if i % 100 == 0:
                print("==> Epoch [{}] ({} / {}) : Loss : {:.5}".format(
                    epoch, i, len(customImageLoader), loss.item()))
                print("==> Epoch [{}] ({} / {}) : Loss_D : {:.5}".format(
                    epoch, i, len(customImageLoader), loss_D.item()))
        scheduler.step()
        # save_checkpoint(net, epoch, optimizer, criterion)


def save_checkpoint(model, epoch, optimizer, loss):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    if not os.path.exists("checkpoint/"):
        os.mkdir("checkpoint/")

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def PSNR(gen_img, label_img):
    mse = np.mean((gen_img - label_img) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr


if __name__ == '__main__':
    train()
