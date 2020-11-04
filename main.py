import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from datasets import CustomDataset
from model import SRresnet


def train():
    cuda = torch.cuda.is_available()

    # Import Model
    net = SRresnet()
    net.train()

    # DataLoader
    dataset = CustomDataset()
    customImageLoader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 10

    # Criterion, Optimizer, Scheduler
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=1, gamma=0.1)

    if cuda:
        net.cuda()
        criterion.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    for epoch in range(epochs):
        for i, imgs in enumerate(customImageLoader):
            hr_img = Variable(imgs['hr'], requires_grad=False).type(Tensor)
            lr_img = Variable(imgs['lr']).type(Tensor)

            gen_hr = net(lr_img)
            loss = criterion(gen_hr, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("==> Epoch [{}] ({} / {}) : Loss : {:.5}".format(
                    epoch, i, len(customImageLoader), loss.item()))
        scheduler.step()
        save_checkpoint(net, epoch, optimizer, criterion)


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


if __name__ == '__main__':
    train()
