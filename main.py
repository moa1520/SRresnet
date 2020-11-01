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
    customImageLoader = DataLoader(dataset, batch_size=64, shuffle=True)

    epochs = 1

    # Criterion, Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    if cuda:
        net.cuda()
        criterion.cuda()

    Tensor = torch.cuda.FloatTenor if cuda else torch.Tensor

    for epoch in range(epochs):
        for i, imgs in enumerate(customImageLoader):
            if i != 0:
                break
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
        save_checkpoint(net, epoch)


def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("checkpoint/"):
        os.mkdir("checkpoint/")

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    train()
