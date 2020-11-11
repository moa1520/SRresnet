from main import PSNR
from model import SRresnet
import os
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def main():
    model = SRresnet()
    # model.load_state_dict(torch.load(
    #     "checkpoint/model_epoch_9.pth")['model_state_dict'])
    # model.cuda()
    model.eval()

    validation_img(model)


def validation_img(net, img_path='test_img.jpg'):
    origin_img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize(
            (origin_img.height // 2, origin_img.width // 2), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    img = transform(origin_img)
    # if not os.path.exists('result/'):
    #     os.makedirs('result/')
    # save_image(img, "result/output_before.jpg", normalize=True)
    img = img.unsqueeze(0)
    # img = Variable(img).type(torch.cuda.FloatTensor)
    output = net(img)

    img = img.numpy()
    output = output.squeeze(0).permute(1, 2, 0).detach().numpy()

    psnr = PSNR(output, origin_img)
    print(psnr)
    # save_image(output, "result/output.jpg", normalize=True)


def foo():
    origin = Image.open('result/origin.jpg')
    origin = origin.resize((origin.width // 2 - 1, origin.height // 2 - 1))
    output = Image.open('result/output.jpg')

    output = np.asarray(output)
    origin = np.asarray(origin)

    psnr = PSNR(output, origin)
    print('psnr :', psnr)


if __name__ == '__main__':
    foo()
