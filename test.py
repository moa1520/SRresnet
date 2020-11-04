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
    model.load_state_dict(torch.load(
        "checkpoint/model_epoch_9.pth")['model_state_dict'])
    model.cuda()
    model.eval()

    validation_img(model)


def validation_img(net, img_path='test_img.jpg'):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize((img.height // 4, img.width // 4), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    img = transform(img)
    if not os.path.exists('result/'):
        os.makedirs('result/')
    save_image(img, "result/output_before.jpg", normalize=True)
    img = img.unsqueeze(0)
    img = Variable(img).type(torch.cuda.FloatTensor)
    output = net(img)
    save_image(output, "result/output.jpg", normalize=True)


if __name__ == '__main__':
    main()
    # print(torch.cuda.is_available())
    # img = Image.open('test_img.jpg')
    # img = Image.open('result/output.jpg')
    # img = np.asarray(img)
    # print(img.shape)
