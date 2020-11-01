import os
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from PIL import Image


def main():
    model = torch.load("checkpoint/model_epoch_0.pth")['model']
    model.eval()

    validation_img(model)


def validation_img(net, img_path='SRDB/DIV2K_valid_LR_bicubic/X2/0801x2.png'):
    img = Image.open(img_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
    img = transform(img)
    img = img.unsqueeze(0)
    output = net(img)
    out_img = transforms.ToPILImage()(output[0].data.cpu())
    if not os.path.exists("result/"):
        os.makedirs("result/")
    out_img.save("result/output.jpg")


if __name__ == '__main__':
    main()
