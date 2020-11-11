import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        output = self.residual(x)
        output = torch.add(x, output)

        return output


class SRresnet(nn.Module):
    def __init__(self) -> None:
        super(SRresnet, self).__init__()

        # first layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=9 // 2),
            nn.PReLU()
        )

        # residual layer
        resi = [ResidualBlock() for _ in range(5)]

        self.layer2 = nn.Sequential(*resi)

        # third layer
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(64)
        )

        # pixel shuffle layer
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=3 // 2),
            # nn.PixelShuffle(2),
            nn.ConvTranspose2d(256, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.PReLU()
        )

        # last layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, padding=9 // 2)
        )

    def forward(self, x):
        output1 = self.layer1(x)
        output2 = self.layer2(output1)
        output2 = self.layer3(output2)

        output = torch.add(output1, output2)
        output = self.layer4(output)
        output = self.layer5(output)

        return output


class Discriminator(nn.Module):
    def __init__(self, batch_size) -> None:
        super(Discriminator, self).__init__()

        self.batch_size = batch_size

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=3 // 2),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=3 // 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=3 // 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=3 // 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=3 // 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(1)

        self.layer9 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.AdaptiveAvgPool(x)
        x = x.reshape(self.batch_size, -1)
        x = self.layer9(x)

        return x


if __name__ == '__main__':
    net = Discriminator(batch_size=32)
    img = torch.Tensor(32, 3, 64, 64)
    print("input shape :", img.shape)

    output = net(img)
    print("output shape :", output.shape)
