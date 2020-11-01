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


if __name__ == '__main__':
    net = SRresnet()
    img = torch.Tensor(16, 3, 64, 64)

    output = net(img)
    print(net)
