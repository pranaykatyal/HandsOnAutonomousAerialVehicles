
import torch
import torch.nn as nn
import torch.nn.functional as F

# let's implement resnet34 as backbone, U-net as architecture, we mixed and matched
# https://raw.githubusercontent.com/rasbt/deeplearning-models/18e046926551378cd691fd871dda0f21dcd272ab/pytorch_ipynb/images/resnets/resnet152/resnet152-arch-1.png


class Network(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Network, self).__init__()

        self.debug = False

        # first conv goes from 3 rgb channels to 64 feature channels, then does it again
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=9,
                      stride=1, padding=4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=7,
                      stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )
        # self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.down1 = DownModule(64, 128)
        self.down2 = DownModule(128, 256)
        self.down3 = DownModule(256, 512)
        self.down4 = DownModule(512, 1024)

        self.up1 = UpModule(1024, 512)
        self.up2 = UpModule(512, 256)
        self.up3 = UpModule(256, 128)
        self.up4 = UpModule(128, 64)

        self.outConv = nn.Conv2d(
            64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # we simply pass the x through each layer
        x_1 = self.conv(x)
        # x = self.maxPool(x)
        if self.debug:
            print("X1", x_1.shape)
        x_2 = self.down1(x_1)
        if self.debug:
            print("X2", x_2.shape)
        x_3 = self.down2(x_2)
        if self.debug:
            print("X3", x_3.shape)
        x_4 = self.down3(x_3)
        if self.debug:
            print("X4", x_4.shape)
        x_5 = self.down4(x_4)
        if self.debug:
            print("X5", x_5.shape)

        # we build it back up by upscaling and concatenating the opposite layers
        x = self.up1(x_5, x_4)
        if self.debug:
            print("x_0_up", x.shape)
        x = self.up2(x, x_3)
        if self.debug:
            print("x_1_up", x.shape)
        x = self.up3(x, x_2)
        if self.debug:
            print("x_2_up", x.shape)
        x = self.up4(x, x_1)
        if self.debug:
            print("x_3_up", x.shape)

        x = self.outConv(x)
        if self.debug:
            print("out", x.shape)

        return x


class DownModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # this decreases the size of the image by half
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # just perform the standard Conv, BN, ReLU twice without changing the image size
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.maxPool(x)
        x = self.conv(x)
        return x


class UpModule(nn.Module):
    # example is 1024, 512
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # this increases the size of the image by 2
        # channels will be halved 1024 -> 512
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2,
                                     stride=2, padding=0)

        # we perform the standard Conv, BN, ReLU twice without changing the image size
        # since we perform the concatenation with the opposite layer the channels will be in_ch on the first conv2d (512+512) -> 512
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_opposite):
        x = self.up(x)
        # paste the x from the previous layer
        x = torch.cat([x_opposite, x], dim=1)
        x = self.conv(x)
        return x
