import torch
from torch import nn


class ConvBlock(nn.Module):
    """ Basic Convolutional Block: Conv -> BN -> Leaky/PReLU """
    def __init__(self, in_channels, out_channels, discriminator=False, use_act=True, use_bn=True, **kwargs):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
        self.act = self.act if self.use_act else nn.Identity()
        
    def forward(self, x):
        x = self.cnn(x)
        x = self.bn(x)
        return self.act(x) if self.use_act else x


class UpsampleBlock(nn.Module):
    """ Basic Upsample Block: Conv -> PS -> PReLU """
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)  # in_channels * 4, H, W --> in_channels, H*2, W*2
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """ Basic Residual Block: 2 x Conv Blocks """
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=False)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class Generator(nn.Module):
    """ Generator of SRGAN paper
    input  : N x in_ch x 24 x 24 (low resolution image)
    output : N x in_ch x 96 x 96 (high resolution image)
    """
    def __init__(self, img_channels, num_channels=64, num_blocks=16):
        """ 
        Parameters
        ----------
        img_channels : number of channels of image generated
        num_channels : number of channels in residual's convolutional blocks
        num_blocks   : number of repeated residual blocks
        """
        super().__init__()
        self.initial = ConvBlock(img_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 2))
        self.final = nn.Conv2d(num_channels, img_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        x = self.final(x)
        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    """ Discriminator of SRGAN paper
    input  : N x in_ch x 96 x 96 (high resolution image)
    output : N x 1 (probability)
    """
    def __init__(self, img_channels, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        """ 
        Parameters
        ----------
        img_channels   : number of channels of image generated
        features : features array for convolutional blocks
        """
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(ConvBlock(img_channels, feature, kernel_size=3, stride=1+idx%2, padding=1, use_act=True))
            img_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)), # doesnt do anything for 96 x 96
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    """ Super resolution x4: 96 x 96 -> 24 x 24 """
    low_res = 24
    img_channels = 3
    with torch.cuda.amp.autocast():
        x = torch.randn((5, img_channels, low_res, low_res))
        G = Generator(img_channels)
        D = Discriminator(img_channels)
        print("Generator output:     ", G(x).shape)
        print("Discriminator output: ", D(G(x)).shape)


