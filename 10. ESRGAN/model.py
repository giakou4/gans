import torch
from torch import nn


class ConvBlock(nn.Module):
    """ Basic Convolutional Block: Conv -> LReLU """
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        x = self.cnn(x)
        x = self.act(x)
        return x


class UpsampleBlock(nn.Module):
    """ Basic Convolutional Block: Upsample -> Conv -> LReLU """
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.act(x)
        return x


class ResidualDenseBlock(nn.Module):
    """ Residual Dense Block """
    def __init__(self, in_channels, gr_channels=32, b=0.2):
        """ 
        Parameters
        ----------
        in_channels : number of channels in input and final output
        gr_channels : growth chanels, intermediate features
        """
        super().__init__()
        self.b = b
        self.conv1 = ConvBlock(in_channels, gr_channels, kernel_size=3, stride=1, padding=1, use_act=True)
        self.conv2 = ConvBlock(in_channels + gr_channels, gr_channels, kernel_size=3, stride=1, padding=1, use_act=True)
        self.conv3 = ConvBlock(in_channels + 2 * gr_channels, gr_channels, kernel_size=3, stride=1, padding=1, use_act=True)
        self.conv4 = ConvBlock(in_channels + 3 * gr_channels, gr_channels, kernel_size=3, stride=1, padding=1, use_act=True)
        self.conv5 = ConvBlock(in_channels + 4 * gr_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=False)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.b + x


class RRDB(nn.Module):
    """ Residual-in-Residual Dense Block: 3 x RDB blocks """
    def __init__(self, in_channels, gr_channels=32, b=0.2):
        """ 
        Parameters
        ----------
        in_channels : number of channels of each RDB block
        gr_channels : growth chanels, intermediate features of each RDB block
        """
        super().__init__()
        self.b = b
        self.rdb1 = ResidualDenseBlock(in_channels, gr_channels)
        self.rdb2 = ResidualDenseBlock(in_channels, gr_channels)
        self.rdb3 = ResidualDenseBlock(in_channels, gr_channels)
    
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.b + x


class Generator(nn.Module):
    """ Generator of ESRGAN paper
    input  : N x img_channels x 24 x 24 (low resolution image)
    output : N x img_channels x 96 x 96 (high resolution image)
    """
    def __init__(self, img_channels, in_channels=64, num_blocks=23, gr_channels=32):
        """ 
        Parameters
        ----------
        img_channels : number of channels of image generated
        in_channels  : number of channels of RRDB block's input
        num_blocks   : number of repeated RRDB blocks
        gr_channels  : growth chanels, intermediate features of each RDB block
        """
        super().__init__()
        self.initial = nn.Conv2d(img_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.residuals = nn.Sequential(*[RRDB(in_channels, gr_channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.upsamples = nn.Sequential(UpsampleBlock(in_channels), UpsampleBlock(in_channels),)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, img_channels, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.conv(self.residuals(initial)) + initial
        x = self.upsamples(x)
        return self.final(x)


class Discriminator(nn.Module):
    """ Discriminator of SRGAN paper
    input  : N x img_channels x 96 x 96 (high resolution image)
    output : N x 1 (probability)
    """
    def __init__(self, img_channels, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        """ 
        Parameters
        ----------
        img_channels : number of channels of image generated
        features     : features array for convolutional blocks
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
    
    
def initialize_weights(model, scale=0.1):
    """ Initialize weights baed on WGAN paper """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale


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




