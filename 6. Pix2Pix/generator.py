import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """ Basic Convolutional Block: Conv - BN - LReLU """
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(ConvBlock, self).__init__()      
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """ Generator of Pix2Pix paper - UNET with skip connections
    input  : N x img_channels x 256 x 256 (image)
    output : N x img_channels x 256 x 256 (image)
    """
    def __init__(self, img_channels, num_features=64):
        super().__init__()
        """ 
        Parameters
        ----------
        img_channels : number of channels of image
        num_features : number of features
        """
        nf = num_features
        self.initial_down = nn.Sequential(
            nn.Conv2d(img_channels, nf, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = ConvBlock(nf,   nf*2, down=True, act="leaky", use_dropout=False) # 64 x 64
        self.down2 = ConvBlock(nf*2, nf*4, down=True, act="leaky", use_dropout=False) # 32 x 32
        self.down3 = ConvBlock(nf*4, nf*8, down=True, act="leaky", use_dropout=False) # 16 x 16
        self.down4 = ConvBlock(nf*8, nf*8, down=True, act="leaky", use_dropout=False) # 8 x 8
        self.down5 = ConvBlock(nf*8, nf*8, down=True, act="leaky", use_dropout=False) # 4 x 4
        self.down6 = ConvBlock(nf*8, nf*8, down=True, act="leaky", use_dropout=False) # 2 x 2
        self.bottleneck = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, 4, 2, 1),                                       # 1 x 1
            nn.ReLU(),
            )

        self.up1 = ConvBlock(nf*8,   nf*8, down=False, act="relu", use_dropout=True)
        self.up2 = ConvBlock(nf*8*2, nf*8, down=False, act="relu", use_dropout=True)
        self.up3 = ConvBlock(nf*8*2, nf*8, down=False, act="relu", use_dropout=True)
        self.up4 = ConvBlock(nf*8*2, nf*8, down=False, act="relu", use_dropout=False)
        self.up5 = ConvBlock(nf*8*2, nf*4, down=False, act="relu", use_dropout=False)
        self.up6 = ConvBlock(nf*4*2, nf*2, down=False, act="relu", use_dropout=False)
        self.up7 = ConvBlock(nf*2*2, nf,   down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(nf * 2, img_channels, 4, 2, 1),
            nn.Tanh(), # each pixel in [-1, 1]
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        final_up = self.final_up(torch.cat([up7, d1], 1))
        return final_up


if __name__ == "__main__":
    img_channels = 3
    x = torch.randn((5, img_channels, 256, 256))
    G = Generator(img_channels)
    print("Generator output: ", G(x).shape)
