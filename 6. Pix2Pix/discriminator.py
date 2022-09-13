import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """ Basic Convolutional Block: Conv - BN - LReLU """
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    """ Discriminator  of Pix2Pix paper
    input  : N x img_channels x 256 x 256, N x img_channels x 256 x 256 (image x 2)
    output : N x 1 x 30 x 30 (probability in a grid - PatchGAN; each cell grid is responsible for a patch
    """
    def __init__(self, img_channels, num_features=[64, 128, 256, 512]):
        """ 
        Parameters
        ----------
        img_channels : number of channels of image
        num_features : number of features
        """
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels * 2, num_features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        img_channels = num_features[0]
        for feature in num_features[1:]:
            layers.append(ConvBlock(img_channels, feature, stride=1 if feature == num_features[-1] else 2))
            img_channels = feature
        layers.append(nn.Conv2d(img_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),)
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1) # x: target, y: generated (fake) or real 
        x = self.initial(x)
        x = self.model(x)
        return x


if __name__ == "__main__":
    img_channels = 3
    x = torch.randn((5, img_channels, 256, 256))
    y = torch.randn((5, img_channels, 256, 256))
    D = Discriminator(img_channels)
    print("Discriminator output: ", D(x, y).shape)

