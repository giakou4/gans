import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """ Discriminator of WGAN paper
    input  : N x img_channels x 64 x 64 (image)
    output : N x 1 x 1 (probability)
    """
    def __init__(self, img_channels, num_features=8):
        """ 
        Parameters
        ----------
        img_channels : number of channels of image generated
        num_features : number of features
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, num_features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(num_features,   num_features*2, 4, 2, 1), # img: 16 x 16
            self._block(num_features*2, num_features*4, 4, 2, 1), # img: 8 x 8
            self._block(num_features*4, num_features*8, 4, 2, 1), # img: 4 x 4
            nn.Conv2d(num_features*8, 1, 4, 2, 0),                # img: 1 x 1
        )

    def _block(self, img_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(img_channels, out_channels, kernel_size, stride, padding, bias=False,),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    """ Generator  of WGAN paper
    input  : N x z_dim x 1 x 1 (noise)
    output : N x img_channels x 64 x 64 (image)
    """
    def __init__(self, z_dim, img_channels, num_features=8):
        """ 
        Parameters
        ----------
        z_dim        : noise dimension of input
        img_channels : number of channels of image generated
        num_features : number of features
        """
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, num_features*16, 4, 1, 0),           # img: 4 x 4
            self._block(num_features*16, num_features*8, 4, 2, 1),  # img: 8 x 8
            self._block(num_features*8,  num_features*4, 4, 2, 1),  # img: 16 x 16
            self._block(num_features*4,  num_features*2, 4, 2, 1),  # img: 32 x 32
            nn.ConvTranspose2d(num_features*2, img_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def _block(self, img_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(img_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    """ Initializes weights according to the DCGAN paper """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":
    z_dim = 100
    img_channels = 3
    z = torch.randn((5, z_dim, 1, 1))
    x = torch.randn((5, img_channels, 64, 64))
    G = Generator(z_dim, img_channels, 64)
    D = Discriminator(img_channels, 64)
    print("Generator output:     ", G(z).shape)
    print("Discriminator output: ", D(x).shape)