import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """ Discriminator of a simple GAN
    input  : N x img_dim (flatten image 28 x 28)
    output : N x 1 (probability)
    """
    def __init__(self, img_dim):
        """ 
        Parameters
        ----------
        img_dim : number of channels of image generated
        """
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128), # 28 x 28 x 1 -> 784
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    """ Generator of a simple GAN
    input  : N x z_dim (noise)
    output : N x img_dim (flatten image 28 x 28)
    """
    def __init__(self, z_dim, img_dim):
        """ 
        Parameters
        ----------
        z_dim  : noise dimension of input
        img_dim : number of channels of image generated
        """
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim), # 28 x 28 x 1 -> 784
            nn.Tanh(),               # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


if __name__ == "__main__":
    img_dim = 784
    z_dim = 64
    z = torch.randn((5, z_dim))
    x = torch.randn((5, img_dim))
    G = Generator(z_dim, img_dim)
    D = Discriminator(img_dim)
    print("Generator output:     ", G(z).shape)
    print("Discriminator output: ", D(x).shape)
