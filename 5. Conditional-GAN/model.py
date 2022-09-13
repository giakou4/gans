import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """ Discriminator of Conditional GAN paper
    input  : N x img_channels x 64 x 64 (image)
    output : N x 1 x 1 (probability)
    """
    def __init__(self, img_channels, num_classes, num_features=16, img_size=64):
        """ 
        Parameters
        ----------
        img_channels : number of channels of image generated
        num_classes  : number of classes
        num_features : number of features
        img_size     : size of image where hight=weight
        """
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels+1, num_features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            self._block(num_features,   num_features*2, 4, 2, 1),
            self._block(num_features*2, num_features*4, 4, 2, 1),
            self._block(num_features*4, num_features*8, 4, 2, 1),
            nn.Conv2d(num_features*8, 1, 4, 2, 0),
        )
        
        self.embed = nn.Embedding(num_classes, img_size*img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False,),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1) # N x channels x H x W
        return self.disc(x)


class Generator(nn.Module):
    """ Generator of Conditional GAN paper
    input  : N x z_dim x 1 x 1 (noise)
    output : N x img_channels x 64 x 64 (image)
    """
    def __init__(self, z_dim, img_channels, num_classes, num_features=16, img_size=64, embed_size=100):
        """ 
        Parameters
        ----------
        z_dim        : noise dimension of input
        img_channels : number of channels of image generated
        num_classes  : number of classes
        num_features : number of features
        img_size     : size of image where hight=weight
        embed_size   : size of embedding
        """
        super(Generator, self).__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            self._block(z_dim, num_features*16, 4, 1, 0), # img: 4 x 4
            self._block(num_features*16, num_features*8, 4, 2, 1),  # img: 8 x 8
            self._block(num_features*8,  num_features*4, 4, 2, 1),  # img: 16 x 16
            self._block(num_features*4,  num_features*2, 4, 2, 1),  # img: 32 x 32
            nn.ConvTranspose2d(num_features*2, img_channels, 4, 2, 1),
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
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
    G = Generator(z_dim, img_channels, 10)
    D = Discriminator(img_channels, 10)
    print("Generator output:     ", G(z).shape)
    print("Discriminator output: ", D(x).shape)
