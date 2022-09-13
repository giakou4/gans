"""
Implementation of ProGAN generator and discriminator with the key
attributions from the paper. We have tried to make the implementation
compact but a goal is also to keep it readable and understandable.
Specifically the key points implemented are:

1) Progressive growing (of model and layers)
2) Minibatch std on Discriminator
3) Normalization with PixelNorm
4) Equalized Learning Rate (here I cheated and only did it on Conv layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

"""
Factors is used in Discrmininator and Generator for how much
the channels should be multiplied and expanded for each layer,
so specifically the first 5 layers the channels stay the same,
whereas when we increase the img_size (towards the later layers)
we decrease the number of chanels by 1/2, 1/4, etc.
"""

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]


class WSConv2d(nn.Module):
    """
    Weight scaled Conv2d (Equalized Learning Rate)
    Note that input is multiplied rather than changing weights
    this will have the same result.

    Inspired and looked at:
    https://github.com/nvnbny/progressive_growing_of_gans/blob/master/modelUtils.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * (kernel_size**2))) ** 0.5
        self.bias = self.conv.bias # Save bias
        self.conv.bias = None      # Remove bias

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
        return x


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        x = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
        return x


class ConvBlock(nn.Module):
    """ Basic Convolutional Block: Conv - LReLU - PN - Conv - LReLU - PN """
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        
        self.use_pn = use_pixelnorm
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.net = nn.Sequential(
            WSConv2d(in_channels, out_channels),
            nn.LeakyReLU(0.2),
            PixelNorm() if self.use_pn else nn.Identity(),
            WSConv2d(out_channels, out_channels),
            nn.LeakyReLU(0.2),
            PixelNorm() if self.use_pn else nn.Identity(),
            )
        
    def forward(self, x):
        x = self.net(x)
        return x


class Generator(nn.Module):
    """ Generator of Conditional GAN paper
    input  : N x z_dim x 1 x 1 (noise)
    output : N x img_channels x img_size x img_size (image, img_size depends on steps)
    """
    def __init__(self, z_dim, img_channels, in_channels=512):
        """ 
        Parameters
        ----------
        z_dim        : noise dimension of input
        img_channels : number of channels of image generated
        in_channels  : number of input channels in convolutional layer
        """
        super(Generator, self).__init__()

        self.initial_block = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_toRGB = WSConv2d(in_channels, img_channels, 1, 1, 0)
        
        self.prog_blocks = nn.ModuleList([])
        self.toRGB_blocks = nn.ModuleList([self.initial_toRGB])

        for i in range(len(factors) - 1):
            # For each ProgBlock, there is an RGB Block that convolves the output channels of ProgBlock to 3
            conv_in_c = int(in_channels * factors[i])       # 1, 1, 1, 1,   1/2, 1/4, 1/8, 1/16
            conv_out_c = int(in_channels * factors[i + 1])  # 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.toRGB_blocks.append(WSConv2d(conv_out_c, img_channels, 1, 1, 0))

    def fade_in(self, alpha, x, y):
        assert x.shape == y.shape
        assert alpha <= 1 and alpha >=0
        return torch.tanh(alpha * y + (1 - alpha) * x)

    def forward(self, x, alpha, steps): 
        #  steps=0 (4x4), steps=1 (8x8), steps=2 (16x16)
        out = self.initial_block(x)

        if steps == 0:
            out = self.initial_toRGB(out)
            return out

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest") # multiplies spatial size by 2
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.toRGB_blocks[steps - 1](upscaled)
        final_out = self.toRGB_blocks[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


class Discriminator(nn.Module):
    """ Discriminator of ProGAN paper
    input  : N x img_channels x img_size x img_size (image, img_size depends on steps)
    output : N x 1 x 1 (probability)
    """
    def __init__(self, img_channels, in_channels=512):
        """ 
        Parameters
        ----------
        img_channels : number of channels of image generated
        in_channels  : number of input channels in convolutional layer
        """
        super(Discriminator, self).__init__()
        
        self.prog_blocks = nn.ModuleList([])
        self.toRGB_blocks = nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.toRGB_blocks.append(WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0))

        self.initial_toRGB = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.toRGB_blocks.append(self.initial_toRGB)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # down sampling using avg pool

        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),  # we use this instead of linear layer
        )

    def fade_in(self, alpha, x, y):
        """Used to fade in downscaled using avg pooling and output from CNN"""
        assert x.shape == y.shape
        assert alpha <= 1 and alpha >=0
        return alpha * y + (1 - alpha) * x

    def minibatch_std(self, x):
        """ 
        We take the std for each example (across all channels, and pixels) 
        then we repeat it for a single channel and concatenate it with the 
        image. In this way the discriminator will get information about
        the variation in the batch/image
        """
        batch_statistics = (torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]))
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps

        out = self.leaky(self.toRGB_blocks[cur_step](x))

        if steps == 0:  # i.e, image is 4x4
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.avg_pool(x)
        downscaled = self.leaky(self.toRGB_blocks[cur_step + 1](downscaled))
        
        out = self.prog_blocks[cur_step](out)
        out = self.avg_pool(out)
        
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
            
        out = self.minibatch_std(out)
        
        out = self.final_block(out)
        return out.view(out.shape[0], -1)


if __name__ == "__main__":
    z_dim = 512
    img_channels = 3
    G = Generator(z_dim, img_channels)
    D = Discriminator(img_channels)
    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        z = torch.randn((1, z_dim, 1, 1))
        x = torch.randn((1, img_channels, img_size, img_size))
        print("Generator output:     ", G(z, 0.5, num_steps).shape)
        print("Discriminator output: ", D(x, 0.5, num_steps).shape)
        print(f"Success! At img size: {img_size} \n")
