import torch
import torch.nn.functional as F
from torchvision.models import vgg19


def loss_fn_disc(disc_fake, disc_real):
    """ 
    SRGAN Discriminator's Loss Function
    
    Parameters
    ----------
    disc_fake : D(G(z))
    disc_real : D(x)
    
    Returns
    -------
    loss_disc: loss of Discriminator: max log(D(x)) + log(1 - D(G(z)))
    """
    loss_disc_real = F.binary_cross_entropy_with_logits(disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real))
    loss_disc_fake = F.binary_cross_entropy_with_logits(disc_fake, torch.zeros_like(disc_fake))
    loss_disc = loss_disc_fake + loss_disc_real
    return loss_disc
    

def vgg_loss(input, target, device="cuda"):
    """
    VGG Loss for SRGAN
    
    Parameters
    ----------
    input  : input torch vector
    target : target torch vector
    device : device to transfer inputs
    
    Returns
    -------
    loss   : VGG loss
    """
    vgg = vgg19(pretrained=True).features[:36].eval().to(device)
    input.to(device)
    target.to(device)
    x_features = vgg(input)
    y_features = vgg(input)
    loss = F.mse_loss(x_features, y_features)
    return loss
    
    
def loss_fn_gen(disc_fake, fake, high_res):
    """ 
    ESRGAN Generator's Loss Function
    
    Parameters
    ----------
    disc_fake : D(G(low_res))
    fake      : G(low_res)
    high_res  : high resolution image
    
    Returns
    -------
    loss_gen: loss of Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
    """
    adversarial_loss = 1e-3 * F.binary_cross_entropy_with_logits(disc_fake, torch.ones_like(disc_fake))
    loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
    loss_gen = loss_for_vgg + adversarial_loss
    return loss_gen


