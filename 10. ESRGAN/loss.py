import torch
import torch.nn.functional as F
from torchvision.models import vgg19


def loss_fn_disc(disc_fake, disc_real, lambda_gp, gp):
    """ 
    ESRGAN Discriminator's Loss Function
    
    Parameters
    ----------
    disc_fake : D(G(z))
    disc_real : D(x)
    lambda_gp : coefficient of gradient penalty
    gp        : gradient penalty
    
    Returns
    -------
    loss_disc: loss of Discriminator: max log(D(x)) + log(1 - D(G(z)))
    """
    loss_disc = (-(torch.mean(disc_real) - torch.mean(disc_fake)) + lambda_gp * gp)
    return loss_disc
    
        
def vgg_loss(input, target, device="cuda"):
    """
    VGG Loss for ESRGAN 
    
    Parameters
    ----------
    input  : input torch vector
    target : target torch vector
    device : device to transfer inputs
    
    Returns
    -------
    loss   : VGG loss
    """
    vgg = vgg19(pretrained=True).features[:35].eval().to(device)
    input.to(device)
    target.to(device)
    x_features = vgg(input)
    y_features = vgg(input)
    loss = F.mse_loss(x_features, y_features)
    return loss
    
    
def loss_fn_gen(disc_fake, fake, high_res):
    """ 
    Parameters
    ESRGAN Generator's Loss Function
    
    ----------
    disc_fake : D(G(low_res))
    fake      : G(low_res)
    high_res  : high resolution image
    
    Returns
    -------
    loss_gen: loss of Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
    """
    l1_loss = 1e-2 * F.l1_loss(fake, high_res)
    adversarial_loss = 5e-3 * -torch.mean(disc_fake)
    loss_for_vgg = vgg_loss(fake, high_res)
    loss_gen = l1_loss + loss_for_vgg + adversarial_loss
    return loss_gen
