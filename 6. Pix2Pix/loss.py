import torch
import torch.nn.functional as F


def loss_fn_disc(disc_fake, disc_real):
    """ 
    Pix2Pix Discriminator's Loss Function
    
    Parameters
    ----------
    disc_fake : D(G(z))
    disc_real : D(x)
    
    Returns
    -------
    loss_disc: loss of Discriminator
    """
    loss_disc_real = F.binary_cross_entropy_with_logits(disc_real, torch.ones_like(disc_real))
    loss_disc_fake = F.binary_cross_entropy_with_logits(disc_fake, torch.zeros_like(disc_fake))
    loss_disc = (loss_disc_real + loss_disc_fake) / 2
    return loss_disc
    
    
def loss_fn_gen(disc_fake, y_fake, y_target, l1_lambda=0):
    """ 
    Pix2Pix Generator's Loss Function
    
    Parameters
    ----------
    disc_fake : D(G(x))
    y_fake    : G(x)
    y_target  : real image
    l1_lambda : coefficient of L1 loss
    
    Returns
    -------
    loss_gen: loss of Generator
    """
    loss_gen_fake = F.binary_cross_entropy_with_logits(disc_fake, torch.ones_like(disc_fake))
    loss_l1 = F.mse_loss(y_fake, y_target)
    gen_loss = loss_gen_fake + loss_l1 * l1_lambda
    return gen_loss