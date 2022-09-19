import torch
import torch.nn.functional as F


def loss_fn_disc(disc_fake, disc_real):
    """ 
    Parameters
    ----------
    disc_fake : D(G(z))
    disc_real : D(x)
    
    Returns
    -------
    loss_disc: loss of Discriminator: max log(D(x)) + log(1 - D(G(z)))
    """
    loss_disc_real = F.binary_cross_entropy(disc_real, torch.ones_like(disc_real))
    loss_disc_fake = F.binary_cross_entropy(disc_fake, torch.zeros_like(disc_fake))
    loss_disc = (loss_disc_real + loss_disc_fake) / 2
    return loss_disc
    
def loss_fn_gen(disc_fake):
    """ 
    Parameters
    ----------
    disc_fake : D(G(z))
    
    Returns
    -------
    loss_gen: loss of Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
    """
    loss_gen = F.binary_cross_entropy(disc_fake, torch.ones_like(disc_fake))
    return loss_gen