import torch
import torch.nn.functional as F


def loss_fn_disc(D_H_real, D_H_fake, D_Z_real, D_Z_fake):
    """ 
    Cycle GAN Discriminator's Loss Function
    
    Parameters
    ----------
    D_H_real : D_H(horse)
    D_H_fake : D_H(G_H(zebra))
    D_Z_real : D_Z(zebra)
    D_Z_fake : DZ(G_Z(horse))
    
    Returns
    -------
    D_loss   : loss of Discriminator
    """
                       
    D_H_real_loss = F.mse_loss(D_H_real, torch.ones_like(D_H_real))
    D_H_fake_loss = F.mse_loss(D_H_fake, torch.zeros_like(D_H_fake))
    D_H_loss = D_H_real_loss + D_H_fake_loss
  
    D_Z_real_loss = F.mse_loss(D_Z_real, torch.ones_like(D_Z_real))
    D_Z_fake_loss = F.mse_loss(D_Z_fake, torch.zeros_like(D_Z_fake))
    D_Z_loss = D_Z_real_loss + D_Z_fake_loss

    D_loss = (D_H_loss + D_Z_loss)/2
    return D_loss
    
    
def loss_fn_gen(D_H_fake, D_Z_fake, zebra, horse, cycle_zebra, cycle_horse, identity_zebra, identity_horse, lambda_cycle=0, lambda_identity=0):
    """ 
    Cycle GAN Generators's Loss Function
    
    Parameters
    ----------
    D_H_fake       : D_H(G_H(zebra))
    D_Z_fake       : D_Z(G_Z(horse))
    cycle_zebra    : G_Z(G_H(zebra))
    cycle_horse    : G_H(G_Z(horse))
    identity_zebra : G_Z(zebra)
    identity_horse : G_H(horse)
    
    Returns
    -------
    G_loss         : loss of Generator
    """
    # Adversarial Loss for both Generators
    loss_G_H = F.mse_loss(D_H_fake, torch.ones_like(D_H_fake))
    loss_G_Z = F.mse_loss(D_Z_fake, torch.ones_like(D_Z_fake))

    # Cycle Loss (remove these for efficiency if you set lambda_cycle=0)
    cycle_zebra_loss = F.l1_loss(zebra, cycle_zebra)
    cycle_horse_loss = F.l1_loss(horse, cycle_horse)

    # Identity loss (remove these for efficiency if you set lambda_identity=0)
    identity_zebra_loss = F.l1_loss(zebra, identity_zebra)
    identity_horse_loss = F.l1_loss(horse, identity_horse)

    # add all togethor
    G_loss = (
                  loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * lambda_cycle
                + cycle_horse_loss * lambda_cycle
                + identity_horse_loss * lambda_identity
                + identity_zebra_loss * lambda_identity
            )
    return G_loss