import torch
import torch.nn.functional as F


def loss_fn_critic(critic_fake, critic_real, gp, lambda_gp=0):
    """ 
    ProGAN Critic's Loss Function
    
    Parameters
    ----------
    critic_fake : C(G(z))
    critic_real : C(x)
    
    Returns
    -------
    loss_critic: loss of Critic: max E[C(real)] - E[C(fake)] <-> min -E[C(real)] + E[C(fake)]
    """           
    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp + (0.001 * torch.mean(critic_real ** 2))
    return loss_critic
    
    
def loss_fn_gen(gen_fake):
    """ 
    WGAN/WGAP-GP/Conditional GAN/Pro GAN Generator's Loss Function
    
    Parameters
    ----------
    gen_fake : C(G(z))
    
    Returns
    -------
    loss_gen: loss of Generator: max E[C(G(z))] <-> min -E[C(G(z))]
    """
    loss_gen = -torch.mean(gen_fake)
    return loss_gen
