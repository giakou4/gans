import torch
import random
import numpy as np
import os
import torchvision


def plot_to_tensorboard(writer, loss_critic, loss_gen, real, fake, tensorboard_step):
    """ 
    Print losses occasionally and print to tensorboard
    
    Parameters
    ----------
    writer           : writer to use
    loss_critic      : critic's current loss
    loss_gen         : generator's current loss
    real             : real image
    fake             : generated image
    tensorboard_step : tensorboard step to use
    """
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    """
    Altered Gradient Penalty for ProGANs
    
    Parameters
    ----------
    critic     : model critic to calculate gradient penalty
    real       : real image
    fake       : fake image generated
    alpha      : alpha of epoch
    train_step : train step of epoch
    device     : device to calculate gradient penalty
    """
    batch_size, channels, height, width = real.shape
    beta = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, height, width).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """ 
    Saves checkpoint of current model
    
    Parameters
    ----------
    model     : model, either generator or discriminator
    optimizer : save model's optimzier
    filename  : path of model to be saved
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    """ 
    Loads checkpoint of model
    
    Parameters
    ----------
    checkpoint_file : load model in specific path
    model           : model, either generator or discriminator
    optimizer       : load model's optimzier
    lr              : set optimizer's learning rate
    device          : device where model is stored
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    """
    Seeds all python libraries
    
    Parameters
    ----------
    seed : seed to use
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False