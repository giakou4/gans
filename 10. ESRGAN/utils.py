import torch
import os
import numpy as np
from PIL import Image
from torchvision.utils import save_image

    
def gradient_penalty(critic, real, fake, device="cpu"):
    """
    Gradient Penalty
    
    Parameters
    ----------
    critic : model critic to calculate gradient penalty
    real   : real image
    fake   : fake image generated
    device : device to calculate gradient penalty
    """
    batch_size, channels, height, width = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, height, width).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    interpolated_images.requires_grad_(True)
    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

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


def plot_examples(low_res_folder, gen, gen_folder, device, test_transform):
    """
    Save generated examples of Pix2Pix
    
    Parameters
    ----------
    low_res_folder  : folder where some low resolution images are stored to evaluate generator
    gen             : model generator
    gen_folder      : folder where generated images are saved
    device          : device to use
    test_transform  : test transform to use
    """
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = Image.open(low_res_folder + file)
        with torch.no_grad():
            upscaled_img = gen(test_transform(image=np.asarray(image))["image"].unsqueeze(0).to(device))
        save_image(upscaled_img * 0.5 + 0.5, f"{low_res_folder}/{file}")
    gen.train()


