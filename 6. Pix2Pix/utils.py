import torch
from torchvision.utils import save_image

def save_some_examples(gen, val_loader, epoch, folder, device):
    """
    Save generated examples of Pix2Pix
    
    Parameters
    ----------
    gen        : model generator
    val_loader : loader to use
    epoch      : epoch images are generated (includes in image's name)
    folder     : folder to store generated images
    device     : device to use
    """
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization
        save_image(y_fake, folder + f"/y_generated_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
    if epoch == 1:
        save_image(y * 0.5 + 0.5, folder + f"/y_target_{epoch}.png")
    gen.train()


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
