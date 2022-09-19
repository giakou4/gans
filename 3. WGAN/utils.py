import torch


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
