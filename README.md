<p align="center">
  <img src="https://raw.githubusercontent.com/giakou4/gans/main/bg.jpg">
</p>

# Generative Adversarial Networks (GANs)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-python](https://img.shields.io/badge/Made%20with-PyTorch-red)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/gans/LICENSE)

The following GANs are implemented in [PyTorch](https://pytorch.org/):
1. DCGAN
2. WGAN and WGAN-GP
3. Conditional GAN
4. Pix2Pix
5. CycleGAN
6. ProGAN
7. SRGAN
8. ESRGAN

## Folder Structure

The structure of code is as follows:  
```bash
gan/  
├── logs
├── checkpoints
│   ├── disc.pth.tar  
│   └── gen.pth.tar  
├── data
│   ├── MNIST
│   ├── DIV2K_train_HR
│   └── ...
├── train.py  
├── model.py 
├── dataset.py 
├── loss.py 
├── utils.py 
└── README.md
```

## Code Organization

Each _model.py_ has the two following class implementations: 
```python 
class Discriminator(torch.nn.Module):
    """ Discriminator of XXX paper """
    def __init__(self, img_channels=3):
        pass
    def forward(self, x)
        return x

class Generator(torch.nn.Module):
    """ Generator of XXX paper """
    def __init__(self, img_channels=3, noise_dim=512):
        pass
    def forward(self, x)
        return x
```

In the _dataset.py_ we define, unless PyTorchs ```ImageFolder``` is fine, the class
```python
class MyImageFolder(torch.utils.data.Dataset):
    """ My image dataset """
    pass
```

We centralize the loss functions of Discriminator and Generator in the _loss.py_
```python
def loss_fn_disc():
    """ Discriminator Loss Function """
    pass
    
def loss_fn_gen():
    """ Generator Loss Function """
    pass
```

Each _train.py_ has an arguement parser, a function for single epoch training and the main function 
```python
def parse_opt():
    parser = argparse.ArgumentParser()
    # ...
    opt = parser.parse_args()
    return opt

def train_one_epoch(loader, gen, disc, opt_gen, opt_disc, loss, g_scaler, d_scaler, writer, tb_step, epoch, num_epochs, **kwargs):
  pass

def main(config):
  pass
  
if __name__ == "__main__":
    config = prase_opt()
    main(config)
```

Finally, in the _utils.py_, we define two basic functions: ```save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar")``` and ```load_checkpoint(checkpoint_file, model, optimizer, lr, device)``` among other essential for training.
