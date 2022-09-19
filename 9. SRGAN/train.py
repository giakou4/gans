import argparse
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import Generator, Discriminator
from dataset import MyImageFolder
from utils import load_checkpoint, save_checkpoint, plot_examples
from loss import loss_fn_disc, loss_fn_gen
torch.backends.cudnn.benchmark = True


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser(description='Hyperparameters for training')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='batch sizes')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers')
    parser.add_argument('--high-res-size', type=int, default=96, help='images high resolution')
    parser.add_argument('--low-res-size', type=int, default=96 // 4, help='images low resolution')  
    parser.add_argument('--load-model', action='store_true', help='load pre-trained model')
    parser.add_argument('--save-model', action='store_true', help='save model after each train epoch')
    parser.add_argument('--checkpoint-gen', type=str, default='./checkpoints/gen.pth.tar', help='path to save generators checkpoint')
    parser.add_argument('--checkpoint-disc', type=str, default='./checkpoints/disc.pth.tar', help='path to save discriminator checkpoint')
    parser.add_argument('--logs', type=str, default='./logs/', help='tensorflow logs directory')
    parser.add_argument('--dataset-dir', type=str, default='./data/', help='training dataset directory')
    opt = parser.parse_args()
    return opt


def train_one_epoch(loader, disc, gen, opt_gen, opt_disc, epoch, num_epochs, config, test_transform):
    """ One forward pass of Discriminator and Generator """
    
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch {epoch}/{num_epochs}")
    
    for batch_idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.device)
        low_res = low_res.to(config.device)
        
        # Train Discriminator
        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake)
        loss_disc = loss_fn_disc(disc_fake, disc_real)
        opt_disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator
        disc_fake = disc(fake)
        loss_gen = loss_fn_gen(disc_fake, fake, high_res)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0 and batch_idx > 0:
            plot_examples("test_images/", gen, "saved_images", config.device, test_transform)
            
        loop.set_postfix(loss_disc=loss_disc.item(), loss_gen=loss_gen.item())


def main(config):
    """ Training of Discriminator and Generator """
    
    t_tensor = ToTensorV2()
    t_norm = A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    highres_transform = A.Compose([t_norm, t_tensor])
    lowres_transform = A.Compose([A.Resize(width=config.low_res_size, height=config.low_res_size, interpolation=Image.BICUBIC), t_norm, t_tensor])
    both_transforms = A.Compose([A.RandomCrop(width=config.high_res_size, height=config.high_res_size), A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5)])
    test_transform = A.Compose([t_norm, t_tensor])
    
    dataset = MyImageFolder(config.dataset_dir, both_transforms, highres_transform, lowres_transform)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
    
    gen = Generator(img_channels=3).to(config.device)
    disc = Discriminator(img_channels=3).to(config.device)
    
    opt_gen  = optim.Adam(gen.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    
    if config.load_model:
        load_checkpoint(config.checkpoint_gen,  gen, opt_gen, config.learning_rate, config.device)
        load_checkpoint(config.checkpoint_disc, disc, opt_disc, config.learning_rate, config.device)

    for epoch in range(config.num_epochs):
        train_one_epoch(loader, disc, gen, opt_gen, opt_disc, epoch, config.num_epochs, config, test_transform)

        if config.save_model:
            save_checkpoint(gen, opt_gen,  filename=config.checkpoint_gen)
            save_checkpoint(disc, opt_disc, filename=config.checkpoint_disc)


if __name__ == "__main__":
    config = parse_opt()
    main(config)