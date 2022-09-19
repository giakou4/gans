import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Discriminator, Generator
from utils import save_checkpoint, load_checkpoint
from loss import loss_fn_disc, loss_fn_gen


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser(description='Hyperparameters for training')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument('--num-epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='base learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--z-dim', type=int, default=64, help='noise dimension')
    parser.add_argument('--image-dim', type=int, default=784, help='image dimension')
    parser.add_argument('--load-model', action='store_true', help='load pre-trained model')
    parser.add_argument('--save-model', action='store_true', help='save model after each train epoch')
    parser.add_argument('--checkpoint-gen', type=str, default='./checkpoints/gen.pth.tar', help='path to save generators checkpoint')
    parser.add_argument('--checkpoint-disc', type=str, default='./checkpoints/disc.pth.tar', help='path to save discriminators checkpoint')
    parser.add_argument('--logs', type=str, default='./logs/', help='tensorflow logs directory')
    parser.add_argument('--dataset-dir', type=str, default='./data/', help='dataset directory')
    opt = parser.parse_args()
    return opt


def train_one_epoch(loader, gen, disc, opt_gen, opt_disc, loss, tb_step, epoch, num_epochs, fixed_noise, writer_fake, writer_real, config):
    """ One forward pass of Discriminator and Generator """
    
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch {epoch}/{config.num_epochs}")
    
    for batch_idx, (real, _) in enumerate(loop):
        
        real = real.view(-1, 784).to(config.device)
        noise = torch.randn(config.batch_size, config.z_dim).to(config.device)
        fake = gen(noise)
    
        # Train Discriminator
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake).view(-1)
        loss_disc = loss_fn_disc(disc_fake, disc_real)
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator
        disc_fake = disc(fake).view(-1)
        loss_gen = loss_fn_gen(disc_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                real = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real, normalize=True)
                writer_fake.add_image("Fake Images", img_grid_fake, global_step=tb_step)
                writer_real.add_image("Real Images", img_grid_real, global_step=tb_step)
                tb_step += 1
            
        loop.set_postfix(loss_disc=loss_disc.item(), loss_gen=loss_gen.item())
        
    return tb_step


def main(config):
    """ Training of Discriminator and Generator """
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    dataset = datasets.MNIST(root=config.dataset_dir, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    gen = Generator(config.z_dim, config.image_dim).to(config.device)
    disc = Discriminator(config.image_dim).to(config.device)
    
    opt_gen = optim.Adam(gen.parameters(), lr=config.learning_rate)
    opt_disc = optim.Adam(disc.parameters(), lr=config.learning_rate)
    
    loss = nn.BCELoss()
   
    fixed_noise = torch.randn((config.batch_size, config.z_dim)).to(config.device)
    writer_fake = SummaryWriter(config.logs + "fake")
    writer_real = SummaryWriter(config.logs + "real")
    tb_step = 0
    
    if config.load_model:
        load_checkpoint(config.checkpoint_gen, gen, opt_gen, config.learning_rate, config.device)
        load_checkpoint(config.checkpoint_disc, disc, opt_disc, config.learning_rate, config.device)
        
    disc.train()
    gen.train()
        
    for epoch in range(config.num_epochs):
        tb_step = train_one_epoch(loader, gen, disc, opt_gen, opt_disc, loss, tb_step, epoch, config.num_epochs, fixed_noise, writer_fake, writer_real, config)
        
        if config.save_model:
            save_checkpoint(gen, opt_gen, filename=config.checkpoint_gen)
            save_checkpoint(disc, opt_disc, filename=config.checkpoint_disc)
            
            
if __name__ == "__main__":
    config = parse_opt()
    main(config)