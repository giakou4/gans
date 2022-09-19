import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
from dataset import MapDataset
from utils import save_checkpoint, load_checkpoint, save_some_examples
from loss import loss_fn_disc, loss_fn_gen
torch.backends.cudnn.benchmark = True


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser(description='Hyperparameters for training')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument('--num-epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='base learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
    parser.add_argument('--l1-lambda', type=float, default=100, help='L1 lambda')
    parser.add_argument('--lambda-gp', type=float, default=10, help='gradient penalty lambda')
    parser.add_argument('--load-model', action='store_true', help='load pre-trained model')
    parser.add_argument('--save-model', action='store_true', help='save model after each train epoch')
    parser.add_argument('--checkpoint-gen', type=str, default='./checkpoints/gen.pth.tar', help='path to save generators checkpoint')
    parser.add_argument('--checkpoint-critic', type=str, default='./checkpoints/critic.pth.tar', help='path to save discriminators checkpoint')
    parser.add_argument('--logs', type=str, default='./logs/', help='tensorflow logs directory')
    parser.add_argument('--train-dir', type=str, default='./data/maps/train/', help='training dataset directory')
    parser.add_argument('--val-dir', type=str, default='./data/maps/val/', help='validation dataset directory')
    opt = parser.parse_args()
    return opt


def train_one_epoch(disc, gen, loader, opt_disc, opt_gen, g_scaler, d_scaler, epoch, num_epochs, config):
    """ One forward pass of Discriminator and Generator """
    
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch {epoch}/{num_epochs}")

    for batch_idx, (x, y_target) in enumerate(loop):
        x = x.to(config.device)
        y_target = y_target.to(config.device)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            disc_real = disc(x, y_target)
            disc_fake = disc(x, y_fake)
            loss_disc = loss_fn_disc(disc_fake, disc_real)
        disc.zero_grad()
        d_scaler.scale(loss_disc).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            disc_fake = disc(x, y_fake)
            loss_gen = loss_fn_gen(disc_fake, y_fake, y_target, config.l1_lambda)
        opt_gen.zero_grad()
        g_scaler.scale(loss_gen).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix(loss_disc=loss_disc.item(), 
                         loss_gen=loss_gen.item(),
                         disc_real=torch.sigmoid(disc_real).mean().item(),
                         disc_fake=torch.sigmoid(disc_fake).mean().item(),
                         )


def main(config):
    """ Training of Discriminator and Generator for 3 x 256 x 256 images """
    
    both_transform = A.Compose([A.Resize(width=256, height=256),], additional_targets={"image0": "image"},)
    transform_only_input = A.Compose([A.ColorJitter(p=0.2), A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,), ToTensorV2(),])
    transform_only_mask = A.Compose([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,), ToTensorV2()])
    
    disc = Discriminator(img_channels=3).to(config.device)
    gen = Generator(img_channels=3).to(config.device)
    
    opt_disc = optim.Adam(disc.parameters(), lr=config.learning_rate, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))


    train_dataset = MapDataset(config.train_dir, both_transform, transform_only_input, transform_only_mask)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
       
    val_dataset = MapDataset(config.val_dir, both_transform, transform_only_input, transform_only_mask)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    if config.load_model:
        load_checkpoint(config.checkpoint_gen, gen, opt_gen, config.learning_rate,)
        load_checkpoint(config.checkpoint_disc, disc, opt_disc, config.learning_rate,)
        
    gen.train()
    disc.train()

    for epoch in range(config.num_epochs):
        train_one_epoch(disc, gen, train_loader, opt_disc, opt_gen, g_scaler, d_scaler, epoch, config.num_epochs, config)

        if config.save_model:
            save_checkpoint(gen, opt_gen, filename=config.checkpoint_gen)
            save_checkpoint(disc, opt_disc, filename=config.checkpoint_disc)
            
        save_some_examples(gen, val_loader, epoch, "evaluation", config.device)


if __name__ == "__main__":
    config = parse_opt()
    main(config)
