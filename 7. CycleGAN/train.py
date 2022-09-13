import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from dataset import HorseZebraDataset
from utils import save_checkpoint, load_checkpoint


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser(description='Hyperparameters for training')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='base learning rate')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
    parser.add_argument('--lambda-identity', type=float, default=0.0, help='L1 lambda')
    parser.add_argument('--lambda-cycle', type=float, default=10, help='gradient penalty lambda')   
    parser.add_argument('--load-model', action='store_true', help='load pre-trained model')
    parser.add_argument('--save-model', action='store_true', help='save model after each train epoch')
    parser.add_argument('--checkpoint-gen-h', type=str, default='./checkpoints/genh.pth.tar', help='path to save generator 1 checkpoint')
    parser.add_argument('--checkpoint-gen-z', type=str, default='./checkpoints/genz.pth.tar', help='path to save generator 2 checkpoint')
    parser.add_argument('--checkpoint-disc-h', type=str, default='./checkpoints/critich.pth.tar', help='path to save discriminator 1 checkpoint')
    parser.add_argument('--checkpoint-disc-z', type=str, default='./checkpoints/criticz.pth.tar', help='path to save discriminator 2 checkpoint')
    parser.add_argument('--logs', type=str, default='./logs/', help='tensorflow logs directory')
    parser.add_argument('--train-dir', type=str, default='./data/horse2zebra/train/', help='training dataset directory')
    parser.add_argument('--val-dir', type=str, default='./data/horse2zebra/test/', help='validation dataset directory')
    opt = parser.parse_args()
    return opt


def train_one_epoch(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler, epoch, num_epochs, config):
    """ One forward pass of Discriminators and Generators """
    
    H_reals, H_fakes = 0, 0
    
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch {epoch}/{num_epochs}")

    for batch_idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.device)
        horse = horse.to(config.device)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            # Discriminator Horse
            fake_horse = gen_H(zebra)
            
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            
            D_H_real_loss = MSE(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = MSE(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            # Discriminator Zebra
            fake_zebra = gen_Z(horse)
            
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            
            D_Z_real_loss = MSE(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = MSE(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            
            # Adversarial Loss for both Generators
            loss_G_H = MSE(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = MSE(D_Z_fake, torch.ones_like(D_Z_fake))

            # Cycle Loss (remove these for efficiency if you set lambda_cycle=0)
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = L1(zebra, cycle_zebra)
            cycle_horse_loss = L1(horse, cycle_horse)

            # Identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = L1(zebra, identity_zebra)
            identity_horse_loss = L1(horse, identity_horse)

            # add all togethor
            G_loss = (
                  loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.lambda_cycle
                + cycle_horse_loss * config.lambda_cycle
                + identity_horse_loss * config.lambda_identity
                + identity_zebra_loss * config.lambda_identity
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if batch_idx % 100 == 0:
            save_image(fake_horse*0.5+0.5, f"evaluation/fake_horse_{epoch}_{batch_idx}.png")
            save_image(fake_zebra*0.5+0.5, f"evaluation/fake_zebra_{epoch}_{batch_idx}.png")

        loop.set_postfix(H_real=H_reals/(batch_idx+1), 
                         H_fake=H_fakes/(batch_idx+1),
                         D_loss=D_loss.item(),
                         G_loss=G_loss.item(),
                         )


def main(config):
    """ Training of Discriminators and Generators for 3 x 256 x 256 images """
    
    transform = A.Compose([A.Resize(width=256, height=256), A.HorizontalFlip(p=0.5),  A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255), ToTensorV2()], additional_targets={"image0": "image"},)
    
    disc_H = Discriminator(img_channels=3).to(config.device) # Classifying image of Horses
    disc_Z = Discriminator(img_channels=3).to(config.device) # Classifying image of Zebras
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.device) # Takes an image and generates a Zebra
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.device) # Takes an image and generates a Horse
    
    opt_disc = optim.Adam(list(disc_H.parameters()) + list(disc_Z.parameters()), lr=config.learning_rate, betas=(0.5, 0.999))
    opt_gen = optim.Adam(list(gen_Z.parameters()) + list(gen_H.parameters()), lr=config.learning_rate, betas=(0.5, 0.999))

    L1 = nn.L1Loss()   # for Cycle Consistency and Identity Loss
    MSE = nn.MSELoss() # for Adverserial Loss

    if config.load_model:
        load_checkpoint(config.checkpoint_gen_h, gen_H, opt_gen, config.learning_rate, config.device)
        load_checkpoint(config.checkpoint_gen_z, gen_Z, opt_gen, config.learning_rate, config.device)
        load_checkpoint(config.checkpoint_disc_h, disc_H, opt_disc, config.learning_rate, config.device)
        load_checkpoint(config.checkpoint_disc_z, disc_Z, opt_disc, config.learning_rate, config.device)

    train_dataset = HorseZebraDataset(root_horse=config.train_dir+"/trainA", root_zebra=config.train_dir+"/trainB", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    val_dataset = HorseZebraDataset(root_horse=config.val_dir+"/testA", root_zebra=config.val_dir+"/testB", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.num_epochs):
        train_one_epoch(disc_H, disc_Z, gen_Z, gen_H, train_loader, opt_disc, opt_gen, L1, MSE, d_scaler, g_scaler, epoch, config.num_epochs, config)

        if config.save_model:
            save_checkpoint(gen_H, opt_gen, filename=config.checkpoint_gen_h)
            save_checkpoint(gen_Z, opt_gen, filename=config.checkpoint_gen_z)
            save_checkpoint(disc_H, opt_disc, filename=config.checkpoint_disc_h)
            save_checkpoint(disc_Z, opt_disc, filename=config.checkpoint_disc_z)

if __name__ == "__main__":
    config = parse_opt()
    main(config)