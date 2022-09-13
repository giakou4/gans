import argparse
from math import log2
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Discriminator, Generator
from utils import gradient_penalty, plot_to_tensorboard, save_checkpoint, load_checkpoint
torch.backends.cudnn.benchmarks = True


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser(description='Hyperparameters for training')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='base learning rate')
    parser.add_argument('--batch-sizes', type=list, default=[8, 8, 8, 4, 4, 4, 4, 2, 1], help='batch sizes')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
    parser.add_argument('--img-channels', type=int, default=3, help='images channels')
    parser.add_argument('--start-train-at-img-size', type=int, default=128, help='initial image size for training')
    parser.add_argument('--z-dim', type=int, default=256, help='noise dimension')
    parser.add_argument('--in-channels', type=int, default=256, help='input channels')
    parser.add_argument('--critic-iterations', type=int, default=1, help='critic iterations')
    parser.add_argument('--lambda-gp', type=float, default=10, help='gradient penalty lambda')   
    parser.add_argument('--load-model', action='store_true', help='load pre-trained model')
    parser.add_argument('--save-model', action='store_true', help='save model after each train epoch')
    parser.add_argument('--checkpoint-gen', type=str, default='./checkpoints/gen.pth.tar', help='path to save generators checkpoint')
    parser.add_argument('--checkpoint-critic', type=str, default='./checkpoints/critic.pth.tar', help='path to save critic checkpoint')
    parser.add_argument('--logs', type=str, default='./logs/', help='tensorflow logs directory')
    parser.add_argument('--dataset-dir', type=str, default='./data/train/', help='training dataset directory')
    opt = parser.parse_args()
    opt.progressive_epochs = [30] * len(opt.batch_sizes)
    opt.fixed_noise = torch.randn(8, opt.z_dim, 1, 1).to(opt.device)
    return opt


def get_loader(image_size, config):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.img_channels)],
                [0.5 for _ in range(config.img_channels)],
            ),
        ]
    )
    batch_size = config.batch_sizes[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.dataset_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    return loader, dataset


def train_one_epoch(dataset, loader, gen, critic, opt_critic, opt_gen, scaler_gen, scaler_critic, writer, epoch, num_epochs, step, alpha, tb_step, config):
    """ One forward pass of Discriminator and Generator """
    
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch {epoch}/{num_epochs}")
    
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        noise = torch.randn(cur_batch_size, config.z_dim, 1, 1).to(config.device)

        with torch.cuda.amp.autocast():
            
            fake = gen(noise, alpha, step)
            
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.device)

            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.lambda_gp * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / ((config.progressive_epochs[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        if batch_idx % 100 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.fixed_noise, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(writer, loss_critic.item(), loss_gen.item(), real.detach(),  fixed_fakes.detach(), tb_step)
            tb_step += 1

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
            loss_gen=loss_gen.item(),
        )

    return tb_step, alpha


def main(config):
    """ Training of Critic and Generator """
    
    gen = Generator(config.z_dim, config.img_channels, config.z_dim).to(config.device)
    critic = Discriminator(config.img_channels, config.z_dim).to(config.device)

    opt_gen = optim.Adam(gen.parameters(), lr=config.learning_rate, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.learning_rate, betas=(0.0, 0.99))
    
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(config.logs + "/images")

    if config.load_model:
        load_checkpoint(config.checkpoint_gen, gen, opt_gen, config.learning_rate, config.device)
        load_checkpoint(config.checkpoint_critic, critic, opt_critic, config.learning_rate, config.device)

    gen.train()
    critic.train()

    tb_step = 0
    step = int(log2(config.start_train_at_img_size / 4))
    
    for num_epochs in config.progressive_epochs[step:]:
        alpha = 1e-5  # start with very low alpha
        loader, dataset = get_loader(image_size=4*2**step, config=config)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        print(f"Current image size: {4 * 2 ** step} x {4 * 2 ** step}")

        for epoch in range(num_epochs):
            tb_step, alpha = train_one_epoch(dataset, loader, gen, critic, opt_critic, opt_gen, scaler_gen, scaler_critic, writer, epoch, config.num_epochs, step, alpha, tb_step, config)

            if config.save_model:
                save_checkpoint(gen, opt_gen, filename=config.checkpoint_gen)
                save_checkpoint(critic, opt_critic, filename=config.checkpoint_critic)


if __name__ == "__main__":
    config = parse_opt()
    main(config)
