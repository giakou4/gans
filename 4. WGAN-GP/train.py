import argparse
import torch
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from loss import loss_fn_critic, loss_fn_gen


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser(description='Hyperparameters for training')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument('--num-epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--z-dim', type=int, default=100, help='noise dimension')
    parser.add_argument('--img-size', type=int, default=64, help='image size')
    parser.add_argument('--img-channels', type=int, default=1, help='images channels')
    parser.add_argument('--features-critic', type=int, default=64, help='discriminators features')
    parser.add_argument('--features-gen', type=int, default=64, help='generators features')
    parser.add_argument('--critic-iterations', type=int, default=5, help='generators features')
    parser.add_argument('--lambda-gp', type=int, default=10, help='generators features')
    parser.add_argument('--load-model', action='store_true', help='load pre-trained model')
    parser.add_argument('--save-model', action='store_true', help='save model after each train epoch')
    parser.add_argument('--checkpoint-gen', type=str, default='./checkpoints/gen.pth.tar', help='path to save generators checkpoint')
    parser.add_argument('--checkpoint-critic', type=str, default='./checkpoints/disc.pth.tar', help='path to save discriminators checkpoint')
    parser.add_argument('--logs', type=str, default='./logs/', help='tensorflow logs directory')
    parser.add_argument('--dataset-dir', type=str, default='./data/', help='dataset directory')
    opt = parser.parse_args()
    return opt


def train_one_epoch(loader, gen, critic, opt_gen, opt_critic, tb_step, epoch, num_epochs, fixed_noise, writer_fake, writer_real, config):
    """ One forward pass of Critic and Generator """
    
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch {epoch}/{num_epochs}")
    
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.device)
        cur_batch_size = real.shape[0]

        # Train Critic
        for _ in range(config.critic_iterations):
            noise = torch.randn(cur_batch_size, config.z_dim, 1, 1).to(config.device)
            fake = gen(noise)
            
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=config.device)
            loss_critic = loss_fn_critic(critic_fake, critic_real, gp, config.lambda_gp)
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator
        gen_fake = critic(fake).reshape(-1)
        loss_gen = loss_fn_gen(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0 and batch_idx > 0:
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=tb_step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=tb_step)
                tb_step += 1
        
        loop.set_postfix(loss_critic=loss_critic.item(), loss_gen=loss_gen.item())
        
    return tb_step
    
    
def main(config):
    """ Training of Critic and Generator """
    
    t1 = transforms.Resize(config.img_size)
    t2 = transforms.ToTensor()
    t3 = transforms.Normalize([0.5 for _ in range(config.img_channels)], [0.5 for _ in range(config.img_channels)])
    transform = transforms.Compose([t1, t2, t3])
    
    dataset = datasets.MNIST(root=config.dataset_dir, transform=transform, download=True)
    #dataset = datasets.ImageFolder(root="./data/celeb_dataset", transform=transform)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    gen = Generator(config.z_dim, config.img_channels, config.features_gen).to(config.device)
    critic = Discriminator(config.img_channels, config.features_critic).to(config.device)
    
    initialize_weights(gen)
    initialize_weights(critic)

    opt_gen = optim.Adam(gen.parameters(), lr=config.learning_rate, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=config.learning_rate, betas=(0.0, 0.9))

    fixed_noise = torch.randn(32, config.z_dim, 1, 1).to(config.device)
    writer_fake = SummaryWriter(config.logs + "fake")
    writer_real = SummaryWriter(config.logs + "real")
    tb_step = 0
    
    if config.load_model:
        load_checkpoint(config.checkpoint_gen, gen, opt_gen, config.learning_rate, config.device)
        load_checkpoint(config.checkpoint_critic, critic, opt_critic, config.learning_rate, config.device)

    gen.train()
    critic.train()

    for epoch in range(config.num_epochs):
        tb_step = train_one_epoch(loader, gen, critic, opt_gen, opt_critic, tb_step, epoch, config.num_epochs, fixed_noise, writer_fake, writer_real, config)
        
        if config.save_model:
            save_checkpoint(gen, opt_gen, filename=config.checkpoint_gen)
            save_checkpoint(critic, opt_critic, filename=config.checkpoint_critic)


if __name__ == "__main__":
    config = parse_opt()
    main(config)